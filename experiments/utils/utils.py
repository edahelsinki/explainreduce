"""Contains functions used for reproducing the experiments in the paper."""

import sys
import os

# Get the absolute path of the directory containing explainreduce
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add to sys.path
sys.path.append(PROJECT_DIR)

import torch
from sklearn.model_selection import train_test_split
from functools import partial
import explainreduce.localmodels as lm
import explainreduce.proxies as px
import explainreduce.metrics as metrics
from explainreduce.utils import read_parquet
import numpy as np
from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
import pandas as pd
import operator
from functools import reduce
from experiments.utils.hyperparameters import get_bb, get_data, get_params
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, Tuple


OUTPUT_DIR = RESULTS_DIR / "k_sensitivity"


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "proxy_method_mod"] = "None"
    df = df.loc[~df["proxy_method"].isna()]
    df.loc[df["proxy_method"].str.contains("minimal_set_cov"), "proxy_method_mod"] = (
        "Min set"
    )
    df.loc[df["proxy_method"].str.contains("max_coverage"), "proxy_method_mod"] = (
        "Max coverage"
    )
    df.loc[
        df["proxy_method"].str.contains("greedy_max_coverage"), "proxy_method_mod"
    ] = "Greedy Max coverage"
    df.loc[df["proxy_method"].str.contains("random"), "proxy_method_mod"] = "Random"
    df.loc[df["proxy_method"].str.contains("min_loss"), "proxy_method_mod"] = "Min loss"
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_[0-9]+"), "proxy_method_mod"
    ] = "Greedy Min loss (fixed k)"
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_[0-9]+_min_cov"),
        "proxy_method_mod",
    ] = "Greedy Min loss (minimum coverage)"
    df.loc[df["proxy_method"].str.contains("B"), "proxy_method_mod"] = "B K-means"
    df.loc[df["proxy_method"].str.contains("L"), "proxy_method_mod"] = "L K-means"
    df.loc[df["proxy_method"].str.contains("X"), "proxy_method_mod"] = "X K-means"
    return df


def prepare_data(dataset_fun, seed, n_total=10_000, n_exp=500):

    X, y, bb, cls = dataset_fun()

    # Limit the data size to a maximum of 10,000 samples
    n = min(n_total, X.shape[0])
    X, y = X[:n, :], y[:n]

    # Ensure y has at least 2 dimensions
    if y.ndim < 2:
        y = y[:, None]

    # Split the data into training and testing sets, with a 50/50 split
    if cls:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42, stratify=y[:, 1]
        )
        # Further split the training to get a smaller training subset for experiments
        X_exp_train, _, y_exp_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_exp,
            random_state=seed,
            stratify=y_train[:, 1],
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=seed
        )
        # Further split the training to get a smaller training subset for experiments
        X_exp_train, _, y_exp_train, _ = train_test_split(
            X_train, y_train, train_size=n_exp, random_state=seed
        )

    # Convert all data arrays into PyTorch tensors
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    X_exp_train = torch.as_tensor(X_exp_train, dtype=torch.float32)
    X_test = torch.as_tensor(X_test, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    y_exp_train = torch.as_tensor(y_exp_train, dtype=torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.float32)

    # Organize the processed data and additional objects into a dictionary
    out = {
        "bb_train": (X_train, y_train),
        "exp_train": (X_exp_train, y_exp_train),
        "test": (X_test, y_test),
        "black_box": bb,
        "classifier": cls,
    }
    return out


def get_explainer(dname, bb, cls):
    """Define the explainers to be used in the experiments."""
    return {
        "SLISEMAP": partial(
            lm.SLISEMAPExplainer, classifier=cls, **get_params("SLISEMAP", dname, bb)
        ),
        "SLIPMAP": partial(
            lm.SLIPMAPExplainer, classifier=cls, **get_params("SLIPMAP", dname, bb)
        ),
        "SmoothGrad": partial(
            lm.SmoothGradExplainer,
            classifier=cls,
            **get_params("SmoothGrad", dname, bb),
        ),
        "LIME": partial(
            lm.LIMEExplainer, classifier=cls, **get_params("LIME", dname, bb)
        ),
        "SHAP": partial(
            lm.KernelSHAPExplainerLegacy,
            classifier=cls,
        ),
    }


def get_explanation_radius(explainer: lm.Explainer):
    if "LIMEExplainer" in str(explainer.__class__):
        return 2 * explainer.explainer_kwargs["kernel_width"]
    elif "SLISEMAP" in str(explainer.__class__) or "SLIPMAP" in str(
        explainer.__class__
    ):
        return 0.1 * explainer.explainer_kwargs["radius"]
    elif "Smoothgrad" in str(explainer.__class__):
        return explainer.explainer_kwargs["noise_level"]
    else:
        return 0.5


def get_optimisation_method(k: int, explainer: lm.Explainer):
    """Define the reducing methods to be used in the experiments."""
    return {
        (f"greedy_max_coverage_k_{k}", k): partial(px.find_proxies_greedy, k=k),
        (f"max_coverage_k_{k}", k): partial(
            px.find_proxies_coverage, k=k, time_limit=100, pulp_msg=False
        ),
        (f"random_k_{k}", k): partial(px.find_proxies_random, k=k),
        (f"greedy_min_loss_{k}", k): partial(px.find_proxies_greedy_k_min_loss, k=k),
        (f"greedy_min_loss_{k}_min_cov", k): partial(
            px.find_proxies_greedy_min_loss_k_min_cov,
            k=k,
            min_coverage=0.8,
            raise_on_infeasible=False,
        ),
        (f"clustering_kmeans_B_cosine_{k}", k): partial(
            px.find_proxies_clustering_wrapper,
            data="local_models",
            n_clusters=k,
            method="kmeans",
            dist="cosine",
        ),
        (f"clustering_kmeans_L_euclidean_{k}", k): partial(
            px.find_proxies_clustering_wrapper,
            data="loss",
            n_clusters=k,
            method="kmeans",
            dist="euclidean",
        ),
        (f"clustering_kmeans_X_euclidean_{k}", k): partial(
            px.find_proxies_clustering_wrapper,
            data="data",
            n_clusters=k,
            method="kmeans",
            dist="euclidean",
        ),
        (f"submodular_pick_{k}", k): partial(px.find_proxies_submodular_pick, k=k),
        (f"max_ball_coverage_{k}", k): partial(
            px.find_proxies_max_cov_ball_cover,
            k=k,
            radius=get_explanation_radius(explainer),
            time_limit=5000,
        ),
        (f"glocalx_{k}", k): partial(px.find_proxies_glocalx, k=k),
    }


def subsample_explainer(explainer: lm.Explainer, n: int) -> lm.Explainer:
    n_models = len(explainer.local_models)
    assert n <= n_models, f"Cannot pick {n} proxies from {n_models} local models!"
    proxies = sorted(list([x.item() for x in torch.randperm(n_models)[:n]]))
    reduced_models = [explainer.local_models[i] for i in proxies]
    mapping = {i: i for i in range(len(proxies))}
    vector_representation = explainer.vector_representation[proxies, :]
    reduced = explainer.clone_reduced(reduced_models, mapping, vector_representation)
    reduced.X = reduced.X[proxies, :]
    reduced.y = reduced.y[proxies, :]
    # pre-recalculate L
    reduced.L = None
    reduced.get_L()
    return reduced


def eval_proxy_method_subsample_sensitivity(
    job_id,
    dname,
    expname,
    explainer,
    pname,
    proxy_method,
    n,
    epsilon,
    X,
    X_test,
    y,
    y_test,
    yhat_test,
    loss_fn,
    L,
    full_fidelity,
    full_loss,
    bb_loss,
    nn,
):
    print(
        f"Reducing: {dname} - {expname} - {pname} - n={n}",
        flush=True,
    )
    try:
        sampled_explainer = subsample_explainer(explainer, n)
        proxies = proxy_method(sampled_explainer, epsilon=epsilon)
    except ValueError as ve:
        print(f"Reduction method resulted in error:\n", ve, flush=True)
        return
    sub_yhat_train = sampled_explainer.predict(X)
    sub_yhat_test = sampled_explainer.predict(X_test)
    sub_loss_train = loss_fn(torch.as_tensor(y), sub_yhat_train)
    sub_loss_test = loss_fn(torch.as_tensor(y_test), sub_yhat_test)
    prx_yhat_train = proxies.predict(X)
    prx_yhat_test = proxies.predict(X_test)
    loss_train = loss_fn(torch.as_tensor(y), prx_yhat_train)
    loss_test = loss_fn(torch.as_tensor(y_test), prx_yhat_test)
    sub_L = sampled_explainer.get_L(X, y)
    reduced_L = proxies.get_L(X, y)
    # expanded_L = reduced_L[list(proxies.mapping.values()), :]
    sub_fidelity = loss_fn(torch.as_tensor(yhat_test), sub_yhat_test).mean().item()
    proxy_fidelity = loss_fn(torch.as_tensor(yhat_test), prx_yhat_test).mean().item()
    # epsilon = torch.quantile(L, 0.33).item()

    res = dict(
        job=job_id,
        data=dname,
        exp_method=expname,
        proxy_method=pname,
        k=len(proxies.local_models),
        n=n,
        epsilon=epsilon,
        full_fidelity=full_fidelity,
        full_stability=L[torch.arange(L.shape[0])[:, None], nn].mean().item(),
        full_coverage=metrics.calculate_coverage(L < epsilon),
        full_loss_test=full_loss,
        bb_loss_test=bb_loss,
        loss_train=loss_train.mean().item(),
        loss_test=loss_test.mean().item(),
        sub_loss_train=sub_loss_train.mean().item(),
        sub_loss_test=sub_loss_test.mean().item(),
        proxy_fidelity=proxy_fidelity,
        proxy_coverage=metrics.calculate_coverage(reduced_L < epsilon),
        sub_fidelity=sub_fidelity,
        sub_coverage=metrics.calculate_coverage(sub_L < epsilon),
        # proxy_stability=reduced_L[torch.arange(reduced_L.shape[0])[:, None], nn]
        # proxy_stability=expanded_L[torch.arange(expanded_L.shape[0])[:, None], nn]
        # .mean()
        # .item(),
    )
    return res


def plot_pyramid():
    x = np.linspace(-3, 3, 50)
    y = np.cos(x)
    y_train = y + np.random.normal(loc=0.0, scale=0.15, size=y.shape)
    sm = lm.SLISEMAPExplainer(x[:, None], y_train, lasso=0)
    sm.fit()
    epsilon = torch.quantile(sm.get_L(), q=0.5).item()
    proxies = px.find_proxies_coverage(sm, k=2, epsilon=epsilon)
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    sns.scatterplot(x=x, y=y_train, ax=ax[0])
    sns.lineplot(x=x, y=y, ax=ax[0])
    positive_mask = sm.vector_representation[:, 0] > 0
    sns.scatterplot(x=x[positive_mask], y=y_train[positive_mask], ax=ax[1])
    for i, p in enumerate(positive_mask):
        if p:
            ax[1].plot(x, sm.local_models[i](x[:, None]), c="tab:blue", linestyle="--")
        else:
            ax[1].plot(
                x, sm.local_models[i](x[:, None]), c="tab:orange", linestyle="--"
            )
    sns.scatterplot(x=x[~positive_mask], y=y_train[~positive_mask], ax=ax[1])
    ax[1].set_ylim(ax[0].get_ylim())
    positive_mask = proxies.mapped_vector_representation()[:, 0] > 0
    errorbars = lambda x: (x - epsilon, x + epsilon)
    lower1, upper1 = errorbars(proxies.local_models[0](x[:, None])[:, 0])
    lower2, upper2 = errorbars(proxies.local_models[1](x[:, None])[:, 0])
    sns.scatterplot(x=x[positive_mask], y=y_train[positive_mask], ax=ax[2])
    sns.scatterplot(x=x[~positive_mask], y=y_train[~positive_mask], ax=ax[2])
    sns.lineplot(x=x, y=proxies.local_models[0](x[:, None])[:, 0], ax=ax[2])
    ax[2].plot(x, lower1, color="tab:blue", alpha=0.1)
    ax[2].plot(x, upper1, color="tab:blue", alpha=0.1)
    ax[2].fill_between(x, lower1, upper1, color="tab:blue", alpha=0.2)
    sns.lineplot(
        x=x, y=proxies.local_models[1](x[:, None])[:, 0], ax=ax[2], errorbar=errorbars
    )
    ax[2].plot(x, lower2, color="tab:orange", alpha=0.1)
    ax[2].plot(x, upper2, color="tab:orange", alpha=0.1)
    ax[2].fill_between(x, lower2, upper2, color="tab:orange", alpha=0.2)
    ax[2].set_ylim(ax[0].get_ylim())
    ax[2].set_ylabel("")
    ax[0].set_ylabel("f(x)")
    ax[1].set_xlabel("x")
    # plt.savefig(MANUSCRIPT_DIR / "pyramid_example.pdf", dpi=600)


def paper_theme(
    width: float = 1.0,
    aspect: float = 1.0,
    cols: int = 1,
    rows: int = 1,
    page_width: float = 347.0,
    figsize: bool = False,  # return figsize instead of dict
) -> Union[Dict[str, float], Tuple[float, float]]:
    """Set theme and sizes for plots added to papers.

    Usage:
        sns.relplot(..., **paper_theme(...))
        plt.subplots(..., figsize=paper_theme(..., figsize=True))

    Args:
        width: Fraction of page width. Defaults to 1.0.
        aspect: Aspect ratio of plots. Defaults to 1.0.
        cols: Number of columns in plot. Defaults to 1.
        rows: Number of rows in plot. Defaults to 1.
        page_width: Page width in points. Defaults to 347.0.
        figsize: Return figsize instead of dict (for use with bare pyplot figures). Defaults to False.

    Returns:
        The size.
    """
    if width == 1.0:
        width = 0.99
    scale = page_width / 72.27  # from points to inches
    size = (width * scale, width / cols * rows / aspect * scale)
    sns.set_theme(
        context={k: v * 0.6 for k, v in sns.plotting_context("paper").items()},
        style=sns.axes_style("ticks"),
        palette="bright",
        # font="cmr10",
        rc={
            "figure.figsize": size,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 1e-4,
        },
    )
    if figsize:
        return size
    else:
        return dict(height=width * scale / cols * aspect, aspect=aspect)
