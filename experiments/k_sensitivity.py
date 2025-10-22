###############################################################################
#
# This experiment evaluate the fidelity as a function of the proxy number k,
# with certain combinations of datasets and explanation methods.
#
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments,
# where job_id is an integer indicating the id of the parallel task.
# For example (the number of proxies is defined within the main function):
#   `python experiment/k_sensitivity.py $job_id`
#
# Run this script again without additional arguments to produce plots of fidelity from the results:
#   `python experiment/k_sensitivity.py`
#
###############################################################################

from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
from glob import glob
import sys
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import (
    rename_columns,
    paper_theme,
    prepare_data,
    read_parquet,
    get_explainer,
    reduce,
    get_explanation_radius,
)
import operator
from utils.hyperparameters import get_data, get_bb
import torch
import explainreduce.localmodels as lm
import explainreduce.proxies as px
from functools import partial
import explainreduce.metrics as metrics

OUTPUT_DIR = RESULTS_DIR / "k_sensitivity"


def plot_results(df: pd.DataFrame):
    """Produce the small and large (appendix) plots for k-sensitivity."""
    for f in OUTPUT_DIR.glob("**/*.parquet"):
        try:
            new_df = pq.ParquetFile(f).read().to_pandas()
            if "0" in new_df.columns:
                new_df = new_df.drop(["0"], axis=1)
            df = pd.concat((new_df, df), ignore_index=True)
        except Exception as e:
            print(f"Failed to read {f}")
            print(e)
    df = rename_columns(df)
    exp_methods = ["LIME", "SHAP", "SLISEMAP"]
    datasets = ["Gas Turbine", "Jets", "QM9"]
    p_df = df.loc[df["exp_method"].isin(exp_methods)]
    p_df = p_df.loc[p_df["data"].isin(datasets)]
    p_df = p_df.sort_values(["data", "exp_method"])
    p_df.loc[:, "proxy fidelity/stability"] = (
        p_df["proxy_fidelity"] / p_df["proxy_stability"]
    )
    p_df = p_df.rename(
        columns={"proxy_method_mod": "Reduction method", "proxy_fidelity": "Fidelity"}
    )
    mean_df = (
        p_df.groupby(["data", "exp_method"])
        .mean(numeric_only=True)[
            ["full_fidelity", "full_stability", "full_coverage", "bb_loss_test"]
        ]
        .reset_index()
    )
    g = sns.relplot(
        p_df,
        x="k",
        y="Fidelity",
        col="exp_method",
        row="data",
        hue="Reduction method",
        style="Reduction method",
        kind="line",
        facet_kws={"sharey": False},
        **paper_theme(cols=len(exp_methods), rows=len(datasets)),
    )
    i_f = 0
    for i, dname in enumerate(p_df["data"].unique()):
        # set ylims based on dataset
        data_y_min, data_y_max = np.inf, -np.inf
        for j, exp_name in enumerate(p_df["exp_method"].unique()):
            y_min, y_max = g.axes[i, j].get_ylim()
            if y_min < data_y_min:
                data_y_min = y_min
            if y_max > data_y_max:
                data_y_max = y_max
        for j, exp_name in enumerate(p_df["exp_method"].unique()):
            g.axes[i, j].axhline(
                mean_df.loc[
                    (mean_df["data"] == dname) & (mean_df["exp_method"] == exp_name)
                ]["full_fidelity"].values[0],
                color="gray",
                linestyle="--",
            )
            # g.axes[i, j].set_ylim((data_y_min, data_y_max))
            g.axes[i, j].set_title(f"Dataset: {dname}, XAI: {exp_name}")
            i_f += 1
    plt.savefig(MANUSCRIPT_DIR / "fidelity_k.pdf", dpi=600)


def plot_results_full(df: pd.DataFrame):
    """Produce the small and large (appendix) plots for k-sensitivity."""
    for f in OUTPUT_DIR.glob("**/*.parquet"):
        try:
            new_df = pq.ParquetFile(f).read().to_pandas()
            if "0" in new_df.columns:
                new_df = new_df.drop(["0"], axis=1)
            df = pd.concat((new_df, df), ignore_index=True)
        except Exception as e:
            print(f"Failed to read {f}")
            print(e)
    df = rename_columns(df)
    exp_methods = ["LIME", "SHAP", "SLISEMAP", "SmoothGrad"]
    datasets = df["data"].unique()
    p_df = df.loc[df["exp_method"].isin(exp_methods)]
    p_df = p_df.loc[p_df["data"].isin(datasets)]
    p_df = p_df.sort_values(["data", "exp_method"])
    p_df.loc[:, "proxy fidelity/stability"] = (
        p_df["proxy_fidelity"] / p_df["proxy_stability"]
    )
    p_df = p_df.rename(
        columns={"proxy_method_mod": "Reduction method", "proxy_fidelity": "Fidelity"}
    )
    mean_df = (
        p_df.groupby(["data", "exp_method"])
        .mean(numeric_only=True)[
            ["full_fidelity", "full_stability", "full_coverage", "bb_loss_test"]
        ]
        .reset_index()
    )
    g = sns.relplot(
        p_df,
        x="k",
        y="Fidelity",
        col="exp_method",
        row="data",
        hue="Reduction method",
        style="Reduction method",
        kind="line",
        facet_kws={"sharey": False},
    )
    i_f = 0
    for i, dname in enumerate(p_df["data"].unique()):
        # set ylims based on dataset
        data_y_min, data_y_max = np.inf, -np.inf
        for j, exp_name in enumerate(p_df["exp_method"].unique()):
            y_min, y_max = g.axes[i, j].get_ylim()
            if y_min < data_y_min:
                data_y_min = y_min
            if y_max > data_y_max:
                data_y_max = y_max
        for j, exp_name in enumerate(p_df["exp_method"].unique()):
            g.axes[i, j].axhline(
                mean_df.loc[
                    (mean_df["data"] == dname) & (mean_df["exp_method"] == exp_name)
                ]["full_fidelity"].values[0],
                color="gray",
                linestyle="--",
            )
            # g.axes[i, j].set_ylim((data_y_min, data_y_max))
            g.axes[i, j].set_title(f"Dataset: {dname}, XAI: {exp_name}")
            i_f += 1
    # Reduce the size to fit into the paper
    fig = g.figure
    original_size = fig.get_size_inches()
    fig.set_size_inches(original_size[0], original_size[1] * 5 / 8)
    plt.savefig(MANUSCRIPT_DIR / "fidelity_k_full.pdf", dpi=600)


def get_optimisation_method(k: int, explainer: lm.Explainer):
    return {
        (f"greedy_max_coverage_k_{k}", k): partial(px.find_proxies_greedy, k=k),
        (f"max_coverage_k_{k}", k): partial(
            px.find_proxies_coverage, k=k, time_limit=1000, pulp_msg=False
        ),
        (f"random_k_{k}", k): partial(px.find_proxies_random, k=k),
        (f"greedy_min_loss_{k}", k): partial(px.find_proxies_greedy_k_min_loss, k=k),
        (f"greedy_descent_min_loss_{k}", k): partial(
            px.find_proxies_greedy_k_min_loss_descent, k=k
        ),
        (f"greedy_min_loss_{k}_min_cov", k): partial(
            px.find_proxies_greedy_min_loss_k_min_cov,
            k=k,
            min_coverage=0.8,
            raise_on_infeasible=False,
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


def eval_proxy_method_k_sensitivity(
    job_id,
    dname,
    expname,
    explainer,
    pname,
    proxy_method,
    k,
    global_epsilon,
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
        f"Reducing: {dname} - {expname} - {pname} - k={k}",
        flush=True,
    )
    try:
        proxies = proxy_method(explainer, epsilon=global_epsilon)
    except ValueError as ve:
        print(f"Reduction method resulted in error:\n", ve, flush=True)
        return
    print(
        f"Evaluating: {dname} - {expname} - {pname} - k={k}",
        flush=True,
    )

    # Calculate the loss of the proxy model on the training set
    prx_yhat_train = proxies.predict(X)
    loss_train = loss_fn(torch.as_tensor(y), prx_yhat_train)

    # Calculate the loss of the proxy model on the testing set, and the test fidelity
    prx_yhat_test = proxies.predict(X_test)
    loss_test = loss_fn(torch.as_tensor(y_test), prx_yhat_test)
    proxy_fidelity = loss_fn(torch.as_tensor(yhat_test), prx_yhat_test).mean().item()

    # Get the loss matrix of the proxy model, and the expanded loss matrix
    reduced_L = proxies.get_L()
    expanded_L = reduced_L[list(proxies.mapping.values()), :]

    # Get the results
    res = dict(
        job=job_id,
        data=dname,
        exp_method=expname,
        proxy_method=pname,
        k=len(proxies.local_models),
        epsilon=global_epsilon,
        full_fidelity=full_fidelity,
        full_stability=L[torch.arange(L.shape[0])[:, None], nn].mean().item(),
        full_coverage=metrics.calculate_coverage(L < global_epsilon),
        full_loss_test=full_loss,
        bb_loss_test=bb_loss,
        loss_train=loss_train.mean().item(),
        loss_test=loss_test.mean().item(),
        proxy_fidelity=proxy_fidelity,
        proxy_coverage=metrics.calculate_coverage(reduced_L < global_epsilon),
        proxy_stability=expanded_L[torch.arange(expanded_L.shape[0])[:, None], nn]
        .mean()
        .item(),
    )

    return res


def evaluate_k_sensitivity(job_id: int, ks: list[int]) -> None:
    """Evaluate the full set of optimisation problems."""
    print(f"Begin job {job_id}.", flush=True)

    # Set random seeds for reproducibility
    np.random.seed(1618 + job_id)
    torch.manual_seed(1618 + job_id)

    # Determine output file path and ensure its directory exists
    output_file = OUTPUT_DIR / f"k_sensitivity_{job_id:02d}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load the results if they exist, otherwise create an empty DataFrame
    if output_file.exists():
        results = read_parquet(output_file)
    else:
        results = pd.DataFrame()

    # Iterate through the datasets and their respective preparation functions
    for dname, ds_fn in get_data().items():
        print(f"Preparing:   {dname}")

        # Prepare the dataset and extract its components
        ds = prepare_data(ds_fn, seed=1618 + job_id)
        cls = ds["classifier"]
        bb_name = ds["black_box"]
        X_test, y_test = ds["test"]

        # Define the loss function based on whether it is a classification task
        loss_fn = (
            torch.nn.MSELoss(reduction="none")
            if not cls
            else lm.logistic_regression_loss
        )

        # Retrieve the training data, the black-box model, and its prediction function
        X_bb, y_bb = ds["bb_train"]
        m, pred_fn = get_bb(bb_name, cls, X_bb, y_bb, dname)

        # Make predictions on the test set and training set
        yhat_test = pred_fn(X_test)
        yhat_test = torch.as_tensor(yhat_test, dtype=y_bb.dtype)
        if yhat_test.ndim < y_bb.ndim:
            yhat_test = yhat_test[:, None]
        yhat_train = torch.as_tensor(pred_fn(X_bb), dtype=y_bb.dtype)
        if yhat_train.ndim < y_bb.ndim:
            yhat_train = yhat_train[:, None]

        # Compute the loss on the training data and determine the global epsilon
        loss_train = loss_fn(yhat_train, y_bb)
        global_epsilon = torch.quantile(loss_train, 0.2).item()

        # Iterate through different explainer methods
        for expname, expfn in get_explainer(dname, bb_name, cls).items():
            X, y = ds["exp_train"]

            # Compute pairwise distances and nearest neighbors
            D = torch.cdist(torch.as_tensor(X), torch.as_tensor(X))
            D += torch.eye(D.shape[0]) * torch.max(D)
            nn = torch.argsort(D, 1)[:, :5]

            # Fit the explainer to the training data
            yhat = pred_fn(X)
            explainer = expfn(X, yhat, black_box_model=m)
            explainer.fit()

            # Get the explanation matrix and make predictions on the test set
            L = explainer.get_L()
            explainer_yhat_test = explainer.predict(X_test)

            # Compute fidelity and loss metrics
            full_fidelity = (
                loss_fn(
                    torch.as_tensor(yhat_test),
                    explainer_yhat_test,
                )
                .mean()
                .item()
            )
            full_loss = (
                loss_fn(
                    torch.as_tensor(y_test),
                    explainer_yhat_test,
                )
                .mean()
                .item()
            )
            bb_loss = (
                loss_fn(
                    torch.as_tensor(y_test),
                    yhat_test,
                )
                .mean()
                .item()
            )

            # Loop over k values and their corresponding optimization methods
            print("Loop over k values.")
            k_options = [get_optimisation_method(k=k, explainer=explainer) for k in ks]
            k_options = reduce(operator.ior, k_options, {})
            for (pname, k), proxy_method in k_options.items():
                # Skip already computed results
                if (
                    not results.empty
                    and (
                        (results["data"] == dname)
                        & (results["exp_method"] == expname)
                        & (results["proxy_method"] == pname)
                        & (results["k"] == k)
                    ).any()
                ):
                    continue
                if expname == "GlobalLinear" and k > 1:
                    continue
                if "max_ball_coverage" in pname and not explainer.classifier:
                    continue
                if "glocalx" in pname and not ("LORERule" in str(explainer.__class__)):
                    continue
                if "submodular_pick" in pname and not (
                    "LIME" in str(explainer.__class__)
                ):
                    continue
                res = eval_proxy_method_k_sensitivity(
                    job_id,
                    dname,
                    expname,
                    explainer,
                    pname,
                    proxy_method,
                    k,
                    global_epsilon,
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
                )
                if res is not None:
                    results = pd.concat(
                        (results, pd.DataFrame([res])), ignore_index=True
                    )
                    results.to_parquet(output_file)
    print("Done.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        plot_results(df)
        plot_results_full(df)
    else:
        ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        time = timer()
        evaluate_k_sensitivity(int(sys.argv[1]), ks=ks)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
