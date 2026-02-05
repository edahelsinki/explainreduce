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
)
from utils.hyperparameters import get_data, get_bb, get_params
from sklearn.model_selection import train_test_split
from functools import reduce, partial
import torch
import explainreduce.localmodels as lm
import explainreduce.proxies as px
import operator
from explainreduce import metrics

# import warnings
# warnings.simplefilter("error")

OUTPUT_DIR = RESULTS_DIR / "coverage_epsilon_sensitivity"


def preprocess_results(odf: pd.DataFrame) -> pd.DataFrame:
    df = odf.copy(deep=True)
    df.loc[:, "proxy_method_mod"] = "None"
    df = df.loc[~df["proxy_method"].isna()]
    df.loc[df["proxy_method"].str.contains("minimal_set_cov"), "proxy_method_mod"] = (
        "Min set"
    )
    df.loc[df["proxy_method"].str.contains("max_coverage"), "proxy_method_mod"] = (
        "Exact Max coverage"
    )
    df.loc[
        df["proxy_method"].str.contains("greedy_max_coverage"), "proxy_method_mod"
    ] = "Max coverage"
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_k_[0-9]+"),
        "proxy_method_mod",
    ] = "Min loss"
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_k_[0-9]+_min_cov"),
        "proxy_method_mod",
    ] = "Const Min loss"
    df.loc[
        df["proxy_method"].str.contains("balanced"),
        "proxy_method_mod",
    ] = "Balanced"
    df.loc[df["proxy_method"].str.contains("random"), "proxy_method_mod"] = "Random"
    df = df.loc[df["proxy_method_mod"] == "Balanced"]
    df = df.loc[~df["exp_method"].isin(["VanillaGrad", "GlobalLinear", "SLIPMAP"])]
    df = df.loc[df["init_p"] != 0.41]
    df = df.loc[df["k"] == 6]
    return df


def plot_results(df: pd.DataFrame):
    """Produce a small and large plot of the results and save to the repo."""
    df = preprocess_results(df)
    g = sns.relplot(
        df,
        x="init_p",
        y="proxy_fidelity",
        hue="exp_method",
        style="exp_method",
        col="data",
        col_wrap=4,
        kind="line",
        facet_kws={"sharey": False},
    )
    for i, ax in enumerate(g.axes):
        if i > 7:
            ax.set_xlabel("p")
        if i % 4 == 0:
            ax.set_ylabel("Fidelity")
        ax.set_title(df["data"].unique()[i])
    g.fig.savefig(MANUSCRIPT_DIR / "epsilon_sensitivity_new.pdf", dpi=300)


def prepare_data(dataset_fun):

    X, y, bb, cls = dataset_fun()
    n = min(1_000, X.shape[0] * 3 // 4)
    X, y = X[:n, :], y[:n]
    if y.ndim < 2:
        y = y[:, None]
    if cls:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42, stratify=y[:, 1]
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    X_test = torch.as_tensor(X_test, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.float32)
    out = {
        "original": (X_train, y_train),
        "test": (X_test, y_test),
        "black_box": bb,
        "classifier": cls,
    }
    return out


def get_explainer(dname, bb, cls):
    return {
        "SLISEMAP": partial(
            lm.SLISEMAPExplainer, classifier=cls, **get_params("SLISEMAP", dname, bb)
        ),
        "SLIPMAP": partial(
            lm.SLIPMAPExplainer, classifier=cls, **get_params("SLIPMAP", dname, bb)
        ),
        "VanillaGrad": partial(
            lm.VanillaGradExplainer,
            classifier=cls,
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


def get_optimisation_method(coverage, p):
    return {
        ("greedy_balanced_k_6", coverage, p): partial(
            px.find_proxies_loss_cov_linear, k=6, p=p
        ),
    }


def calculate_loss_distance(
    original_explainer, yhat_original, yhat_other, dataset_type
):
    loss_vector = original_explainer.loss_fn(yhat_original, yhat_other)
    if dataset_type == "offset":
        loss_vector = loss_vector[: (loss_vector.shape[0] // 2)]
    return loss_vector.mean().item()


def calculate_vector_distance(orig_vectors, other_vectors, dataset_type):
    D = torch.cdist(orig_vectors, other_vectors)
    if dataset_type == "offset":
        n = D.shape[0] // 2
        D = D[:n, :n]
    return D.mean().item()


def evaluate(job_id: int, ps: list) -> None:
    """Evaluate the full set of optimisation problems."""
    print(f"Begin job {job_id}.", flush=True)
    np.random.seed(1618 + job_id)
    torch.manual_seed(1618 + job_id)
    output_file = OUTPUT_DIR / f"sensitivity_{job_id:02d}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        results = read_parquet(output_file)
    else:
        results = pd.DataFrame()

    for dname, ds_fn in get_data().items():
        print(f"Preparing:   {dname}")
        ds = prepare_data(ds_fn)
        cls = ds["classifier"]
        bb_name = ds["black_box"]
        X_test, y_test = ds["test"]
        loss_fn = (
            torch.nn.MSELoss(reduction="none")
            if not cls
            else lm.logistic_regression_loss
        )
        X, y = ds["original"]
        m, pred_fn = get_bb(bb_name, cls, X, y, dname)
        yhat_train = torch.as_tensor(pred_fn(X), dtype=y.dtype)
        yhat_test = pred_fn(X_test)
        if yhat_train.ndim < y.ndim:
            yhat_train = yhat_train[:, None]
        loss_train = loss_fn(yhat_train, y)
        global_epsilon = torch.quantile(loss_train, 0.2).item()
        for expname, expfn in get_explainer(dname, bb_name, cls).items():
            D = torch.cdist(torch.as_tensor(X), torch.as_tensor(X))
            D += torch.eye(D.shape[0]) * torch.max(D)
            nn = torch.argsort(D, 1)[:, :5]
            explainer = expfn(X, yhat_train, black_box_model=m)
            explainer.fit()

            L = explainer.get_L()
            explainer_yhat_test = explainer.predict(X_test)
            full_fidelity = (
                loss_fn(torch.as_tensor(yhat_test), explainer_yhat_test).mean().item()
            )

            print("Loop over ps.")
            p_options = [get_optimisation_method(coverage=0.61, p=p) for p in ps]
            p_options = reduce(operator.ior, p_options, {})
            for (pname, coverage, p), proxy_method in p_options.items():
                if (
                    not results.empty
                    and (
                        (results["data"] == dname)
                        & (results["exp_method"] == expname)
                        & (results["proxy_method"] == pname)
                        & (results["init_coverage"] == coverage)
                        & (results["init_p"] == p)
                    ).any()
                ):
                    continue
                res = eval_proxy_method(
                    job_id,
                    dname,
                    expname,
                    explainer,
                    pname,
                    proxy_method,
                    coverage,
                    p,
                    X,
                    X_test,
                    y,
                    y_test,
                    yhat_test,
                    loss_fn,
                    L,
                    full_fidelity,
                    nn,
                )
                results = pd.concat((results, pd.DataFrame([res])), ignore_index=True)
                results.to_parquet(output_file)
    print("Done.", flush=True)


def eval_proxy_method(
    job_id,
    dname,
    expname,
    explainer,
    pname,
    proxy_method,
    coverage,
    p,
    X,
    X_test,
    y,
    y_test,
    yhat_test,
    loss_fn,
    L,
    full_fidelity,
    nn,
):
    print(
        f"Reducing: {dname} - {expname} - {pname} - c={coverage} - p={p}",
        flush=True,
    )
    try:
        proxies = proxy_method(explainer)
    except ValueError as ve:
        print(f"Reduction method resulted in error:\n", ve, flush=True)
        return
    print(
        f"Evaluating: {dname} - {expname} - {pname} - c={coverage} - p={p}",
        flush=True,
    )
    prx_yhat_train = proxies.predict(X)
    prx_yhat_test = proxies.predict(X_test)
    loss_train = loss_fn(torch.as_tensor(y), prx_yhat_train)
    loss_test = loss_fn(torch.as_tensor(y_test), prx_yhat_test)
    reduced_L = proxies.get_L()
    expanded_L = reduced_L[list(proxies.mapping.values()), :]
    proxy_fidelity = loss_fn(torch.as_tensor(yhat_test), prx_yhat_test).mean().item()
    epsilon = torch.quantile(loss_train, q=p).item()

    res = dict(
        job=job_id,
        data=dname,
        exp_method=expname,
        proxy_method=pname,
        k=len(proxies.local_models),
        init_coverage=coverage,
        init_p=p,
        epsilon=epsilon,
        full_fidelity=full_fidelity,
        full_stability=L[torch.arange(L.shape[0])[:, None], nn].mean().item(),
        full_coverage=metrics.calculate_coverage(L < epsilon),
        loss_train=loss_train.mean().item(),
        loss_test=loss_test.mean().item(),
        proxy_fidelity=proxy_fidelity,
        proxy_coverage=metrics.calculate_coverage(reduced_L < epsilon),
        proxy_stability=expanded_L[torch.arange(expanded_L.shape[0])[:, None], nn]
        .mean()
        .item(),
    )
    return res


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        plot_results(df)
    else:
        ps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        time = timer()
        evaluate(int(sys.argv[1]), ps=ps)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
