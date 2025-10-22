import torch

from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
from glob import glob
import sys
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial, reduce
import operator
from timeit import default_timer as timer
import explainreduce.localmodels as lm
import explainreduce.proxies as px
from utils.utils import read_parquet, paper_theme
from explainreduce import metrics
from utils.hyperparameters import get_params, get_bb, get_data
import matplotlib.pyplot as plt
import seaborn as sns

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
        "Max coverage"
    )
    df.loc[
        df["proxy_method"].str.contains("greedy_max_coverage"), "proxy_method_mod"
    ] = "Greedy Max coverage"
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_k_[0-9]+"),
        "proxy_method_mod",
    ] = "Greedy Min loss (fixed k)"
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_k_[0-9]+_min_cov"),
        "proxy_method_mod",
    ] = "Greedy Min loss (minimum coverage)"
    df.loc[df["proxy_method"].str.contains("random"), "proxy_method_mod"] = "Random"
    return df


def plot_result_small(df: pd.DataFrame):
    plt.rcParams.update({"font.size": 16})
    exp_methods = ["LIME", "SHAP", "SLISEMAP"]
    datasets = ["Gas Turbine", "Jets"]
    p_df = df.loc[df["exp_method"].isin(exp_methods)]
    p_df = p_df.loc[p_df["data"].isin(datasets)]
    fig, ax = plt.subplots(ncols=4, figsize=(25, 5.5))
    r_df = p_df.rename(
        columns={
            "proxy_method_mod": "Reduction method",
            "exp_method": "XAI method",
        }
    )
    dot_size = 150
    gs = r_df.loc[r_df["data"] == "Gas Turbine"]
    j = r_df.loc[r_df["data"] == "Jets"]
    # gas turbine - epsilon
    sns.scatterplot(
        gs.loc[gs["init_p"] != 0.41],
        x="init_p",
        y="proxy_fidelity",
        hue="Reduction method",
        style="XAI method",
        ax=ax[0],
        s=dot_size,
    )
    ax[0].set_title("Gas Turbine")
    ax[0].set_ylim([0, 1.0])
    ax[0].get_legend().remove()
    ax[0].set_ylabel("Fidelity")
    ax[0].set_xlabel("Error tolerance")
    # gas turbine - c
    sns.scatterplot(
        gs.loc[
            (gs["init_coverage"] != 0.61)
            & (gs["Reduction method"] == "Greedy Min loss (minimum coverage)")
        ],
        x="init_coverage",
        y="proxy_fidelity",
        color="tab:orange",
        style="XAI method",
        ax=ax[1],
        s=dot_size,
    )
    ax[1].set_title("Gas Turbine")
    ax[1].set_ylim([0, 1.0])
    ax[1].get_legend().remove()
    ax[1].set_ylabel(None)
    ax[1].set_yticklabels([])
    ax[1].set_xlabel("Minimum coverage")
    # jets - epsilon
    sns.scatterplot(
        j.loc[j["init_coverage"] != 0.61],
        x="init_coverage",
        y="proxy_fidelity",
        hue="Reduction method",
        style="XAI method",
        ax=ax[2],
        s=dot_size,
    )
    ax[2].set_title("Jets")
    ax[2].set_ylim([0, 1.0])
    ax[2].legend(bbox_to_anchor=(2.75, -0.15), ncol=9)
    ax[2].set_ylabel(None)
    ax[2].set_yticklabels([])
    ax[2].set_xlabel("Error tolerance")
    # jets - c
    sns.scatterplot(
        j.loc[
            (j["init_p"] != 0.41)
            & (j["Reduction method"] == "Greedy Min loss (minimum coverage)")
        ],
        x="init_p",
        y="proxy_fidelity",
        color="tab:orange",
        style="XAI method",
        ax=ax[3],
        s=dot_size,
    )
    ax[3].set_title("Jets")
    ax[3].set_ylim([0, 1.0])
    ax[3].get_legend().remove()
    ax[3].set_ylabel(None)
    ax[3].set_yticklabels([])
    ax[3].set_xlabel("Minimum coverage")
    # fig.tight_layout()
    plt.savefig(
        MANUSCRIPT_DIR / "coverage_p_sensitivity.pdf", bbox_inches="tight", dpi=300
    )


def plot_result_full(df: pd.DataFrame):
    plt.rcParams.update({"font.size": 20})
    num_datasets = len(pd.unique(df["data"]))
    r_df = df.rename(
        columns={
            "proxy_method_mod": "Reduction method",
            "exp_method": "XAI method",
        }
    )

    fig, ax = plt.subplots(
        nrows=num_datasets // 2 + (num_datasets % 2), ncols=4, figsize=(30, 30)
    )
    row_counter = 0
    col_counter = 0
    for dname, gdf in r_df.groupby("data"):
        curr_ax = ax[row_counter, col_counter]
        # error tolerance
        sns.lineplot(
            gdf.loc[gdf["init_p"] != 0.41],
            x="init_p",
            y="proxy_fidelity",
            hue="Reduction method",
            style="XAI method",
            ax=curr_ax,
        )
        curr_ax.set_title(dname)
        curr_ax.set_ylim(bottom=0)
        curr_ax.set_ylabel("Fidelity")
        curr_ax.set_xlabel("Error tolerance")
        if row_counter == 0 and col_counter == 0:
            curr_ax.legend(bbox_to_anchor=(4.5, -3.75), ncol=6)
        else:
            curr_ax.get_legend().remove()
        col_counter += 1
        # coverage
        curr_ax = ax[row_counter, col_counter]
        sns.lineplot(
            gdf.loc[
                (gdf["init_coverage"] != 0.61)
                & (gdf["Reduction method"] == "Greedy Min loss (minimum coverage)")
            ],
            x="init_coverage",
            y="proxy_fidelity",
            color="tab:red",
            style="XAI method",
            ax=curr_ax,
        )
        curr_ax.set_title(dname)
        curr_ax.set_ylim(bottom=0)
        curr_ax.get_legend().remove()
        curr_ax.set_ylabel(None)
        curr_ax.set_yticklabels([])
        curr_ax.set_xlabel("Minimum coverage")
        col_counter += 1
        if col_counter > 3:
            row_counter += 1
            col_counter = 0
    fig.tight_layout()
    # fig.subplots_adjust(hspace=0.4, left=-0.1)
    plt.savefig(
        MANUSCRIPT_DIR / "coverage_p_sensitivity_full.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_results(df: pd.DataFrame):
    """Produce a small and large plot of the results and save to the repo."""
    df = preprocess_results(df)
    plot_result_small(df)
    plot_result_full(df)


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
        # ("minimal_set_cov", coverage, p): partial(
        # px.find_proxies_minimal_set, min_coverage=coverage, p=p, time_limit=30
        # ),
        ("greedy_max_coverage_k_5", coverage, p): partial(
            px.find_proxies_greedy, k=5, p=p
        ),
        ("greedy_min_loss_k_5_min_cov", coverage, p): partial(
            px.find_proxies_greedy_min_loss_k_min_cov, k=5, p=p, min_coverage=coverage
        ),
        #         ("max_coverage_k_5", coverage, p): partial(
        #             px.find_proxies_coverage, k=5, p=p, time_limit=30
        #         ),
        ("random_k_10", coverage, p): partial(px.find_proxies_random, k=10),
        # ("greedy_min_loss", coverage, p): partial(
        # px.find_proxies_greedy_min_loss, min_coverage=coverage, p=p
        # ),
        # (f"min_loss_k_5", coverage, p): partial(
        # px.find_proxies_loss, k=5, time_limit=300
        # ),
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


def evaluate(job_id: int, coverages: list, ps: list) -> None:
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

            print("Loop over coverages.")
            cov_options = [
                get_optimisation_method(coverage=c, p=0.41) for c in coverages
            ]
            cov_options = reduce(operator.ior, cov_options, {})
            for (pname, coverage, p), proxy_method in cov_options.items():
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
        coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        time = timer()
        evaluate(int(sys.argv[1]), coverages=coverages, ps=ps)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
