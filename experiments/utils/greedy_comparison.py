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
from project_paths import MANUSCRIPT_DIR, RESULTS_DIR
import pandas as pd
import operator
from functools import reduce
from hyperparameters import get_bb, get_data, get_params
from typing import List
from timeit import default_timer as timer

OUTPUT_DIR = RESULTS_DIR / "greedy_comparison"


def preprocess_results(df: pd.DataFrame):
    df.loc[:, "proxy_method_mod"] = "None"
    df = df.loc[~df["proxy_method"].isna()]
    df.loc[df["proxy_method"].str.contains("minimal_set_cov"), "proxy_method_mod"] = (
        "Min set"
    )
    df.loc[df["proxy_method"].str.contains("max_coverage"), "proxy_method_mod"] = (
        "Optimal Max Coverage"
    )
    df.loc[
        df["proxy_method"].str.contains("greedy_max_coverage"), "proxy_method_mod"
    ] = "Greedy Max Coverage"
    df.loc[df["proxy_method"].str.contains("random"), "proxy_method_mod"] = "Random"
    df.loc[df["proxy_method"].str.contains("min_loss"), "proxy_method_mod"] = (
        "Optimal Min Loss"
    )
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_[0-9]+"), "proxy_method_mod"
    ] = "Greedy Min Loss (fixed k)"
    df.loc[
        df["proxy_method"].str.contains("greedy_min_loss_[0-9]+_min_cov"),
        "proxy_method_mod",
    ] = "Greedy Min Loss (minimum coverage)"
    df.loc[df["proxy_method"].str.contains("B"), "proxy_method_mod"] = "B K-means"
    df.loc[df["proxy_method"].str.contains("L"), "proxy_method_mod"] = "L K-means"
    df.loc[df["proxy_method"].str.contains("X"), "proxy_method_mod"] = "X K-means"
    min_loss = df.loc[df["proxy_method_mod"] == "Optimal Min Loss"]
    min_loss = min_loss.rename(
        columns={
            "proxy_fidelity_train": "proxy_fidelity_train_opt",
            "proxy_fidelity": "proxy_fidelity_opt",
        }
    )
    merge_cols = ["job", "data", "exp_method", "k"]
    df = df.merge(
        min_loss[merge_cols + ["proxy_fidelity_train_opt", "proxy_fidelity_opt"]],
        on=merge_cols,
    )
    df.loc[:, "fidelity_approximation_ratio_train"] = (
        df["proxy_fidelity_train"] / df["proxy_fidelity_train_opt"]
    )
    df.loc[:, "fidelity_approximation_ratio_test"] = (
        df["proxy_fidelity"] / df["proxy_fidelity_opt"]
    )
    max_cov = df.loc[df["proxy_method_mod"] == "Optimal Max Coverage"]
    max_cov = max_cov.rename(columns={"proxy_coverage": "proxy_coverage_opt"})
    df = df.merge(max_cov[merge_cols + ["proxy_coverage_opt"]], on=merge_cols)
    df.loc[:, "coverage_approximation_ratio"] = (
        df["proxy_coverage"] / df["proxy_coverage_opt"]
    )
    return df


def concanate_results(
    df: pd.DataFrame,
    exp_methods=["LIME"],
    datasets=["Air Quality", "QM9", "Jets", "Gas Turbine"],
):
    min_methods = [
        "Optimal Min Loss",
        "Greedy Min Loss (fixed k)",
        "Greedy Min Loss (minimum coverage)",
    ]
    cov_methods = [
        "Optimal Max Coverage",
        "Greedy Max Coverage",
        "Greedy Min Loss (minimum coverage)",
    ]
    metrics = ["proxy_coverage", "proxy_fidelity_train", "proxy_fidelity"]
    mean_dfs = []
    std_dfs = []
    for metric in metrics:
        if metric == "proxy_coverage":
            filter_list = cov_methods
            approx_ratio = "coverage_approximation_ratio"
        elif metric == "proxy_fidelity_train":
            filter_list = min_methods
            approx_ratio = "fidelity_approximation_ratio_train"
        else:
            filter_list = min_methods
            approx_ratio = "fidelity_approximation_ratio_test"
        p_df = df.loc[df["exp_method"].isin(exp_methods)]
        p_df = p_df.loc[p_df["data"].isin(datasets)]
        p_df = p_df.loc[p_df["proxy_method_mod"].isin(filter_list)]
        p_df = p_df.rename(columns={"data": "Dataset"})
        p_m = pd.pivot_table(
            p_df,
            index="Dataset",
            columns="proxy_method_mod",
            values=metric,
            aggfunc="mean",
        )
        p_a_m = pd.pivot_table(
            p_df,
            index="Dataset",
            columns="proxy_method_mod",
            values=approx_ratio,
            aggfunc="mean",
        )
        p_a_m.columns = [c + "_approx" for c in p_m.columns]
        p_m = p_m.merge(p_a_m, on=["Dataset"])
        mean_dfs.append(p_m.reset_index())
        p_s = pd.pivot_table(
            p_df,
            index="Dataset",
            columns="proxy_method_mod",
            values=metric,
            aggfunc="std",
        )
        p_a_s = pd.pivot_table(
            p_df,
            index="Dataset",
            columns="proxy_method_mod",
            values=approx_ratio,
            aggfunc="std",
        )
        p_a_s.columns = [c + "_approx" for c in p_s.columns]
        p_s = p_s.merge(p_a_s, on=["Dataset"])
        std_dfs.append(p_s.reset_index())
    return mean_dfs, std_dfs


def produce_latex(mean_dfs: List[pd.DataFrame], std_dfs: List[pd.DataFrame]):
    metrics = ["proxy_coverage", "proxy_fidelity_train", "proxy_fidelity"]
    round_col = lambda x: np.round(x, 2)
    latexify = lambda x: f"${x}$"
    l_dfs = []
    for i in range(len(mean_dfs)):
        l_df = mean_dfs[i].copy(deep=True)
        m_df = mean_dfs[i]
        s_df = std_dfs[i]
        numeric_cols = m_df.select_dtypes(include=np.number).columns.tolist()
        for column in numeric_cols:
            l_df.loc[:, column] = l_df[column].apply(round_col)
            l_df[column] = l_df[column].astype(str)
            l_df.loc[:, column] = (
                l_df[column] + " \pm " + s_df[column].apply(round_col).astype(str)
            )
            l_df.loc[:, column] = l_df[column].apply(latexify)
        l_dfs.append(l_df)

    out = ""
    for exp_name, l_df in zip(metrics, l_dfs):
        if exp_name == "proxy_coverage":
            l_df = l_df[
                [
                    "Dataset",
                    "Greedy Max Coverage",
                    "Greedy Max Coverage_approx",
                    "Greedy Min Loss (minimum coverage)",
                    "Greedy Min Loss (minimum coverage)_approx",
                    "Optimal Max Coverage",
                ]
            ]
        else:
            l_df = l_df[
                [
                    "Dataset",
                    "Greedy Min Loss (fixed k)",
                    "Greedy Min Loss (fixed k)_approx",
                    "Greedy Min Loss (minimum coverage)",
                    "Greedy Min Loss (minimum coverage)_approx",
                    "Optimal Min Loss",
                ]
            ]
        l_df = l_df.rename(
            columns={
                "Greedy Min Loss (minimum coverage)": "G. Min Loss (min $c$)",
                "Greedy Min Loss (fixed k)": "G. Min Loss",
                "Greedy Max Coverage": "G. Max $c$",
                "Optimal Max Coverage": "Max $c$",
                "Optimal Min Loss": "Min Loss",
                "Greedy Max Coverage_approx": "A. ratio",
                "Greedy Min Loss (minimum coverage)_approx": "A. ratio",
                "Greedy Min Loss (fixed k)_approx": "A. ratio",
            }
        )
        l_df.columns = [f"\\bfseries {x}" for x in l_df.columns]
        ltx = l_df.style.hide(axis=0).to_latex(
            column_format="l@{\\hspace{3mm}} " + "r@{\\hspace{3mm}}" * 6 + "r",
            hrules=True,
            convert_css=False,
        )
        if out == "":
            out += ltx.split("\n")[0] + "\n"
        metric_name = None
        if exp_name == "proxy_fidelity_train":
            metric_name = "Train fidelity $\\downarrow$"
        elif exp_name == "proxy_fidelity":
            metric_name = "Test fidelity $\\downarrow$"
        else:
            metric_name = "Coverage $\\uparrow$"
        out += (
            "\\hline \\\\\n"
            + f"\\multicolumn{{6}}{{|c|}}{{{metric_name}}}\\\\\n"
            + "\\hline\\\\\n"
        )
        for line in ltx.split("\n"):
            if "tabular" in line or line in ["", "\\bottomrule", "\\toprule"]:
                continue
            else:
                out += line + "\n"

    out += "\\bottomrule\n" + "\\end{tabular}"
    print(out)
    with open(MANUSCRIPT_DIR / "greedy_comparison.tex", "w") as f:
        f.write(out)


def prepare_data(dataset_fun, seed):

    X, y, bb, cls = dataset_fun()
    n = min(10_000, X.shape[0])
    X, y = X[:n, :], y[:n]
    if y.ndim < 2:
        y = y[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed
    )
    X_exp_train, _, y_exp_train, _ = train_test_split(
        X_train, y_train, train_size=100, random_state=seed
    )
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    X_exp_train = torch.as_tensor(X_exp_train, dtype=torch.float32)
    X_test = torch.as_tensor(X_test, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    y_exp_train = torch.as_tensor(y_exp_train, dtype=torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.float32)
    out = {
        "bb_train": (X_train, y_train),
        "exp_train": (X_exp_train, y_exp_train),
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
        "LIME": partial(
            lm.LIMEExplainer, classifier=cls, **get_params("LIME", dname, bb)
        ),
        "SHAP": partial(
            lm.KernelSHAPExplainerLegacy,
            classifier=cls,
        ),
    }


def get_optimisation_method(k: int):
    return {
        (f"greedy_max_coverage_k_{k}", k): partial(px.find_proxies_greedy, k=k, p=0.33),
        (f"max_coverage_k_{k}", k): partial(
            px.find_proxies_coverage, k=k, p=0.33, time_limit=500, pulp_msg=False
        ),
        (f"random_k_{k}", k): partial(px.find_proxies_random, k=k),
        # (f"min_loss_{k}", k): partial(px.find_proxies_loss, k=k, time_limit=300),
        (f"greedy_min_loss_{k}", k): partial(px.find_proxies_greedy_k_min_loss, k=k),
        (f"greedy_min_loss_{k}_min_cov", k): partial(
            px.find_proxies_greedy_min_loss_k_min_cov, k=k, min_coverage=0.8, p=0.33
        ),
        (f"min_loss_{k}", k): partial(px.find_proxies_loss_recursive, k=k),
    }


def evaluate(job_id: int, ks: list[int]) -> None:
    """Evaluate the full set of optimisation problems."""
    print(f"Begin job {job_id}.", flush=True)
    np.random.seed(1618 + job_id)
    torch.manual_seed(1618 + job_id)
    output_file = OUTPUT_DIR / f"greedy_comp_{job_id:02d}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        results = read_parquet(output_file)
    else:
        results = pd.DataFrame()

    for dname, ds_fn in get_data().items():
        print(f"Preparing:   {dname}")
        ds = prepare_data(ds_fn, seed=1618 + job_id)
        cls = ds["classifier"]
        bb_name = ds["black_box"]
        X_test, y_test = ds["test"]
        loss_fn = (
            torch.nn.MSELoss(reduction="none")
            if not cls
            else lm.logistic_regression_loss
        )
        X_bb, y_bb = ds["bb_train"]
        global_epsilon = None
        m, pred_fn = get_bb(bb_name, cls, X_bb, y_bb, dname)
        yhat_test = pred_fn(X_test)
        yhat_test = torch.as_tensor(yhat_test, dtype=y_bb.dtype)
        if yhat_test.ndim < y_bb.ndim:
            yhat_test = yhat_test[:, None]
        for expname, expfn in get_explainer(dname, bb_name, cls).items():
            X, y = ds["exp_train"]
            D = torch.cdist(torch.as_tensor(X), torch.as_tensor(X))
            D += torch.eye(D.shape[0]) * torch.max(D)
            nn = torch.argsort(D, 1)[:, :5]
            yhat = pred_fn(X)
            explainer = expfn(X, yhat, black_box_model=m)
            explainer.fit()

            if global_epsilon is None:
                global_epsilon = metrics.calculate_global_epsilon(explainer)
            L = explainer.get_L()
            explainer_yhat_test = explainer.predict(X_test)
            explainer_yhat_train = explainer.predict(X)
            full_fidelity = (
                loss_fn(
                    torch.as_tensor(yhat_test),
                    explainer_yhat_test,
                )
                .mean()
                .item()
            )
            full_fidelity_train = (
                loss_fn(
                    torch.as_tensor(yhat),
                    explainer_yhat_train,
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

            print("Loop over k values.")
            k_options = [get_optimisation_method(k=k) for k in ks]
            k_options = reduce(operator.ior, k_options, {})
            for (pname, k), proxy_method in k_options.items():
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
                res = eval_proxy_method(
                    job_id,
                    dname,
                    expname,
                    explainer,
                    pname,
                    proxy_method,
                    k,
                    X,
                    X_test,
                    y,
                    y_test,
                    yhat,
                    yhat_test,
                    loss_fn,
                    L,
                    full_fidelity,
                    full_fidelity_train,
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


def eval_proxy_method(
    job_id,
    dname,
    expname,
    explainer,
    pname,
    proxy_method,
    k,
    X,
    X_test,
    y,
    y_test,
    yhat,
    yhat_test,
    loss_fn,
    L,
    full_fidelity,
    full_fidelity_train,
    full_loss,
    bb_loss,
    nn,
):
    print(
        f"Reducing: {dname} - {expname} - {pname} - k={k}",
        flush=True,
    )
    try:
        time = timer()
        proxies = proxy_method(explainer)
        print(f"Reduction took {timer() - time:.2f} s.", flush=True)
    except ValueError as ve:
        print(f"Reduction method resulted in error:\n", ve, flush=True)
        return
    print(
        f"Evaluating: {dname} - {expname} - {pname} - k={k}",
        flush=True,
    )
    prx_yhat_train = proxies.predict(X)
    prx_yhat_test = proxies.predict(X_test)
    loss_train = loss_fn(torch.as_tensor(y), prx_yhat_train)
    loss_test = loss_fn(torch.as_tensor(y_test), prx_yhat_test)
    reduced_L = proxies.get_L()
    expanded_L = reduced_L[list(proxies.mapping.values()), :]
    proxy_fidelity = loss_fn(torch.as_tensor(yhat_test), prx_yhat_test).mean().item()
    proxy_fidelity_train = loss_fn(torch.as_tensor(yhat), prx_yhat_train).mean().item()
    epsilon = torch.quantile(L, 0.33).item()

    res = dict(
        job=job_id,
        data=dname,
        exp_method=expname,
        proxy_method=pname,
        k=len(proxies.local_models),
        epsilon=epsilon,
        full_fidelity=full_fidelity,
        full_fidelity_train=full_fidelity_train,
        full_stability=L[torch.arange(L.shape[0])[:, None], nn].mean().item(),
        full_coverage=metrics.calculate_coverage(L < epsilon),
        full_loss_test=full_loss,
        bb_loss_test=bb_loss,
        loss_train=loss_train.mean().item(),
        loss_test=loss_test.mean().item(),
        proxy_fidelity=proxy_fidelity,
        proxy_fidelity_train=proxy_fidelity_train,
        proxy_coverage=metrics.calculate_coverage(reduced_L < epsilon),
        proxy_stability=expanded_L[torch.arange(expanded_L.shape[0])[:, None], nn]
        .mean()
        .item(),
    )
    return res
