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

OUTPUT_DIR = RESULTS_DIR / "lambda_sensitivity"


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
    df.loc[df["proxy_method"].str.contains("balanced"), "proxy_method_mod"] = "Balanced"
    df = df.loc[df["proxy_method_mod"] == "Balanced"]
    df = df.loc[~df["exp_method"].isin(["VanillaGrad", "GlobalLinear", "SLIPMAP"])]
    return df


def plot_results(df: pd.DataFrame):
    df = preprocess_results(df)
    g = sns.relplot(
        df,
        x="lambda_weight",
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
            ax.set_xlabel(r"$\lambda$")
        if i % 4 == 0:
            ax.set_ylabel("Fidelity")
        ax.set_title(df["data"].unique()[i])
    g.fig.savefig(MANUSCRIPT_DIR / "lambda_sensitivity_fidelity.pdf", dpi=300)
    g = sns.relplot(
        df,
        x="lambda_weight",
        y="proxy_coverage",
        hue="exp_method",
        style="exp_method",
        col="data",
        col_wrap=4,
        kind="line",
        facet_kws={"sharey": False},
    )
    for i, ax in enumerate(g.axes):
        if i > 7:
            ax.set_xlabel(r"$\lambda$")
        if i % 4 == 0:
            ax.set_ylabel("Coverage")
        ax.set_title(df["data"].unique()[i])
    g.fig.savefig(MANUSCRIPT_DIR / "lambda_sensitivity_coverage.pdf", dpi=300)


def prepare_data(dataset_fun, job_id):

    X, y, bb, cls = dataset_fun()
    n = min(1_000, X.shape[0] * 3 // 4)
    if y.ndim < 2:
        y = y[:, None]
    if cls:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=n // 2,
            test_size=n // 2,
            random_state=42 + job_id,
            stratify=y[:, 1],
            shuffle=True,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n // 2, test_size=n // 2
        )
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
    exp_methods = {
        "SLISEMAP": lm.SLISEMAPExplainer,
        "SLIPMAP": lm.SLIPMAPExplainer,
        "VanillaGrad": lm.VanillaGradExplainer,
        "SmoothGrad": lm.SmoothGradExplainer,
        "LIME": lm.LIMEExplainer,
        "SHAP": lm.KernelSHAPExplainerLegacy,
        "GlobalLinear": lm.GlobalLinearExplainer,
    }
    if cls:
        exp_methods["LORE"] = lm.LORERuleExplainer
    out = {}
    for exp, model_obj in exp_methods.items():
        if exp not in ["SHAP", "LORE", "VanillaGrad"]:
            params = get_params(exp, dname, bb)
            if params is not None:
                out[exp] = partial(model_obj, classifier=cls, **params)
            else:
                print(
                    f"Did not find hyperparams for {dname} - {bb} - {exp}, please run hyperopt first!",
                    flush=True,
                )
        else:
            out[exp] = partial(model_obj, classifier=cls)
    return out


def get_optimisation_method(lambda_weight, k):
    return {
        (f"greedy_balanced_k_{k}", lambda_weight): partial(
            px.find_proxies_loss_cov_linear, k=k, lambda_weight=lambda_weight
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


def evaluate(job_id: int, lambdas: list) -> None:
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
        ds = prepare_data(ds_fn, job_id)
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
        if yhat_test.ndim < y.ndim:
            yhat_test = yhat_test[:, None]
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
            l_options = [get_optimisation_method(lambda_weight=l, k=6) for l in lambdas]
            l_options = reduce(operator.ior, l_options, {})
            for (pname, lambda_weight), proxy_method in l_options.items():
                if (
                    not results.empty
                    and (
                        (results["data"] == dname)
                        & (results["exp_method"] == expname)
                        & (results["proxy_method"] == pname)
                        & (results["init_l"] == lambda_weight)
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
                    lambda_weight,
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
    lambda_weight,
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
        f"Reducing: {dname} - {expname} - {pname} - lambda={lambda_weight}",
        flush=True,
    )
    try:
        proxies = proxy_method(explainer)
    except ValueError as ve:
        print(f"Reduction method resulted in error:\n", ve, flush=True)
        return
    print(
        f"Evaluating: {dname} - {expname} - {pname} - lambda={lambda_weight}",
        flush=True,
    )
    prx_yhat_train = proxies.predict(X)
    prx_yhat_test = proxies.predict(X_test)
    loss_train = loss_fn(torch.as_tensor(y), prx_yhat_train)
    loss_test = loss_fn(torch.as_tensor(y_test), prx_yhat_test)
    reduced_L = proxies.get_L()
    expanded_L = reduced_L[list(proxies.mapping.values()), :]
    proxy_fidelity = loss_fn(torch.as_tensor(yhat_test), prx_yhat_test).mean().item()
    epsilon = torch.quantile(loss_train, q=0.33).item()

    res = dict(
        job=job_id,
        data=dname,
        exp_method=expname,
        proxy_method=pname,
        k=len(proxies.local_models),
        lambda_weight=lambda_weight,
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
        lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        time = timer()
        evaluate(int(sys.argv[1]), lambdas=lambdas)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
