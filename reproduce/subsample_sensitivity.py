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
import explainreduce.metrics as metrics
from explainreduce.utils import read_parquet
from reproduce.utils.hyperparameters import get_params, get_bb, get_data
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt

# import warnings
# warnings.simplefilter("error")

OUTPUT_DIR = RESULTS_DIR / "n_sensitivity"


def table_results(df: pd.DataFrame):
    """Produce a latex table of the results and save to the repo."""
    df = pd.DataFrame()
    for f in OUTPUT_DIR.glob("**/*.parquet"):
        try:
            new_df = pq.ParquetFile(f).read().to_pandas()
            if "0" in new_df.columns:
                new_df = new_df.drop(["0"], axis=1)
            df = pd.concat((new_df, df), ignore_index=True)
        except Exception as e:
            print(f"Failed to read {f}")
            print(e)
    # df.loc[:, 'full_fidelity'] = -(df['full_fidelity'] - 1)
    # df.loc[:, 'proxy_fidelity'] = -(df['proxy_fidelity'] - 1)
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
    exp_methods = ["LIME", "SHAP", "SLISEMAP", "SmoothGrad"]
    datasets = ["Gas Turbine", "Jets", "QM9"]
    p_df = df.loc[df["exp_method"].isin(exp_methods)]
    p_df = p_df.loc[p_df["data"].isin(datasets)]
    s_df = p_df.copy(deep=True)
    mean_df = (
        p_df.groupby(["data", "exp_method"])
        .mean(numeric_only=True)[["full_fidelity", "full_coverage", "bb_loss_test"]]
        .reset_index()
    )
    s_df.loc[:, "proxy_fidelity"] = s_df["sub_fidelity"]
    s_df.loc[:, "proxy_method_mod"] = "All local explanations"
    p_df = pd.concat((p_df, s_df), ignore_index=True)
    p_df = p_df.sort_values(["data", "exp_method"])
    p_df = p_df.rename(
        columns={"proxy_method_mod": "Reduction method", "proxy_fidelity": "Fidelity"}
    )
    g = sns.relplot(
        p_df,
        x="n",
        y="Fidelity",
        col="exp_method",
        row="data",
        hue="Reduction method",
        style="Reduction method",
        kind="line",
        facet_kws={"sharey": False},
        legend=None,
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
            g.axes[i, j].set_title(f"Dataset: {dname}, XAI: {exp_name}")
            i_f += 1
    for ax in g.axes.flat:
        for line, label, ci in zip(
            ax.get_lines(), p_df["Reduction method"].unique(), ax.collections
        ):
            if label == "All local explanations":
                line.set_linewidth(3)
                line.set_color("k")
                line.set_zorder(9)
                ci.set_color("k")
                ci.set_alpha(0.2)

    # Dynamically build the legend based on the final lines
    handles = []
    for line, label in zip(
        g.axes.flat[0].get_lines(), p_df["Reduction method"].unique()
    ):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=line.get_color(),  # Use the modified color
                lw=line.get_linewidth(),  # Use the modified line width
                linestyle=line.get_linestyle(),
                label=label,  # Label from the data
            )
        )
    g.figure.legend(
        handles=handles, title="Reduction method", bbox_to_anchor=(1.17, 0.60)
    )
    g.figure.savefig(MANUSCRIPT_DIR / "fidelity_n.pdf", dpi=600, bbox_inches="tight")
    exp_methods = ["LIME", "SHAP", "SLISEMAP", "SLIPMAP", "SmoothGrad"]
    p_df = df.loc[df["exp_method"].isin(exp_methods)]
    datasets = [x for x in p_df["data"] if x not in ["Synthetic"]]
    p_df = p_df.loc[p_df["data"].isin(datasets)]
    p_df = p_df.sort_values(["data", "exp_method"])
    p_df = p_df.rename(
        columns={"proxy_method_mod": "Reduction method", "proxy_fidelity": "Fidelity"}
    )
    g = sns.relplot(
        p_df,
        x="n",
        y="Fidelity",
        col="Reduction method",
        row="exp_method",
        hue="data",
        kind="line",
        facet_kws={"sharey": False},
    )
    i_f = 0
    for i, exp_name in enumerate(p_df["exp_method"].unique()):
        # set ylims based on dataset
        data_y_min, data_y_max = np.inf, -np.inf
        for j, rname in enumerate(p_df["Reduction method"].unique()):
            y_min, y_max = g.axes[i, j].get_ylim()
            if y_min < data_y_min:
                data_y_min = y_min
            if y_max > data_y_max:
                data_y_max = y_max
        for j, rname in enumerate(p_df["Reduction method"].unique()):
            # g.axes[i, j].axhline(mean_df.iloc[i_f]['full_fidelity'], color='gray', linestyle='--')
            g.axes[i, j].set_ylim((data_y_min, data_y_max))
            g.axes[i, j].set_title(f"Reduction: {rname}, XAI: {exp_name}")
            i_f += 1
    plt.savefig(MANUSCRIPT_DIR / "fidelity_n_datasets.pdf", dpi=600)
    g = sns.relplot(
        p_df,
        x="n",
        y="Fidelity",
        col="Reduction method",
        row="data",
        hue="exp_method",
        style="exp_method",
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
        for j, exp_name in enumerate(p_df["Reduction method"].unique()):
            # full_value = mean_df.loc[(mean_df['data'] == dname) &
            # (mean_df['exp_method'] == exp_name)][f'full_fidelity'].values[0]
            # g.axes[i, j].axhline(full_value, color='gray', linestyle='--')
            g.axes[i, j].set_ylim((data_y_min, data_y_max))
            g.axes[i, j].set_title(f"Dataset: {dname}, Reduction: {exp_name}")
            i_f += 1
    plt.savefig(MANUSCRIPT_DIR / "fidelity_n_exp_methods.pdf", dpi=600)


def table_results_full(df: pd.DataFrame):
    """Produce a latex table of the results and save to the repo."""
    df = pd.DataFrame()
    for f in OUTPUT_DIR.glob("**/*.parquet"):
        try:
            new_df = pq.ParquetFile(f).read().to_pandas()
            if "0" in new_df.columns:
                new_df = new_df.drop(["0"], axis=1)
            df = pd.concat((new_df, df), ignore_index=True)
        except Exception as e:
            print(f"Failed to read {f}")
            print(e)
    # df.loc[:, 'full_fidelity'] = -(df['full_fidelity'] - 1)
    # df.loc[:, 'proxy_fidelity'] = -(df['proxy_fidelity'] - 1)
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
    exp_methods = ["LIME", "SHAP", "SLISEMAP", "SmoothGrad"]
    datasets = df["data"].unique()
    p_df = df.loc[df["exp_method"].isin(exp_methods)]
    p_df = p_df.loc[p_df["data"].isin(datasets)]
    s_df = p_df.copy(deep=True)
    mean_df = (
        p_df.groupby(["data", "exp_method"])
        .mean(numeric_only=True)[["full_fidelity", "full_coverage", "bb_loss_test"]]
        .reset_index()
    )
    s_df.loc[:, "proxy_fidelity"] = s_df["sub_fidelity"]
    s_df.loc[:, "proxy_method_mod"] = "All local explanations"
    p_df = pd.concat((p_df, s_df), ignore_index=True)
    p_df = p_df.sort_values(["data", "exp_method"])
    p_df = p_df.rename(
        columns={"proxy_method_mod": "Reduction method", "proxy_fidelity": "Fidelity"}
    )
    g = sns.relplot(
        p_df,
        x="n",
        y="Fidelity",
        col="exp_method",
        row="data",
        hue="Reduction method",
        style="Reduction method",
        kind="line",
        facet_kws={"sharey": False},
        legend=None,
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
            g.axes[i, j].set_title(f"Dataset: {dname}, XAI: {exp_name}")
            i_f += 1
    for ax in g.axes.flat:
        for line, label, ci in zip(
            ax.get_lines(), p_df["Reduction method"].unique(), ax.collections
        ):
            if label == "All local explanations":
                line.set_linewidth(3)
                line.set_color("k")
                line.set_zorder(9)
                ci.set_color("k")
                ci.set_alpha(0.2)

    # Dynamically build the legend based on the final lines
    handles = []
    for line, label in zip(
        g.axes.flat[0].get_lines(), p_df["Reduction method"].unique()
    ):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=line.get_color(),  # Use the modified color
                lw=line.get_linewidth(),  # Use the modified line width
                linestyle=line.get_linestyle(),
                label=label,  # Label from the data
            )
        )
    g.figure.legend(
        handles=handles, title="Reduction method", bbox_to_anchor=(1.17, 0.60)
    )
    g.figure.savefig(
        MANUSCRIPT_DIR / "fidelity_n_full.pdf", dpi=600, bbox_inches="tight"
    )


def prepare_data(dataset_fun, seed):

    X, y, bb, cls = dataset_fun()
    n = min(10_000, X.shape[0])
    X, y = X[:n, :], y[:n]
    if y.ndim < 2:
        y = y[:, None]
    test_size = min(X.shape[0] - 1500, X.shape[0] // 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    train_size = 1000
    X_exp_train, _, y_exp_train, _ = train_test_split(
        X_train, y_train, train_size=train_size, random_state=seed
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


def get_optimisation_method(k: int = 5):
    return {
        (f"greedy_max_coverage_k_{k}", k): partial(px.find_proxies_greedy, k=k),
        (f"max_coverage_k_{k}", k): partial(
            px.find_proxies_coverage, k=k, time_limit=300, pulp_msg=False
        ),
        (f"random_k_{k}", k): partial(px.find_proxies_random, k=k),
        # (f"min_loss_{k}", k): partial(px.find_proxies_loss, k=k, time_limit=300),
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
    }


def evaluate(job_id: int, ns: list[int]) -> None:
    """Evaluate the full set of optimisation problems."""
    print(f"Begin job {job_id}.", flush=True)
    np.random.seed(1618 + job_id)
    torch.manual_seed(1618 + job_id)
    output_file = OUTPUT_DIR / f"n_sensitivity_{job_id:02d}.parquet"
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
        m, pred_fn = get_bb(bb_name, cls, X_bb, y_bb, dname)
        yhat_test = pred_fn(X_test)
        yhat_test = torch.as_tensor(yhat_test, dtype=y_bb.dtype)
        if yhat_test.ndim < y_bb.ndim:
            yhat_test = yhat_test[:, None]
        yhat_train = torch.as_tensor(pred_fn(X_bb), dtype=y_bb.dtype)
        if yhat_train.ndim < y_bb.ndim:
            yhat_train = yhat_train[:, None]
        loss_train = loss_fn(yhat_train, y_bb)
        global_epsilon = torch.quantile(loss_train, 0.33).item()
        for expname, expfn in get_explainer(dname, bb_name, cls).items():
            X, y = ds["exp_train"]
            D = torch.cdist(torch.as_tensor(X), torch.as_tensor(X))
            D += torch.eye(D.shape[0]) * torch.max(D)
            nn = torch.argsort(D, 1)[:, :5]
            yhat = pred_fn(X)
            explainer = expfn(X, yhat, black_box_model=m)
            explainer.fit()

            # if global_epsilon is None:
            # global_epsilon = metrics.calculate_global_epsilon(explainer)
            L = explainer.get_L()
            explainer_yhat_test = explainer.predict(X_test)
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

            print("Loop over n values.")
            ks = [5]
            k_options = [get_optimisation_method(k=k) for k in ks]
            k_options = reduce(operator.ior, k_options, {})
            for (pname, k), proxy_method in k_options.items():
                for n in ns:
                    if (
                        not results.empty
                        and (
                            (results["data"] == dname)
                            & (results["exp_method"] == expname)
                            & (results["proxy_method"] == pname)
                            & (results["k"] == k)
                            & (results["n"] == n)
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
                        n,
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


def eval_proxy_method(
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        table_results_full(df)
    else:
        ns = [25, 50, 100, 200, 300, 400, 500, 750, 1000]
        time = timer()
        evaluate(int(sys.argv[1]), ns=ns)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
