"""Contains functions used for reproducing the experiments in the paper."""

import torch
from sklearn.model_selection import train_test_split
from functools import partial
from reproduce.utils.hyperparameters import get_params
import explainreduce.localmodels as lm
import explainreduce.proxies as px
import explainreduce.metrics as metrics
from explainreduce.utils import read_parquet
import numpy as np
from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
import pandas as pd
import operator
from functools import reduce
from reproduce.utils.hyperparameters import get_bb, get_data


OUTPUT_DIR = RESULTS_DIR / "k_sensitivity"


def prepare_data(dataset_fun, seed):

    X, y, bb, cls = dataset_fun()

    # Limit the data size to a maximum of 10,000 samples
    n = min(10_000, X.shape[0])
    X, y = X[:n, :], y[:n]

    # Ensure y has at least 2 dimensions
    if y.ndim < 2:
        y = y[:, None]

    # Split the data into training and testing sets, with a 50/50 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed
    )

    # Further split the training to get a smaller training subset for experiments
    X_exp_train, _, y_exp_train, _ = train_test_split(
        X_train, y_train, train_size=500, random_state=seed
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


def get_optimisation_method(k: int):
    """Define the reducing methods to be used in the experiments."""
    return {
        (f"greedy_max_coverage_k_{k}", k): partial(px.find_proxies_greedy, k=k),
        (f"max_coverage_k_{k}", k): partial(
            px.find_proxies_coverage, k=k, time_limit=100, pulp_msg=False
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
            k_options = [get_optimisation_method(k=k) for k in ks]
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
                # Evaluate the proxy method and store results
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


def evaluate_subsample_sensitivity(job_id: int, ns: list[int]) -> None:
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
                    res = eval_proxy_method_subsample_sensitivity(
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
