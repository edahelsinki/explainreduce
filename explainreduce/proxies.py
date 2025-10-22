"""
This module contains the methods which generate proxy sets from local explanations.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from pathlib import Path
from configparser import ConfigParser

curr_path = Path(__file__)
config = ConfigParser()
config.read(str(curr_path.parent.parent / "config.ini"))
GLOCALX_PATH = config["Paths"]["GLOCALX_PATH"]
sys.path.append(GLOCALX_PATH)

import pandas as pd
import pulp
import torch
import numpy as np
from typing import Any, Union, List, Callable, Dict, Tuple
from numpy.typing import ArrayLike
from sklearn.cluster import AgglomerativeClustering, Birch, OPTICS, SpectralClustering
from sklearn.metrics import pairwise_distances, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
import pulp
from pulp import PULP_CBC_CMD
from warnings import warn
from localmodels import (
    Explainer,
    logistic_regression_loss,
    LORERuleExplainer,
    LIMEExplainer,
)
from sklearn.cluster import KMeans
import explainreduce.metrics as metrics
import warnings


def get_global_surrogate_fidelity(X: pd.DataFrame, model: Any, mode: str) -> float:
    """Train a global surrogate model, calculate and return the fidelity."""
    y_preds = model.predict(X)
    if mode == "regression":
        surrogate_model = LinearRegression().fit(X, y_preds)
        fidelity = torch.nn.MSELoss(reduction="mean")(
            torch.tensor(surrogate_model.predict(X), dtype=torch.float32),
            torch.tensor(y_preds, dtype=torch.float32),
        )
        fidelity = mean_squared_error(y_preds, surrogate_model.predict(X))
    else:
        surrogate_model = LogisticRegression().fit(X, y_preds)
        coef = surrogate_model.coef_.flatten()
        intercept = surrogate_model.intercept_[0]
        log_odds = np.dot(X, coef) + intercept
        prob = 1 / (1 + np.exp(-log_odds))
        prob_tensor = torch.tensor(prob, dtype=torch.float32)
        proba_tensor = torch.stack([1 - prob_tensor, prob_tensor], dim=1)
        targets_tensor = torch.tensor(y_preds, dtype=torch.long)
        criterion = lambda y, yhat: logistic_regression_loss(y, yhat).mean()
        fidelity = criterion(proba_tensor, targets_tensor)
    return fidelity.item()


def generate_minimal_loss_mapping(
    explainer: Explainer, reduced_models: List[Callable[[torch.Tensor], torch.Tensor]]
) -> Dict[int, int]:
    """Generate mapping from X to proxies that minimises loss for individual items."""
    reduced_L = torch.empty((explainer.X.shape[0], len(reduced_models)))
    y = explainer.y
    if y.ndim < 2:
        y = y[:, None]
    for i, m in enumerate(reduced_models):
        yhat = m(explainer.X)
        if yhat.ndim < 2:
            yhat = yhat[:, None]
        loss_vector = explainer.loss_fn(yhat.float(), y.float())
        if loss_vector.ndim > 1:
            loss_vector = torch.mean(loss_vector, dim=-1)
        reduced_L[:, i] = loss_vector
    best_models = torch.argmin(reduced_L, dim=1)
    return {i: best_models[i].item() for i in range(explainer.X.shape[0])}


# Optimisation-based methods
def find_proxies_loss(
    explainer: Explainer, k: int = None, time_limit: int = None, pulp_msg=True
) -> Explainer:
    """Get a new explainer with k proxies which minimise the loss on the training set."""
    if not explainer.is_fit:
        print("Explainer object not fitted, fitting.")
        explainer.fit()

    L = explainer.get_L()
    n_proxies, n_instances = L.shape
    L_np = L.numpy()

    # Create the LP problem and decision variables``
    problem = pulp.LpProblem("Loss_Selection", pulp.LpMinimize)
    x = [pulp.LpVariable(f"Proxy_{i}", cat="Binary") for i in range(n_proxies)]
    y = [
        pulp.LpVariable(f"Instance_{j}", lowBound=0, cat="Continuous")
        for j in range(n_instances)
    ]

    # Minimize the loss while selecting k proxies
    problem += pulp.lpSum(y[j] for j in range(n_instances)), "Total Loss"
    problem += pulp.lpSum(x[i] for i in range(n_proxies)) == k, "Select k proxies"
    for j in range(n_instances):
        for i in range(n_proxies):
            problem += y[j] >= L_np[i][j] * x[i]

    # Solve the problem
    if time_limit is not None:
        problem.solve(PULP_CBC_CMD(msg=pulp_msg, timeLimit=time_limit))
    else:
        problem.solve()

    # Retrieve the indices of the selected proxies
    selected_proxies = [i for i in range(n_proxies) if pulp.value(x[i]) > 0.5]
    if not selected_proxies:
        raise ValueError(
            f"TimeLimitNotEnough: Did not find a solution in within time limit {time_limit}."
        )

    # Create a new explainer that only contains the selected proxies
    reduced_models = [explainer.local_models[i] for i in selected_proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[selected_proxies, :]
    reduced = explainer.clone_reduced(reduced_models, mapping, vector_representation)

    return reduced


def find_proxies_coverage(
    explainer: Explainer,
    k: int,
    epsilon: float = None,
    p: float = None,
    time_limit: int = None,
    pulp_msg=True,
) -> Explainer:
    """Find a set of k proxies which attain maximal coverage."""
    if not explainer.is_fit:
        print("Explainer object not fitted, fitting.")
        explainer.fit()

    L = explainer.get_L()
    n_proxies, n_instances = L.shape
    L_np = L.numpy()
    if epsilon is None and p is not None:
        epsilon = np.quantile(L_np, p)
    G_eps = (L_np < epsilon).astype(int)

    if not G_eps.any():
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon})"
        )
    else:
        # Create the LP problem and decision variables
        problem = pulp.LpProblem("Coverage_Selection", pulp.LpMaximize)
        x = [pulp.LpVariable(f"Proxy_{i}", cat="Binary") for i in range(n_proxies)]
        y = [pulp.LpVariable(f"Instance_{j}", cat="Binary") for j in range(n_instances)]

        # Maximize the coverage while selecting k proxies
        problem += pulp.lpSum(y[j] for j in range(n_instances)), "Total Coverage"
        problem += pulp.lpSum(x[i] for i in range(n_proxies)) == k, "Select k proxies"
        for j in range(n_instances):
            problem += (
                y[j] <= pulp.lpSum(G_eps[i][j] * x[i] for i in range(n_proxies)),
                f"Coverage_Upper_bound_Instance_{j}",
            )
            for i in range(n_proxies):
                problem += (
                    y[j] >= G_eps[i][j] * x[i],
                    f"Coverage_Lower_bound_Instance_{j}_Proxy_{i}",
                )

        # Solve the problem
        if time_limit is not None:
            problem.solve(PULP_CBC_CMD(msg=pulp_msg, timeLimit=time_limit))
        else:
            problem.solve()

        # Retrieve the indices of the selected proxies
        selected_proxies = [i for i in range(n_proxies) if pulp.value(x[i]) > 0.5]

    if not selected_proxies:
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon})"
        )
    reduced_models = [explainer.local_models[i] for i in selected_proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[selected_proxies, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, selected_proxies
    )

    return reduced


def find_proxies_random(explainer: Explainer, k: int, epsilon=None) -> Explainer:
    """Find a random set of k proxies."""
    n_models = len(explainer.local_models)
    assert k <= n_models, f"Cannot pick {k} proxies from {n_models} local models!"
    proxies = list([x.item() for x in torch.randperm(n_models)[:k]])
    reduced_models = [explainer.local_models[i] for i in proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[proxies, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, proxies
    )
    return reduced


# Clustering-based methods
def kmeanspp_select_mean(data: np.ndarray, k: int):
    kmeans = KMeans(n_clusters=k, init="k-means++").fit(data)
    labels = kmeans.predict(data)
    pid = []
    for cluster in range(k):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) == 0:
            continue
        distances_to_center = np.linalg.norm(
            data[cluster_indices] - kmeans.cluster_centers_[cluster], axis=1
        )
        mean_index = cluster_indices[
            np.argsort(distances_to_center)[len(cluster_indices) // 2]
        ]
        pid.append(mean_index)
    return pid


def _map_dist_metric(dist_metric: str):
    if dist_metric == "euclidean":
        return lambda p1, p2: np.linalg.norm(p1 - p2)
    elif dist_metric == "manhattan":
        return lambda p1, p2: np.sum(np.abs(p1 - p2))
    elif dist_metric == "cosine":
        return lambda p1, p2: max(
            0, 1 - np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
        )
    elif dist_metric == "jaccard":
        return lambda p1, p2: max(
            0, 1 - np.sum(np.minimum(p1, p2)) / (np.sum(np.maximum(p1, p2)) or 1)
        )
    elif dist_metric == "hamming":
        return lambda p1, p2: np.mean(p1 != p2)
    elif dist_metric == "hanimoto":
        return lambda p1, p2: max(
            0, 1 - (np.dot(p1, p2) / ((np.linalg.norm(p1) * np.linalg.norm(p2)) or 1))
        )
    else:
        raise ValueError(
            f"Unsupported distance metric: {dist_metric}, choose from "
            + "['euclidean','manhattan','cosine','jaccard','hamming','hanimoto']"
        )


def _select_representatives(
    cluster_indices: ArrayLike, data: np.ndarray, labels: ArrayLike, dist_func
):
    representative_indices = []
    for cluster_id in cluster_indices:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_data = data[cluster_indices]
        pairwise_dist = pairwise_distances(cluster_data, metric=dist_func)
        total_dist = np.sum(pairwise_dist, axis=1)
        representative_indices.append(cluster_indices[np.argmin(total_dist)])
    return representative_indices


def agglomerative_select(data: np.ndarray, k: int, dist: str):
    # Get distance function and compute distance matrix
    dist_func = _map_dist_metric(dist)
    distance_matrix = pairwise_distances(data, metric=dist_func)
    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit(distance_matrix)
    # Select representative point from each cluster
    representative_indices = _select_representatives(
        cluster_indices=range(k),
        data=data,
        labels=clustering.labels_,
        dist_func=dist_func,
    )
    return representative_indices


def optics_select(data: np.ndarray, k: int, dist: str, min_samples: int = 5):
    # Get distance function and compute distance matrix
    dist_func = _map_dist_metric(dist)
    distance_matrix = pairwise_distances(data, metric=dist_func)
    # Perform OPTICS clustering
    clustering = OPTICS(min_samples=min_samples, metric="precomputed").fit(
        distance_matrix
    )
    # Get cluster labels, exclude noise points
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)
    # Judge if the number of clusters is valid
    if len(unique_labels) == 0:
        raise ValueError("No cluster found.")
    if len(unique_labels) < k:
        k = len(unique_labels)
    # Extract K largest clusters' labels
    cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    largest_clusters = [label for label, _ in cluster_sizes[:k]]
    # Select representative point from each cluster
    representative_indices = _select_representatives(
        cluster_indices=largest_clusters,
        data=data,
        labels=clustering.labels_,
        dist_func=dist_func,
    )
    return representative_indices


def spectral_select(data: np.ndarray, k: int, dist: str):
    # Perform spectral clustering
    clustering = SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors", assign_labels="cluster_qr"
    ).fit(data)
    # Select representative point from each cluster
    representative_indices = _select_representatives(
        cluster_indices=range(k),
        data=data,
        labels=clustering.labels_,
        dist_func=_map_dist_metric(dist),
    )
    return representative_indices


def birch_select(data: np.ndarray, k: int, dist: str):
    # Perform BIRCH clustering
    clustering = Birch(n_clusters=None).fit(data)
    labels = clustering.labels_
    unique_labels = set(labels)
    # Judge if the number of clusters is valid
    if len(unique_labels) == 0:
        raise ValueError("No cluster found.")
    if len(unique_labels) < k:
        k = len(unique_labels)
    # Extract K largest clusters' labels
    cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    largest_clusters = [label for label, _ in cluster_sizes[:k]]
    # Select representative point from each cluster
    representative_indices = _select_representatives(
        cluster_indices=largest_clusters,
        data=data,
        labels=clustering.labels_,
        dist_func=_map_dist_metric(dist),
    )
    return representative_indices


def find_proxies_clustering(
    explainer: Explainer,
    data: Union[torch.Tensor, pd.DataFrame],
    n_clusters: int,
    method: str,
    dist: str,
):
    """
    Parameters:
        explainer: Explainer object.
        data: torch.Tensor or pd.DataFrame. (X, B, L, Geps)
        n_clusters: Number of clusters to form.
        method: ['kmeans','agglomerative','optics','spectral','birch']
        dist: ['euclidean','manhattan','cosine','jaccard','hamming','hanimoto']

    Returns:
        reduced: Explainer object with reduced local models.
    """
    # Convert data to numpy
    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy()
    elif isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        raise ValueError(
            f"Input data must be either a torch.Tensor or a pandas.DataFrame."
        )
    # Apply clustering method
    if method == "kmeans":
        pids = kmeanspp_select_mean(data_np, n_clusters)
    elif method == "agglomerative":
        pids = agglomerative_select(data_np, n_clusters, dist)
    elif method == "optics":
        pids = optics_select(data_np, n_clusters, dist)
    elif method == "spectral":
        pids = spectral_select(data_np, n_clusters, dist)
    elif method == "birch":
        pids = birch_select(data_np, n_clusters, dist)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    reduced_models = [explainer.local_models[i] for i in pids]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[pids, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, pids
    )

    return reduced


def find_proxies_clustering_wrapper(
    explainer: Explainer,
    data: str,
    n_clusters: int,
    method: str,
    dist: str,
    epsilon=None,
):
    """
    Parameters:
        explainer: Explainer object.
        data: torch.Tensor or pd.DataFrame. (X, B, L, Geps)
        n_clusters: Number of clusters to form.
        method: ['kmeans','agglomerative','optics','spectral','birch']
        dist: ['euclidean','manhattan','cosine','jaccard','hamming','hanimoto']

    Returns:
        reduced: Explainer object with reduced local models.
    """
    # Convert data to numpy
    match data:
        case "data":
            data = explainer.X
        case "local_models":
            data = explainer.vector_representation
        case "loss":
            data = explainer.get_L()
        case "applicability":
            assert (
                epsilon is not None
            ), "For applicability calculation, epsilon must be set!"
            data = metrics.calculate_coverage(explainer.get_L() < epsilon)
        case _:
            raise NotImplementedError(
                "Data argument must be one of ['data', 'local_models', 'loss', 'applicability']"
            )
    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy()
    elif isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        raise ValueError(
            f"Input data must be either a torch.Tensor or a pandas.DataFrame."
        )
    # Apply clustering method
    if method == "kmeans":
        pids = kmeanspp_select_mean(data_np, n_clusters)
    elif method == "agglomerative":
        pids = agglomerative_select(data_np, n_clusters, dist)
    elif method == "optics":
        pids = optics_select(data_np, n_clusters, dist)
    elif method == "spectral":
        pids = spectral_select(data_np, n_clusters, dist)
    elif method == "birch":
        pids = birch_select(data_np, n_clusters, dist)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    reduced_models = [explainer.local_models[i] for i in pids]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[pids, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, pids
    )

    return reduced


# Greedy methods
def find_proxies_greedy(explainer: Explainer, k: int, p=0.33, epsilon=None):
    """Greedily search for the set of k proxies which attain maximal coverage."""

    def greedy_max_coverage(G: torch.Tensor, k: int):
        _, num_cols = G.shape
        covered_columns = torch.zeros(num_cols, dtype=torch.bool)
        selected_rows = []

        for _ in range(k):
            # Calculate the new coverage for each row in parallel
            new_coverage = torch.sum(G & ~covered_columns, dim=1)

            # Mask already selected rows by setting their coverage to -1 (or a very low value)
            if selected_rows:
                new_coverage[selected_rows] = -1

            # Break if no new coverage is possible
            if new_coverage.max() <= 0:
                break

            # Find the row with the maximum new coverage
            best_row = torch.argmax(new_coverage).item()
            selected_rows.append(best_row)

            # Update the covered columns and selected rows
            covered_columns |= G[best_row]

        return selected_rows

    def calculate_coverage(G_eps: torch.Tensor, prototypes: list[int]) -> float:
        """Calculate the coverage (ratio of data points covered by prototypes)."""
        G = G_eps[prototypes, :]
        covered_columns = torch.any(G, dim=0)
        coverage = torch.sum(covered_columns) / G_eps.shape[1]
        return coverage

    # Obtain the distance or loss matrix
    L = explainer.get_L()
    if epsilon is None and p is not None:
        epsilon = torch.quantile(L, p)
    G_eps = L < epsilon

    # Find proxies using the greedy algorithm
    proxies = greedy_max_coverage(G_eps, k=k)

    # Set a minimum epsilon to prevent it from becoming too small
    MIN_EPSILON = torch.min(L) * 0.9
    MAX_ITER = 100
    iter_count = 0

    coverage = 1.0
    while (
        (coverage >= 1.0 or not G_eps.any())
        and epsilon > MIN_EPSILON
        and iter_count < MAX_ITER
    ):
        iter_count += 1
        proxies = greedy_max_coverage(G_eps, k=k)
        coverage = calculate_coverage(G_eps, proxies)
        if coverage >= 1.0 or not proxies:
            epsilon *= 0.8
            G_eps = L < epsilon

    if not proxies:
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon})"
        )

    reduced_models = [explainer.local_models[i] for i in proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[proxies, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, proxies
    )

    return reduced


def find_proxies_greedy_min_loss(
    explainer: Explainer,
    min_coverage: float = 0.8,
    p: float = 0.33,
    epsilon: float = None,
):
    """Greedily search for the minimal set of proxies which attain a minimum coverage while
    minimising the average loss of the proxies on the training set."""

    L = explainer.get_L()
    if epsilon is None:
        epsilon = torch.quantile(L, p)
    G_eps = L < epsilon
    if not G_eps.any():
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon})"
        )
    mod_L = L.detach().clone()
    proxies = []
    total_loss = torch.inf
    for _ in range(L.shape[1]):
        min_idx = torch.argmin(torch.mean(mod_L, dim=1)).item()
        no_improvement = (
            torch.mean(L[proxies + [min_idx], :].min(dim=0).values) >= total_loss
        )
        coverage_met = metrics.calculate_coverage(G_eps, proxies) >= min_coverage
        if no_improvement:
            print("Can no longer improve.")
            break
        if coverage_met:
            print("Met minimum coverage.")
            break
        for j in range(mod_L.shape[1]):
            mod_L[j, :] = torch.minimum(mod_L[j, :], L[min_idx])
        proxies.append(min_idx)
        total_loss = torch.mean(L[proxies, :].min(dim=0).values)

    if not proxies:
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon})"
        )
    reduced_models = [explainer.local_models[i] for i in proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[proxies, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, proxies
    )

    return reduced


def find_proxies_greedy_min_loss_k_min_cov(
    explainer: Explainer,
    k: int,
    min_coverage: float = 0.8,
    p: float = 0.33,
    epsilon: float = None,
    raise_on_infeasible=True,
):
    """Greedily search for the set of k proxies which attain a minimum coverage while
    minimising the average loss of the proxies on the training set."""

    def marginal_coverage(G_eps, proxies):
        n_models, n_items = G_eps.shape
        if not proxies:
            covered = torch.zeros(n_items)
        else:
            covered = torch.sum(G_eps[proxies, :], dim=0) > 0
        n_covered = torch.sum(covered)
        marginal_coverages = torch.zeros(n_models)
        for i in range(n_models):
            marginal_coverages[i] = (
                torch.sum(torch.logical_or(G_eps[i, :], covered)) - n_covered
            ) / n_items
        return marginal_coverages

    L = explainer.get_L()
    if epsilon is None:
        epsilon = torch.quantile(L, p)
    G_eps = L < epsilon
    if not G_eps.any():
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon})"
        )
    # marginal loss tensor
    marginal_L = L.detach().clone()
    proxies = []
    for ik in range(k):
        # calculate marginal loss for each model
        mean_loss = torch.mean(marginal_L, dim=1)
        # when coverage has not been met, normalise loss values by their coverage increase;
        # more marginal coverage, smaller value
        coverage_met = metrics.calculate_coverage(G_eps, proxies) >= min_coverage
        if not coverage_met:
            mean_loss /= marginal_coverage(G_eps, proxies)
        min_idx = torch.argmin(mean_loss)
        for j in range(marginal_L.shape[0]):
            marginal_L[j, :] = torch.minimum(marginal_L[j, :], L[min_idx, :])
        proxies.append(min_idx)

    if not proxies:
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon})"
        )
    final_coverage = metrics.calculate_coverage(G_eps, proxies)
    if final_coverage < min_coverage:
        msg = (
            "Greedy algorithm could not find a feasible solution! Consider "
            + f"relaxing the constraints (min coverage={min_coverage}, "
            + f"epsilon={epsilon}). Final coverage was {final_coverage:.2f}."
        )
        if raise_on_infeasible:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    reduced_models = [explainer.local_models[i] for i in proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[proxies, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, proxies
    )

    return reduced


def find_proxies_greedy_k_min_loss(explainer: Explainer, k: int, epsilon=None):
    """Greedily search for a set of k proxies which attain minimum total loss."""

    L = explainer.get_L()
    mod_L = L.detach().clone()
    proxies = []
    total_loss = torch.inf
    for _ in range(k):
        min_idx = torch.argmin(torch.sum(mod_L, dim=1)).item()
        current_loss = torch.min(mod_L[proxies + [min_idx], :], dim=0)[0].sum().item()
        if current_loss >= total_loss:
            print("Cannot improve, stopping.")
            break
        else:
            proxies.append(min_idx)
            mod_L = torch.min(mod_L, mod_L[min_idx, :])
            total_loss = current_loss

    reduced_models = [explainer.local_models[i] for i in proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[proxies, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, proxies
    )

    return reduced


def find_proxies_min_loss(
    explainer: Explainer,
    k: int,
    min_coverage: float = 0.8,
    epsilon: float = None,
    p: float = None,
    time_limit: int = None,
):
    """Find the minimal set of proxies which attain a minimum coverage while minimising the loss"""
    # depending on the hyperparams, this may be impossible!
    warn("Possibly slow; finding the optimal set requires 2^D comparisons!")
    L = explainer.get_L()
    if epsilon is None and p is not None:
        epsilon = torch.quantile(L, p)
    G_eps = L < epsilon
    nrows, ncols = G_eps.shape
    target_cols_covered = int(np.ceil(min_coverage * ncols))

    # set up the minimisation problem
    problem = pulp.LpProblem("Min_loss", pulp.LpMinimize)

    # decision variables
    # possible proxy sets
    possible_sets = [tuple(s) for s in pulp.allcombinations(range(nrows), k)]
    sets = pulp.LpVariable.dicts("proxy_sets", possible_sets, cat="Binary")
    # covered columns
    cols = pulp.LpVariable.dicts("col_covered", range(ncols), cat="Binary")

    # objective function
    # this cannot be correct
    problem += (
        pulp.lpSum([torch.mean(L[[i], :]) * sets[i] for i in possible_sets]),
        "Minimize loss",
    )

    # allow flipping of cols to covered
    for j in range(ncols):
        problem += (
            pulp.lpSum(torch.sum(G_eps[s, j]) * sets[s] for s in possible_sets) >= 0
        ) >= cols[j], f"Allow flipping of col[{j}] to covered"

    # actual constraint: attain minimum coverage
    problem += pulp.lpSum(cols[i] for i in range(ncols)) >= target_cols_covered

    if time_limit is not None:
        problem.solve(PULP_CBC_CMD(msg=True, timeLimit=time_limit))
    else:
        problem.solve()

    selected_rows = list([s for s in possible_sets if sets[s]][0])
    reduced_models = [explainer.local_models[i] for i in selected_rows]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[selected_rows, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, selected_rows
    )
    return reduced


def find_proxies_minimal_set(
    explainer: Explainer,
    min_coverage: float = 0.8,
    p: float = None,
    epsilon: float = None,
    time_limit: int = None,
) -> Explainer:
    """Find a minimal set of proxies which attain a minimum coverage constraint.

    Args:
        explainer: a trained explainer object
        min_coverage: the coverage threshold. Defaults to 0.8 or 80%.
        p: proportion of applicable models for automatically setting epsilon. if p is
            set, epsilon is calculated as the p-th quantile of L.
        epsilon: the maximum allowed loss value for a model to be considered applicable
            for coverage calculation.

    Returns:
        A new explainer object with a minimal set of proxies which attain at a set level
        of coverage.
    """
    assert (p is not None) or (epsilon is not None), "Specify either epsilon or p!"
    if not explainer.is_fit:
        print("Explainer object not fitted, fitting.")
        explainer.fit()
    L = explainer.get_L()
    if epsilon is None:
        epsilon = torch.quantile(L, p)
    G_eps = (L < epsilon).numpy()
    N, M = G_eps.shape
    target_coverage = int(np.ceil(min_coverage * M))

    # Create the LP problem
    problem = pulp.LpProblem("Minimal_Proxy_Selection", pulp.LpMinimize)

    # Decision variables
    lms = pulp.LpVariable.dicts("x", range(N), cat="Binary")
    items = pulp.LpVariable.dicts("y", range(M), cat="Binary")

    # Objective: Minimize the number of selected proxies
    problem += (
        pulp.lpSum(lms[i] for i in range(N)),
        "Minimize proxies count",
    )

    # Constraints: link proxies to instance coverage
    for j in range(N):
        problem += (
            pulp.lpSum(G_eps[i, j] * lms[i] for i in range(N)) >= items[j],
            f"Instance_{j}_coverage",
        )
        problem += (
            items[j] <= 1,
            f"Instance_{j}_coverage_limit",
        )

    # Ensure the total coverage meets or exceeds the target coverage
    problem += (
        pulp.lpSum(items[j] for j in range(M)) >= target_coverage,
        "Total_coverage_requirement",
    )

    # Solve the problem
    if time_limit is not None:
        problem.solve(PULP_CBC_CMD(msg=True, timeLimit=time_limit))
    else:
        problem.solve()

    # Retrieve the indices of the selected proxies
    selected = [i for i in range(N) if pulp.value(lms[i]) == 1]
    if not selected:
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(epsilon={epsilon}, coverage={min_coverage})"
        )

    reduced_models = [explainer.local_models[i] for i in selected]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[selected, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, selected
    )

    return reduced


def find_proxies_loss_recursive(explainer: Explainer, k: int, epsilon=None):

    # helper methods to use binary numbers as keys in the cache
    def find_first_nonzero_bit(x: int):
        if not x:
            return None
        idx = 0
        while True:
            stop = x & 1
            if stop:
                return idx
            idx += 1
            x = x >> 1

    def list_non_zero_bits(chosen: int):
        indices = []
        idx = 0
        while chosen:
            if chosen & 1:
                indices.append(idx)
            chosen = chosen >> 1
            idx += 1
        return indices

    def build_search_space(chosen, mod_L, cache, search_space):
        """Build/prune the search space and cache the loss values of choices."""
        indices_to_search = torch.zeros(N, dtype=int)
        # calculate current loss vector
        if chosen:
            base_loss_vector = mod_L[find_first_nonzero_bit(chosen), :]
        else:
            # start of the search: set base loss vector to infinity
            base_loss_vector = torch.ones(M) * torch.inf

        # fetch current total loss
        if chosen in cache:
            # base loss should always be cached (except at the start)
            base_loss = cache[chosen]
        else:
            # start of the search: set base loss to infinity
            base_loss = torch.inf

        # prune the search space
        for i in search_space:
            # if i has already been chosen, no need to search again
            if (1 << i) & chosen:
                continue

            key = (1 << i) | chosen
            # if key has been cached, no need to search again
            if key in cache:
                continue
            # calculate and cache the loss value
            loss = torch.min(mod_L[i, :], base_loss_vector).sum().item()
            cache[key] = loss
            # if no marginal decrease in loss, no need to include i in the search space
            if loss >= base_loss:
                continue
            # else include i
            indices_to_search[i] = 1
        return cache, torch.where(indices_to_search)[0].tolist()

    def inner(chosen, mod_L, cache, search_space, best_loss, best_choice):
        # terminate if k choices have been reached
        if chosen.bit_count() == k:
            return best_choice, best_loss
        # initialize search space (only needed at the start/root node)
        if search_space is None:
            search_space = list(range(mod_L.shape[0]))

        cache, search_space = build_search_space(chosen, mod_L, cache, search_space)

        # if there aren't any models the addition of which would improve the solution, terminate
        if len(search_space) == 0:
            return best_choice, best_loss

        for i in search_space:
            new_chosen = (1 << i) | chosen
            # all losses should be cached from building the search space
            loss = cache[new_chosen]
            # update best loss if needed
            if loss < best_loss:
                best_loss = loss
                best_choice = new_chosen
            new_mod_L = torch.min(mod_L, mod_L[i, :])
            # recursive call
            new_chosen, loss = inner(
                new_chosen, new_mod_L, cache, search_space, best_loss, best_choice
            )
            if loss < best_loss:
                best_loss = loss
                best_choice = new_chosen
        return best_choice, best_loss

    best_loss = torch.inf
    best_choices = None
    chosen = 0
    cache = {}
    mod_L = explainer.get_L().clone()
    N, M = mod_L.shape
    final_choices, final_loss = inner(
        chosen, mod_L, cache, None, best_loss, best_choices
    )
    final_choices = list_non_zero_bits(final_choices)

    reduced_models = [explainer.local_models[i] for i in final_choices]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[final_choices, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, final_choices
    )

    return reduced


def find_proxies_greedy_k_min_loss_descent(explainer: Explainer, k: int, epsilon=None):
    """Greedily search for a set of k proxies which attain minimum total loss with a
    worst-out approach."""

    L = explainer.get_L()
    n_proxies, _ = L.shape
    proxies = [True] * n_proxies
    for _ in range(n_proxies - k):
        min_loss = torch.inf
        worst = None
        for i, proxy_on in enumerate(proxies):
            if not proxy_on:
                continue
            proxies[i] = False
            current_loss = torch.min(L[proxies, :], dim=0)[0].sum().item()
            if current_loss < min_loss:
                min_loss = current_loss
                worst = i
            proxies[i] = True
        if worst is None:
            print("Cannot improve!")
            break
        proxies[worst] = False

    prx = [i for i, p in enumerate(proxies) if p]
    reduced_models = [explainer.local_models[i] for i in prx]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[prx, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, prx
    )

    return reduced


def find_proxies_submodular_pick(
    explainer: LIMEExplainer,
    k: int,
    sample_size: int = 20,
    num_features: int = 14,
    epsilon=None,
):
    assert "LIMEExplainer" in str(
        explainer.__class__
    ), f"Submodular pick reduction is only available with LIMEExplainer, got {type(explainer)}!"
    from lime import submodular_pick

    sp_obj = submodular_pick.SubmodularPick(
        explainer._LIME_explainer,
        tonp(explainer.X),
        explainer.black_box_predict(),
        sample_size=sample_size,
        num_features=num_features,
        num_exps_desired=k,
    )
    discretize = explainer.explainer_kwargs.get("discretize", False)

    def explain(i):

        def _sp_predict(X):
            X = tonp(X)
            if discretize:
                X = sp_obj.discretizer.discretize(X) == x1
            Y = np.sum(X * b, -1, keepdims=True) + inter

            if explainer.classifier:
                Y = np.clip(Y, 0.0, 1.0)
                if exp_label == 0:
                    Y = np.concatenate((Y, 1.0 - Y), -1)
                else:
                    Y = np.concatenate((1.0 - Y, Y), -1)

            return torch.as_tensor(Y, dtype=explainer.dtype)

        exp = sp_obj.sp_explanations[i]
        b = np.zeros((1, explainer.X.shape[1]))
        # for classification, SP objects only contain one label; hence, may need to flip the
        # probabilities in the predict function
        # NOTE: this does affect the vector representation; however, the representation
        # should still function as a unique identifier for the model
        if explainer.classifier:
            exp_label = exp.available_labels()[0]
        else:
            exp_label = 1
        for j, v in exp.as_map()[exp_label]:
            b[0, j] = v
        inter = exp.intercept[exp_label]
        if discretize:
            x1 = sp_obj.discretizer.discretize(explainer.X[i : i + 1, :])

        return b, _sp_predict

    sp_models = [explain(i) for i in range(len(sp_obj.sp_explanations))]
    B = torch.vstack(
        [torch.as_tensor(lm[0], dtype=explainer.dtype) for lm in sp_models]
    )
    proxies = [lm[1] for lm in sp_models]
    mapping = generate_minimal_loss_mapping(explainer, proxies)
    # need to find the corresponding rows in the B matrix to get the indices!
    vector_representation = B
    reduced = explainer.clone_reduced(
        proxies, mapping, vector_representation, sp_models
    )
    return reduced


def find_proxies_max_cov_ball_cover(
    explainer: Explainer,
    k: int,
    radius: float,
    phi: float = 0.8,
    time_limit=None,
    epsilon=None,
):
    assert (
        explainer.classifier
    ), "Ball covering aggregation only possible for classifier explainers!"

    n_exp = len(explainer.local_models)
    n_items = explainer.X.shape[0]
    y_tr = torch.max(explainer.y, dim=1)[1]
    match_matrix = torch.zeros((n_exp, n_items))
    for ei in range(n_exp):
        yhat = torch.max(explainer.local_models[ei](explainer.X), dim=1)[1]
        match_matrix[ei] = yhat == y_tr

    dist_matrix = torch.cdist(explainer.X, explainer.X)

    problem = pulp.LpProblem("LiEtAlAggregateExplainer", pulp.LpMaximize)
    # w => explanation is contained in the final set
    w = pulp.LpVariable.dicts(
        "w", range(n_exp), lowBound=0, upBound=1, cat=pulp.LpBinary
    )
    # y => coverage of points by the final set
    y = pulp.LpVariable.dicts(
        "y", range(n_items), lowBound=0, upBound=1, cat=pulp.LpBinary
    )
    # z => coverage of individual explainers
    z = pulp.LpVariable.dicts(
        "z",
        ((i, j) for i in range(n_exp) for j in range(n_items)),
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    # add the coverage maximisation target
    problem += pulp.lpSum(y[j] for j in range(n_items)), "Maximize_Coverage"

    # Constraints
    # z_ij <= w_i or y_j >= z_ij
    for i in range(n_exp):
        for j in range(n_items):
            problem += z[(i, j)] <= w[i], f"z{i}{j}_leq_w{i}"
            problem += y[j] >= z[(i, j)], f"or_y{j}_geq_z{i}{j}"

    # y_j <= sum_i z_ij
    for j in range(n_items):
        problem += (
            y[j] <= pulp.lpSum(z[(i, j)] for i in range(n_exp)),
            f"or_y{j}_leq_sum_z{j}",
        )

    # locality-based constraints
    for i in range(n_exp):
        for j in range(n_items):
            d_ij = dist_matrix[i][j].item()
            problem += d_ij * z[(i, j)] <= radius, f"locality_{i}_{j}"

    # minimum fidelity constraint
    for i in range(n_exp):
        problem += (
            pulp.lpSum(
                (match_matrix[i][j].item() - phi) * z[(i, j)] for j in range(n_items)
            )
            >= 0.0,
            f"min_fidelity_{i}",
        )

    # budget constraint (size of exp set)
    problem += pulp.lpSum(w[i] for i in range(n_exp)) <= k, "budget"

    # Solve the problem
    if time_limit is not None:
        problem.solve(PULP_CBC_CMD(msg=True, timeLimit=time_limit))
    else:
        problem.solve()

    selected_proxies = [i for i in range(n_exp) if pulp.value(w[i]) > 0.5]
    if not selected_proxies:
        raise ValueError(
            "Problem is infeasible! Consider relaxing the constraints "
            + f"(radius={radius}, minimum fidelity={phi})"
        )
    reduced_models = [explainer.local_models[i] for i in selected_proxies]
    mapping = generate_minimal_loss_mapping(explainer, reduced_models)
    vector_representation = explainer.vector_representation[selected_proxies, :]
    reduced = explainer.clone_reduced(
        reduced_models, mapping, vector_representation, selected_proxies
    )
    return reduced


def find_proxies_glocalx(
    explainer: LORERuleExplainer, k: int, strategy="fidelity", epsilon=None
):
    try:
        from glocalx import GLocalX
    except ModuleNotFoundError as e:
        e.add_note("Importing GLocalX failed.")
        e.add_note(
            f"Ensure GLocalX path is set correctly in config.ini (currently {GLOCALX_PATH})."
        )
        raise
    assert "LORERuleExplainer" in str(
        explainer.__class__
    ), f"GLocalX reduction only possible for rule-based local explanations (got {type(explainer)})!"
    X_train, y_train = tonp(explainer.X), tonp(explainer.y)
    tr_set = np.hstack((X_train, y_train[:, 0][:, None]))
    glocalx = GLocalX()
    glocalx.fit(explainer.rules, tr_set, batch_size=2)
    reduced_rules = glocalx.rules(k, tr_set, strategy=strategy)

    dummy_mapping = {i: 1 for i in range(explainer.X.shape[0])}
    dummy_B = torch.zeros((k, explainer.X.shape[1]), dtype=explainer.dtype)
    new_exp = explainer.clone_reduced([], dummy_mapping, dummy_B)
    new_exp.rules = reduced_rules
    new_exp.local_models = [
        lambda x: new_exp._predict_from_rule(new_exp.rules[i], x)
        for i in range(len(new_exp.rules))
    ]
    return new_exp
