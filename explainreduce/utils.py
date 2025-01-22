"""
This module contains miscellaneous utility functions.
"""

import torch
import numpy as np
from typing import Union, Callable
from pathlib import Path
import pandas as pd
from pyarrow import parquet as pq


def predict_from_proxies(
    X_old: torch.Tensor,
    X_new: torch.Tensor,
    L: torch.Tensor,
    proxies: Union[torch.Tensor, list[int]],
    explainer: Callable[[int], Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Generate predictions from proxies. Uses proximity in X-space to pick proxy.

    Args:
        X_old: training data from which the proxies are constructed.
        X_new: data from which predictions are generated.
        L: Loss matrix corresponding to the training data.
        proxies: a list of prototype model indices, corresponding to columns in L.
        explainer: a function of the form yhat = explainer(i)(X_new), where i is a local
        model index.

    Returns:
        a tensor of predictions for X_new.
    """
    if X_new.ndim < 2:
        X_new = X_new[:, None]
    assert (
        X_old.shape[1] == X_new.shape[1]
    ), "New and old data must have the same number of features!"
    X_old, X_new = torch.as_tensor(X_old), torch.as_tensor(X_new)
    D = torch.cdist(X_old, X_new)
    neighbours = torch.argmin(D, axis=0)
    neighbour_loss = L[neighbours, :][:, proxies]
    min_losses = torch.argmin(neighbour_loss, dim=1)
    min_loss_proxies = torch.tensor(proxies)[min_losses]
    proto_ynew = torch.zeros(X_new.shape[0])
    for i in range(X_new.shape[0]):
        if isinstance(proxies, list):
            proto_ynew[i] = torch.as_tensor(explainer(min_loss_proxies[i])(X_new[i]))
        elif isinstance(proxies, torch.Tensor):
            proto_ynew[i] = torch.as_tensor(explainer(min_loss_proxies[i])(X_new[i]))
    return proto_ynew


def tonp(x: Union[torch.Tensor, object]) -> np.ndarray:
    """Convert a `torch.Tensor` to a `numpy.ndarray`.

    If `x` is not a `torch.Tensor` then `np.asarray` is used instead.

    Args:
        x: Input `torch.Tensor`.

    Returns:
        Output `numpy.ndarray`.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return np.asarray(x)


def read_parquet(fname: Union[str, Path]) -> pd.DataFrame:
    """Read and possible repair parquet files."""
    pqfile = pq.ParquetFile(fname)
    df = pqfile.read().to_pandas()
    pqfile.close()
    # broken/interrupted saves
    if "0" in df.columns:
        df = df.drop(["0"], axis=1)
    df = df.dropna(how="all")
    return df
