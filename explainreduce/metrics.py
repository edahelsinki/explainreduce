import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from localmodels import Explainer
from explainreduce.utils import tonp


# Get metrics
def get_fidelity_loss(prototypes: dict, L: torch.Tensor, Geps: torch.Tensor):
    fidelity_loss = 0
    L_prototypes = L[list(prototypes.keys())]
    Geps_prototypes = Geps[list(prototypes.keys())]
    for sample_idx in range(Geps_prototypes.shape[1]):
        min_fidelity_loss = L_prototypes[:, sample_idx].min().item()
        fidelity_loss += min_fidelity_loss
    fidelity_loss /= Geps_prototypes.shape[1]
    return fidelity_loss


def get_coverage(prototypes, Geps):
    coverage = 0
    Geps_prototypes = Geps[list(prototypes.keys())]
    for sample_idx in range(Geps_prototypes.shape[1]):
        if Geps_prototypes[:, sample_idx].sum() > 0:
            coverage += 1
    coverage /= Geps_prototypes.shape[1]
    return coverage


def calculate_coverage(G_eps: torch.Tensor, prototypes: list[int] = None) -> float:
    """Calculate the coverage (ratio of columns of the applicability matrix covered by
    prototypes (rows))."""
    if prototypes is not None:
        G = G_eps[prototypes, :]
    else:
        G = G_eps
    covered_cols = torch.any(G, dim=0)
    coverage = torch.sum(covered_cols) / G_eps.shape[1]
    return coverage.item()


def calculate_global_epsilon(explainer: Explainer, quantile_threshold=0.2):
    """Calculate global epsilon for coverage calculation by comparing to the
    perfomance of a linear model.

    Parameters:
    -----------
    explainer: Explainer object
    quantile_threshold: float
        Quantile for loss values of the global model.

    Returns:
    --------
    Threshold epsilon value.

    """
    cls = explainer.classifier
    linear = LogisticRegression() if cls else LinearRegression()
    explainer_y = tonp(explainer.y)
    explainer_x = tonp(explainer.X)
    if cls and explainer_y.ndim > 1:
        explainer_y = explainer_y[:, 0]
        # encode to class labels
        explainer_y = explainer_y > 0.5
    linear.fit(explainer_x, explainer_y)
    yhat = linear.predict_proba(explainer_x) if cls else linear.predict(explainer_x)
    losses = explainer.loss_fn(explainer.y, torch.as_tensor(yhat))
    epsilon = torch.quantile(losses, quantile_threshold)
    return epsilon.item()
