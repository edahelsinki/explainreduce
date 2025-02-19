###############################################################################
#
# This script demonstrates ExplainReduce on the Jets dataset.
# We train a Slisemap explainer on synthetic data and then reduce the proxies.
#
# To run the script, simply call
#   `python3 experiments/jets_example.py`
# The resulting plots will be saved in the ms/ directory under the project
# root.
#
###############################################################################
import torch
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
import sys
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_DIR)

from project_paths import MANUSCRIPT_DIR
from utils import data
from explainreduce import localmodels as lm
from explainreduce import proxies as px
from utils.hyperparameters import get_data, get_bb, get_params

# set seeds for reproducibility
np.random.seed(42)
torch.random.manual_seed(42)


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
        X_train, y_train, train_size=500, random_state=seed
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


_, _, names = data.get_jets(names=True)
ds_dict = prepare_data(get_data()["Jets"], 1619)
X_bb, y_bb = ds_dict["bb_train"]
X_exp, y_exp = ds_dict["exp_train"]
X_test, y_test = ds_dict["test"]
m, pred_fn = get_bb(
    ds_dict["black_box"], ds_dict["classifier"], X_bb, y_bb, dname="Jets"
)
yhat = torch.as_tensor(pred_fn(X_bb), dtype=y_bb.dtype)
yhat_test = torch.as_tensor(pred_fn(X_test), dtype=y_bb.dtype)
yhat_exp = torch.as_tensor(pred_fn(X_exp), dtype=y_bb.dtype)
loss_fn = (
    torch.nn.MSELoss(reduction="none")
    if not ds_dict["classifier"]
    else lm.logistic_regression_loss
)

print("Training SLISEMAP.")
time = timer()
exp = lm.SLISEMAPExplainer(
    X_exp,
    y_exp,
    classifier=ds_dict["classifier"],
    black_box_model=m,
    **get_params("SLISEMAP", data="Jets", black_box=ds_dict["black_box"]),
)
exp.fit()
explainer_yhat_test = exp.predict(X_test)
full_fidelity = (
    loss_fn(
        torch.as_tensor(yhat_test),
        explainer_yhat_test,
    )
    .mean()
    .item()
)
print(
    f"Done training, took {timer() - time:.1f} s. SLISEMAP fidelity: {full_fidelity:.3f}."
)

print("Finding the proxy set using the constrained greedy min loss algorithm.")
time = timer()
loss_train = loss_fn(yhat, y_bb)
epsilon = torch.quantile(loss_train, 0.2)
proxies = px.find_proxies_greedy_min_loss_k_min_cov(
    exp, k=4, epsilon=epsilon, raise_on_infeasible=False
)
prx_yhat_test = proxies.predict(X_test)
proxy_fidelity = (
    loss_fn(
        torch.as_tensor(yhat_test),
        prx_yhat_test,
    )
    .mean()
    .item()
)
print(
    f"Done reducing, took {timer() - time:.1f} s. Proxy fidelity: {proxy_fidelity:.3f}."
)

df = pd.DataFrame(X_exp, columns=names)
df["gluon_prob"] = yhat_exp[:, 0]
df["true_label"] = y_exp[:, 0]
df["Proxy ID"] = proxies.mapping.values()
params = pd.DataFrame(
    proxies.vector_representation, columns=list(names) + ["intercept"]
)
params["Proxy ID"] = params.index
params = pd.melt(params, id_vars=["Proxy ID"])
fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
sns.barplot(
    params,
    y="variable",
    x="value",
    hue="Proxy ID",
    orient="y",
    palette="deep",
    ax=ax[1],
)
sns.swarmplot(df, x="gluon_prob", hue="Proxy ID", palette="deep", ax=ax[0])
ax[0].set_xlabel("$p(C = \\text{Gluon} | x)$")
ax[1].set_ylabel("Feature name")
ax[1].set_xlabel("Proxy coefficient")
fig.tight_layout()
fig.savefig(MANUSCRIPT_DIR / "jets_example.pdf", dpi=300)
fig.show()
