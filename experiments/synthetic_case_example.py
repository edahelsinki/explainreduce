###############################################################################
#
# This script serves as an example of the ExplainReduce procedure. Here, we
# train a SmoothGrad explainer on synthetic data and then reduce the proxies.
#
# To run the script, simply call
#   `python3 experiments/synthetic_case_example.py`
# The resulting plots will be saved in the ms/ directory under the project
# root.
#
###############################################################################
import numpy as np
import torch
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import sys
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_DIR)
import explainreduce.localmodels as lm
import explainreduce.proxies as prx
from experiments.utils.hyperparameters import get_bb
from project_paths import MANUSCRIPT_DIR
from sklearn.decomposition import PCA
from utils.data import get_rsynth

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

j, X, y, B = get_rsynth(N=2000, M=11, k=4, s=2.5, seed=seed)
# X, y, _, bb = get_data()["Gas Turbine"]()
idx = torch.randperm(X.shape[0])
n_samples = 500
X_train, y_train = X[idx[:n_samples], :], y[idx[:n_samples]]
j_train = j[idx[:n_samples]]
X_test = X[idx[n_samples : 3 * n_samples], :]
y_test = y[idx[n_samples : 3 * n_samples]]
m, pred_fn = get_bb("Neural Network", False, X_train, y_train)
yhat_train = pred_fn(X_train)
j_test = j[idx[n_samples : 3 * n_samples]]

exp = lm.SmoothGradExplainer(X_train, yhat_train, black_box_model=m)
exp.fit()

epsilon = torch.quantile(exp.get_L(), q=0.10).item()
proxies = prx.find_proxies_greedy(exp, k=4, epsilon=epsilon)

# PCA plot
fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
p = PCA(n_components=2)
X_trans = p.fit_transform(X_test)
for cluster_id in np.unique(j_test):
    mask = np.where(j_test == cluster_id)[0]
    ax[0].scatter(
        *X_trans[mask, :].T,
        alpha=0.5,
        label=f"Cluster {cluster_id+1}",
    )
ax[0].set_xlabel("PCA 1")
ax[0].set_ylabel("PCA 2")
ax[0].legend()


mapping = proxies._map_to_training_data(X_test)
prx_j = np.array(list(proxies.mapping.values()))[mapping]
# map cluster labels for easier reading
j_map = {}
for cid in np.unique(j_test):
    max_count = np.unique(j_test[np.where(prx_j == cid)[0]], return_counts=True)
    j_map[cid] = max_count[0][np.argmax(max_count[1])]
prx_j = np.vectorize(j_map.get)(prx_j)
for cluster_id in np.unique(prx_j):
    mask = np.where(prx_j == cluster_id)[0]
    ax[1].scatter(
        *X_trans[mask, :].T, marker="x", alpha=0.5, label=f"Proxy index {cluster_id+1}"
    )
ax[1].legend()
ax[1].set_xlabel("PCA 1")
ax[1].set_ylabel("PCA 2")
fig.savefig(MANUSCRIPT_DIR / "case_PCA.pdf", dpi=300)
print(f"Scatter plot saved at {(MANUSCRIPT_DIR / 'case_PCA.pdf')}")
plt.show()

# radar plot of models
map = np.argmin(cdist(proxies.vector_representation, B), axis=0)
B_p = proxies.vector_representation[map, :]
data = [
    (f"Cluster {i+1}", [np.flip(B[i, :]), np.flip(B_p[i, :].numpy())])
    for i in range(B.shape[0])
]
N = B.shape[1]
theta = radar_factory(N, frame="polygon")

# data = example_data()
spoke_labels = list(
    reversed([f"$X_{{{x+1}}}$" for x in range(B.shape[1] - 1)] + ["Intercept"])
)

fig, axs = plt.subplots(
    figsize=(15, 5), nrows=1, ncols=4, subplot_kw=dict(projection="radar"), sharey=True
)
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

colors = ["b", "r"]
# Plot the four cases from the example data on separate Axes
for ax, (title, case_data) in zip(axs.flat, data):
    ax.set_rgrids([0.5, 1.5, 2.5])
    # ax.set_ylim(0, 2.6)
    ax.set_title(
        title,
        weight="bold",
        size="medium",
        position=(0.5, 1.1),
        horizontalalignment="center",
        verticalalignment="center",
    )
    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")
    ax.set_varlabels(spoke_labels)

# add legend relative to top-left plot
labels = ("Ground truth", "Proxy model")
legend = axs[1].legend(labels, loc=(0.85, 1.1), labelspacing=0.1, fontsize="medium")

fig.savefig(MANUSCRIPT_DIR / "case_radar.pdf", dpi=300)
print(f"Radar plot saved at {(MANUSCRIPT_DIR / 'case_radar.pdf')}")
plt.show()
