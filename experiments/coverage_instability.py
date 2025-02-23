###############################################################################
#
# This experiment evaluate
#   1. Coverage of the proxies as a function of the proxy number k
#   2. Stability of the proxies as a function of the proxy number k
# with certain combinations of datasets and explanation methods.
#
# This experiment is designed to be run in parallel (e.g. on a computer cluster).
#
# Run this script to perform the experiments,
# where job_id is an integer indicating the id of the parallel task.
# For example (the number of proxies is defined within the main function):
#   `python experiment/coverage_instability.py $job_id`
#
# Run this script again without additional arguments to produce plots of coverage and stability from the results:
#   `python experiment/coverage_instability.py`
#
###############################################################################

from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
from glob import glob
import sys
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import evaluate_k_sensitivity, rename_columns, paper_theme

OUTPUT_DIR = RESULTS_DIR / "k_sensitivity"


def plot_results(df: pd.DataFrame):
    """Produce the small and large (appendix) plots for k-sensitivity."""
    for f in OUTPUT_DIR.glob("**/*.parquet"):
        try:
            new_df = pq.ParquetFile(f).read().to_pandas()
            if "0" in new_df.columns:
                new_df = new_df.drop(["0"], axis=1)
            df = pd.concat((new_df, df), ignore_index=True)
        except Exception as e:
            print(f"Failed to read {f}")
            print(e)
    df = rename_columns(df)
    exp_methods = ["LIME", "SHAP", "SLISEMAP", "SmoothGrad"]
    datasets = ["Gas Turbine", "Jets", "QM9"]
    p_df = df.loc[df["exp_method"].isin(exp_methods)]
    p_df = p_df.loc[p_df["data"].isin(datasets)]
    p_df = p_df.sort_values(["data", "exp_method"])
    p_df.loc[:, "proxy fidelity/stability"] = (
        p_df["proxy_fidelity"] / p_df["proxy_stability"]
    )
    p_df = p_df.rename(
        columns={"proxy_method_mod": "Reduction method", "proxy_fidelity": "Fidelity"}
    )
    mean_df = (
        p_df.groupby(["data", "exp_method"])
        .mean(numeric_only=True)[
            ["full_fidelity", "full_stability", "full_coverage", "bb_loss_test"]
        ]
        .reset_index()
    )
    g = sns.relplot(
        p_df,
        x="k",
        y="Fidelity",
        col="exp_method",
        row="data",
        hue="Reduction method",
        style="Reduction method",
        kind="line",
        facet_kws={"sharey": False},
        **paper_theme(cols=3, rows=2),
    )
    dname = "Gas Turbine"
    spdf = p_df.loc[p_df["data"].isin([dname])]
    spdf = spdf.loc[spdf["exp_method"].isin(["LIME", "SHAP", "SLISEMAP"])]
    spdf = spdf[
        [
            "job",
            "exp_method",
            "data",
            "Reduction method",
            "proxy_coverage",
            "proxy_stability",
            "k",
        ]
    ].melt(["job", "exp_method", "data", "Reduction method", "k"])
    g = sns.relplot(
        spdf,
        x="k",
        y="value",
        col="exp_method",
        row="variable",
        hue="Reduction method",
        style="Reduction method",
        kind="line",
        facet_kws={"sharey": False},
    )
    for i, varname in enumerate(spdf["variable"].unique()):
        # set ylims based on dataset
        data_y_min, data_y_max = np.inf, -np.inf
        for j, exp_name in enumerate(spdf["exp_method"].unique()):
            y_min, y_max = g.axes[i, j].get_ylim()
            variable_name = "coverage" if "coverage" in varname else "stability"
            full_value = mean_df.loc[
                (mean_df["data"] == dname) & (mean_df["exp_method"] == exp_name)
            ][f"full_{variable_name}"].values[0]
            if y_min < data_y_min:
                data_y_min = y_min
            if y_max > data_y_max:
                data_y_max = y_max
            y_max = max(y_max, full_value)
        for j, exp_name in enumerate(spdf["exp_method"].unique()):
            variable_name = "coverage" if "coverage" in varname else "stability"
            full_value = mean_df.loc[
                (mean_df["data"] == dname) & (mean_df["exp_method"] == exp_name)
            ][f"full_{variable_name}"].values[0]
            if j == 0:
                if variable_name == "stability":
                    g.axes[i, j].set_ylabel("Instability")
                else:
                    g.axes[i, j].set_ylabel("Coverage")
            g.axes[i, j].axhline(full_value, color="k", linestyle="--")
            g.axes[i, j].set_ylim((data_y_min, data_y_max))
            g.axes[i, j].set_title(f"XAI: {exp_name}")
    plt.savefig(MANUSCRIPT_DIR / "coverage_stability_k_small.pdf", dpi=600)


def plot_results_full(df: pd.DataFrame):
    """Generate separate plots for coverage and instability across all datasets."""
    # Load additional data
    for f in OUTPUT_DIR.glob("**/*.parquet"):
        try:
            new_df = pq.ParquetFile(f).read().to_pandas()
            if "0" in new_df.columns:
                new_df = new_df.drop(["0"], axis=1)
            df = pd.concat((new_df, df), ignore_index=True)
        except Exception as e:
            print(f"Failed to read {f}")
            print(e)

    df = rename_columns(df)

    exp_methods = ["LIME", "SHAP", "SLISEMAP", "SmoothGrad"]
    df = df.loc[df["exp_method"].isin(exp_methods)]
    df = df.sort_values(["data", "exp_method"])

    df = df.rename(columns={"proxy_method_mod": "Reduction method"})

    # Melt dataframe for plotting
    plot_df = df.melt(
        id_vars=["job", "exp_method", "data", "Reduction method", "k"],
        value_vars=["proxy_coverage", "proxy_stability"],
        var_name="Metric",
        value_name="Value",
    )

    # Generate separate plots for coverage and instability
    for metric, ylabel, filename in zip(
        ["proxy_coverage", "proxy_stability"],
        ["Coverage", "Instability"],
        ["coverage_full.pdf", "stability_full.pdf"],
    ):

        sub_df = plot_df[plot_df["Metric"] == metric]
        g = sns.relplot(
            data=sub_df,
            x="k",
            y="Value",
            col="exp_method",
            row="data",
            hue="Reduction method",
            style="Reduction method",
            kind="line",
            facet_kws={"sharey": False},
        )

        for i, dataset in enumerate(sub_df["data"].unique()):
            for j, exp_method in enumerate(sub_df["exp_method"].unique()):
                g.axes[i, j].set_ylabel(ylabel)
                g.axes[i, j].set_title(f"Dataset: {dataset}, XAI: {exp_method}")

        # Reduce the size to fit into the paper
        fig = g.figure
        original_size = fig.get_size_inches()
        fig.set_size_inches(original_size[0], original_size[1] * 5 / 8)

        plt.savefig(MANUSCRIPT_DIR / filename, dpi=600)
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        plot_results(df)
        plot_results_full(df)
    else:
        ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        time = timer()
        evaluate_k_sensitivity(int(sys.argv[1]), ks=ks)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
