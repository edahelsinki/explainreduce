from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
from glob import glob
import sys
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import evaluate_subsample_sensitivity

OUTPUT_DIR = RESULTS_DIR / "n_sensitivity"


def plot_results(df: pd.DataFrame):
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


def plot_results_full(df: pd.DataFrame):
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        plot_results(df)
        plot_results_full(df)
    else:
        ns = [25, 50, 100, 200, 300, 400, 500, 750, 1000]
        time = timer()
        evaluate_subsample_sensitivity(int(sys.argv[1]), ns=ns)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
