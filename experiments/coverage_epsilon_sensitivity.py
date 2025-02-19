from project_paths import RESULTS_DIR
from glob import glob
import sys
import pandas as pd
from timeit import default_timer as timer
from utils.coverage_epsilon_sensitivity import (
    evaluate,
    preprocess_results,
    plot_result_small,
    plot_result_full,
)

OUTPUT_DIR = RESULTS_DIR / "coverage_epsilon_sensitivity"


def plot_results(df: pd.DataFrame):
    """Produce a small and large plot of the results and save to the repo."""
    df = preprocess_results(df)
    plot_result_small(df)
    plot_result_full(df)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        plot_results(df)
    else:
        coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        time = timer()
        evaluate(int(sys.argv[1]), coverages=coverages, ps=ps)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
