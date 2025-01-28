from project_paths import RESULTS_DIR, MANUSCRIPT_DIR
from glob import glob
import sys
import pandas as pd
from timeit import default_timer as timer
from utils.greedy_comparison import (
    evaluate,
    preprocess_results,
    concanate_results,
    produce_latex,
)

OUTPUT_DIR = RESULTS_DIR / "greedy_comparison"


def table_results(df: pd.DataFrame):
    """Produce a latex table of the results and save to the repo."""
    df = preprocess_results(df)
    mean_dfs, std_dfs = concanate_results(df)
    produce_latex(mean_dfs, std_dfs)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        table_results(df)
    else:
        ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        time = timer()
        evaluate(int(sys.argv[1]), ks=ks)
        print(f"Total runtime {timer() - time} seconds.", flush=True)
