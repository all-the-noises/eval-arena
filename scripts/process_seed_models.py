"""
Convert the seed_models evals (Llama 2 architecture trained with different seeds,
evaluated at several checkpoints) into the eval-arena format.

Ported from cell 8 of notebooks/process_internal.ipynb (branch internal_process_data),
which follows the same conventions as the hf_models processing.

Usage:
    python scripts/process_seed_models.py [raw_dir] [out_dir]
    # defaults: raw-data/seed_models/*.csv -> data/seed_models/<benchmark>.jsonl
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "raw-data/seed_models"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "data/seed_models"


def process(path: str) -> tuple[str, pd.DataFrame]:
    dfi = pd.read_csv(path)
    bid = bprefix = set(dfi["benchmark"]).pop()
    if bid == "human_eval":  # consistency with the other humaneval data
        bid = "humaneval"
        dfi["benchmark"] = "humaneval"
        bprefix = "HumanEval"

    maps = {"model_name": "model", "benchmark": "benchmark_id", "example_idx": "example_id"}
    if "pass@1" in dfi.columns:
        maps["pass@1"] = "pass1"
    elif "acc_char" in dfi.columns:
        maps["acc_char"] = "pass1"
        if not np.allclose(dfi["acc_raw"], dfi["acc_char"], rtol=1e-2):
            print(f"{bid}: acc_raw and acc_char are inconsistent, using acc_char")
    elif "em" in dfi.columns:
        maps["em"] = "pass1"

    dfi = dfi[maps.keys()].rename(columns=maps)
    dfi["example_id"] = dfi["example_id"].apply(lambda x: f"{bprefix}/{x}")
    return bid, dfi[["benchmark_id", "example_id", "model", "pass1"]]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in sorted(glob.glob(f"{RAW_DIR}/*.csv")):
        bid, df = process(path)
        out = f"{OUT_DIR}/{bid}.jsonl"
        with open(out, "w") as f:
            f.write(df.to_json(orient="records", lines=True))
        print(f"{path} -> {out}: {len(df)} rows, {df['model'].nunique()} models, "
              f"{df['example_id'].nunique()} examples, mean pass1 {df['pass1'].mean():.3f}")
