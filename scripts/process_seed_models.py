"""
Convert the seed_models evals (Llama 2 architecture trained with different seeds,
evaluated at several checkpoints) into the eval-arena format.

Ported from cell 8 of notebooks/process_internal.ipynb (branch internal_process_data),
which follows the same conventions as the hf_models processing.

Usage:
    python scripts/process_seed_models.py [--raw_dir R] [--out_dir O] [--steps S] [--by_step]
    # defaults: raw-data/seed_models/*.csv -> data/seed_models/<benchmark>.jsonl
    # --steps: comma-separated ckpt_steps to keep, e.g. 50000 or 12500,25000,37500,50000
    #   (all 210 models = 10 seeds x 21 ckpts is too big for arena's pairwise merge)
    # --by_step: make each (benchmark, ckpt_step) its own benchmark_id, e.g. gsm8k_step50000,
    #   with models named by seed, so only models that differ in seed are compared
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def process(path: str, steps: list[int] | None, by_step: bool) -> tuple[str, pd.DataFrame]:
    dfi = pd.read_csv(path)
    if steps is not None:
        dfi = dfi[dfi["ckpt_step"].isin(steps)]
    bid = bprefix = set(dfi["benchmark"]).pop()
    if bid == "human_eval":  # consistency with the other humaneval data
        bid = "humaneval"
        dfi["benchmark"] = "humaneval"
        bprefix = "HumanEval"

    if by_step:
        dfi["benchmark"] = dfi["ckpt_step"].apply(lambda s: f"{bid}_step{s:05d}")
        dfi["model_name"] = dfi["seed"].apply(lambda s: f"seed{s}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="raw-data/seed_models")
    parser.add_argument("--out_dir", default="data/seed_models")
    parser.add_argument("--steps", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--by_step", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in sorted(glob.glob(f"{args.raw_dir}/*.csv")):
        bid, df = process(path, args.steps, args.by_step)
        out = f"{args.out_dir}/{bid}.jsonl"
        with open(out, "w") as f:
            f.write(df.to_json(orient="records", lines=True))
        print(f"{path} -> {out}: {len(df)} rows, {df['benchmark_id'].nunique()} benchmark_ids, "
              f"{df['model'].nunique()} models, {df['example_id'].nunique()} examples, "
              f"mean pass1 {df['pass1'].mean():.3f}")
