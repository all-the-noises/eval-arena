import logging
import pickle
from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd

import arena
from arena import ReportArgs
from render_reports import setup_output, render_reports
from signal_noise import signal_to_noise
from utils import load_jsonl_files, check_data, fill_count, check_and_fill_correct

logger = logging.getLogger(__name__)


def run_arena(args: ReportArgs):
    records = load_jsonl_files(args.data)
    eval_results = pd.DataFrame(records)
    eval_results = fill_count(eval_results)
    eval_results = check_and_fill_correct(eval_results)
    check_data(eval_results)
    logger.info(f"Loaded {len(eval_results)} evaluation results")

    benchmarks = set(eval_results["benchmark_id"])
    logger.info(f"Included benchmarks: {benchmarks}")
    logger.info(f"Included models: {set(eval_results['model'])}")
    tmp_dir = Path(args.out_dir) / "tmp"

    if args.recompute:
        for bid in benchmarks:
            logger.info(f"Processing {bid}...")
            result_bid = eval_results[eval_results["benchmark_id"] == bid]
            arena_res: arena.ArenaResult = arena.summarize_benchmark(result_bid, args)

            sig_to_noise = signal_to_noise(bid, arena_res.summary)
            summary_stats = arena_res.summary_stats
            summary_stats["sig_noise"] = sig_to_noise["signal to noise"].median() if sig_to_noise is not None else float("nan")

            logger.info(f"Summary stats for {bid}:\n{pd.DataFrame([summary_stats])}")
            pd.DataFrame([summary_stats]).to_json(tmp_dir / f"summary-{bid}.jsonl", orient="records", lines=True)
            with open(tmp_dir / f"{bid}.pkl", "wb") as f:
                pickle.dump(arena_res, f)

    render_reports(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(ReportArgs())
    args = OmegaConf.merge(default_cfg, cli_args)
    args = OmegaConf.to_object(args)
    logger.info(f"Running with args: {args}")
    setup_output(args)
    run_arena(args)
