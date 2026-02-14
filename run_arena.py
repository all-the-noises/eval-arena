import logging
import os
from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd

import arena
from arena import ReportArgs
from reports import (
    load_data,
    write_data,
    write_example_report,
    write_model_report,
    write_summary_table,
    write_directory_index,
)
from utils import load_jsonl_files, check_data, fill_count, check_and_fill_correct

logger = logging.getLogger(__name__)


def setup_output(args: ReportArgs):
    # Copy custom.css to the output directory
    os.makedirs(Path(args.out_dir) / "static" / "css" , exist_ok=True)
    with open(Path("templates/custom.css"), "rb") as src_file, open(Path(args.out_dir) / "static" / "css" / "custom.css", "wb") as dst_file:
        dst_file.write(src_file.read())
    
    with open(Path("templates/bulma.min.css"), "rb") as src_file, open(Path(args.out_dir) / "static" / "css" / "bulma.min.css", "wb") as dst_file:
        dst_file.write(src_file.read())


def render_reports(args: ReportArgs, results: dict[str, arena.ArenaResult] | None = None):
    out_dir = Path(args.out_dir)

    if not results:
        results = load_data(out_dir)
    if not results:
        logger.error(f"No benchmark data found in {out_dir}. Run with recompute=True first.")
        return

    for bid, arena_res in results.items():
        benchmark_out_dir = out_dir / bid
        logger.info(f"Rendering reports for {bid}...")
        write_model_report(bid, arena_res, benchmark_out_dir)
        write_example_report(bid, arena_res, benchmark_out_dir)
        write_directory_index(bid, benchmark_out_dir)

    df_summary = pd.DataFrame([res.summary_stats for res in results.values()])
    write_summary_table(df_summary, out_dir / "index.html", include_var_components=args.include_var_components)


def run_arena(args: ReportArgs):
    results = {}
    if args.recompute:
        records = load_jsonl_files(args.data)
        eval_results = pd.DataFrame(records)
        eval_results = fill_count(eval_results)
        eval_results = check_and_fill_correct(eval_results)
        check_data(eval_results)
        logger.info(f"Loaded {len(eval_results)} evaluation results")

        benchmarks = set(eval_results["benchmark_id"])
        logger.info(f"Included benchmarks: {benchmarks}")
        logger.info(f"Included models: {set(eval_results['model'])}")

        for bid in benchmarks:
            logger.info(f"Processing {bid}...")
            result_bid = eval_results[eval_results["benchmark_id"] == bid]
            arena_res: arena.ArenaResult = arena.summarize_benchmark(result_bid, args)

            write_data(bid, arena_res, args.out_dir)
            results[bid] = arena_res

        df_summary = pd.DataFrame([res.summary_stats for res in results.values()])
        df_summary.to_csv(Path(args.out_dir) / "summary.csv")
        logger.info(f"Wrote summary.csv with {len(df_summary)} records")

    render_reports(args, results or None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(ReportArgs())
    args = OmegaConf.merge(default_cfg, cli_args)
    args = OmegaConf.to_object(args)
    logger.info(f"Running with args: {args}")
    setup_output(args)
    run_arena(args)
