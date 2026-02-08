"""
Re-render HTML reports from pre-computed results (pickle files in out_dir/tmp/).

Run run_arena.py first to compute stats, then use this to iterate on presentation.

Standalone usage: 
    python render_reports.py --out_dir gh-pages/ [--no-write_summary] [--no-include_var_components]
"""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from arena import ReportArgs
from reports import (
    write_example_report,
    write_model_report,
    write_data_tables,
    write_summary_table,
    write_sections_index,
    write_directory_index,
)
from utils import load_jsonl_files

logger = logging.getLogger(__name__)


def setup_output(args: ReportArgs):
    os.makedirs(Path(args.out_dir) / "static" / "css", exist_ok=True)
    with open(Path("templates/custom.css"), "rb") as src_file, open(Path(args.out_dir) / "static" / "css" / "custom.css", "wb") as dst_file:
        dst_file.write(src_file.read())
    with open(Path("templates/bulma.min.css"), "rb") as src_file, open(Path(args.out_dir) / "static" / "css" / "bulma.min.css", "wb") as dst_file:
        dst_file.write(src_file.read())
    os.makedirs(Path(args.out_dir) / "tmp", exist_ok=True)


def render_reports(args: ReportArgs):
    tmp_dir = Path(args.out_dir) / "tmp"
    pkl_files = sorted(tmp_dir.glob("*.pkl"))
    if not pkl_files:
        logger.error(f"No .pkl files found in {tmp_dir}. Run run_arena.py first.")
        return

    for pkl_path in pkl_files:
        bid = pkl_path.stem
        logger.info(f"Rendering reports for {bid}...")
        with open(pkl_path, "rb") as f:
            arena_res = pickle.load(f)

        benchmark_out_dir = Path(args.out_dir) / bid
        os.makedirs(benchmark_out_dir, exist_ok=True)

        write_model_report(bid, arena_res, benchmark_out_dir)
        write_example_report(bid, arena_res, benchmark_out_dir)
        write_data_tables(arena_res, benchmark_out_dir)
        write_directory_index(bid, benchmark_out_dir)

    if args.write_summary:
        records = load_jsonl_files(f"{tmp_dir}/summary-*.jsonl")
        logger.info(f"Loaded {len(records)} summary records")
        df_summary = pd.DataFrame(records)
        df_summary.to_csv(Path(args.out_dir) / "summary.csv")
        write_summary_table(df_summary, Path(args.out_dir) / "index.html", include_var_components=args.include_var_components)
        
    write_sections_index(Path(args.out_dir))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Re-render HTML reports from pre-computed results.")
    parser.add_argument("--out_dir", type=str, default="gh-pages/", help="Output directory")
    parser.add_argument("--write_summary", action=argparse.BooleanOptionalAction, default=True, help="Generate summary table")
    parser.add_argument("--include_var_components", action=argparse.BooleanOptionalAction, default=True, help="Include variance components in summary")
    cli_args = parser.parse_args()

    args = ReportArgs(out_dir=cli_args.out_dir, write_summary=cli_args.write_summary, include_var_components=cli_args.include_var_components)
    logger.info(f"Rendering with args: {args}")
    setup_output(args)
    render_reports(args)
