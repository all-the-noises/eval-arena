#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node 0
#SBATCH --cpus-per-task 16
#SBATCH --mem=128G
#SBATCH --job-name=eval_arena_seed_by_step
#SBATCH --output=/checkpoint/amaia/codegen/sida/eval_arena/logs/eval_arena.%j.out
#SBATCH --error=/checkpoint/amaia/codegen/sida/eval_arena/logs/eval_arena.%j.err
#SBATCH --qos=normal

set -e

NAME=${1:-"main"}
STEPS=${2:-""} # optional comma-separated ckpt_steps, e.g. 2500,27500,50000 (first, middle, last); default all
OUTPATH=/checkpoint/amaia/codegen/sida/eval_arena
echo "Argument 1: $NAME, steps: ${STEPS:-all}"

SET=seed_by_step${STEPS:+_${STEPS//,/_}}

# each (benchmark, ckpt_step) is its own benchmark_id with the 10 seeds as models,
# so only models that differ in seed but not step are compared
python -u scripts/process_seed_models.py --out_dir data/${SET} --by_step ${STEPS:+--steps $STEPS}

python -u run_arena.py data="data/${SET}/*.jsonl" \
    out_dir=${OUTPATH}/${NAME}/${SET} \
    max_diff=0.1 recompute=True include_var_components=False
