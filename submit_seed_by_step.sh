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
OUTPATH=/checkpoint/amaia/codegen/sida/eval_arena
echo "Argument 1: $NAME"

# each (benchmark, ckpt_step) is its own benchmark_id with the 10 seeds as models,
# so only models that differ in seed but not step are compared
python -u scripts/process_seed_models.py --out_dir data/seed_by_step --by_step

python -u run_arena.py data="data/seed_by_step/*.jsonl" \
    out_dir=${OUTPATH}/${NAME}/seed_by_step \
    max_diff=0.1 recompute=True include_var_components=False
