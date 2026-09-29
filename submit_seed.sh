#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node 0
#SBATCH --cpus-per-task 16
#SBATCH --job-name=eval_arena_seed
#SBATCH --output=/checkpoint/amaia/codegen/sida/eval_arena/logs/eval_arena.%j.out
#SBATCH --error=/checkpoint/amaia/codegen/sida/eval_arena/logs/eval_arena.%j.err
#SBATCH --qos=normal

set -e

NAME=${1:-"main"}
OUTPATH=/checkpoint/amaia/codegen/sida/eval_arena
echo "Argument 1: $NAME"

# raw-data/seed_models/*.csv -> data/seed_models/*.jsonl
python -u scripts/process_seed_models.py

python -u run_arena.py data="data/seed_models/*.jsonl" \
    out_dir=${OUTPATH}/${NAME}/seed \
    max_diff=0.1 recompute=True include_var_components=False
