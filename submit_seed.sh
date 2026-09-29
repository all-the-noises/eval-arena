#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node 0
#SBATCH --cpus-per-task 16
#SBATCH --mem=384G
#SBATCH --job-name=eval_arena_seed
#SBATCH --output=/checkpoint/amaia/codegen/sida/eval_arena/logs/eval_arena.%j.out
#SBATCH --error=/checkpoint/amaia/codegen/sida/eval_arena/logs/eval_arena.%j.err
#SBATCH --qos=normal

set -e

NAME=${1:-"main"}
OUTPATH=/checkpoint/amaia/codegen/sida/eval_arena
echo "Argument 1: $NAME"

# raw-data/seed_models/*.csv -> data/seed_*/*.jsonl
# all 210 models (10 seeds x 21 ckpts) is too big for the pairwise merge in arena, so subsample ckpts
python -u scripts/process_seed_models.py raw-data/seed_models data/seed_final 50000
python -u scripts/process_seed_models.py raw-data/seed_models data/seed_4ckpt 12500,25000,37500,50000

python -u run_arena.py data="data/seed_final/*.jsonl" \
    out_dir=${OUTPATH}/${NAME}/seed_final \
    max_diff=0.1 recompute=True include_var_components=False

python -u run_arena.py data="data/seed_4ckpt/*.jsonl" \
    out_dir=${OUTPATH}/${NAME}/seed_4ckpt \
    max_diff=0.1 recompute=True include_var_components=False
