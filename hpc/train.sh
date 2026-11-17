#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train.out
#BSUB -e hpc/logs/train.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
cd "${LS_SUBCWD:-$(dirname "$0")/..}"   # bsub < runs a copy from ~/.lsbatch; LS_SUBCWD is where it was submitted
source hpc/env.sh   # uv on the PATH, caches off the home quota

# Your training command, e.g.:
# uv run python train.py
uv run python src/protgen/models/mlp.py --context "${CONTEXT:-3}" --dropout "${DROPOUT:-0}" --epochs "${EPOCHS:-10}"   # bsub -env "all, CONTEXT=10, DROPOUT=0.2, EPOCHS=25" < hpc/train.sh
