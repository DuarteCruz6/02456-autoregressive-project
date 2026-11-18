#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
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

# Everything Phase 1 needs: the architectures, a hyperparameter sweep, the MLP context ablation.
# Subsets:  bsub -env "all, MODELS=gru, BEST=gru" < hpc/train.sh

# 1. the architectures, all at the same settings, and 256 samples from each
MODELS="${MODELS:-mlp rnn gru lstm transformer}"
for M in $MODELS; do
    echo "== training $M"
    uv run python train.py --model "$M"
    uv run python generate.py --run "$M" --num 256
done

# 2. sweep one model, one setting at a time. Not sampled: these are for the validation table.
BEST="${BEST:-gru}"
echo "== sweep on $BEST"
uv run python train.py --model "$BEST" --lr 3e-4        --run-name "${BEST}_lr3e-4"
uv run python train.py --model "$BEST" --lr 3e-3        --run-name "${BEST}_lr3e-3"
uv run python train.py --model "$BEST" --hidden-dim 128 --run-name "${BEST}_h128"
uv run python train.py --model "$BEST" --hidden-dim 512 --run-name "${BEST}_h512"
uv run python train.py --model "$BEST" --dropout 0.3    --run-name "${BEST}_do0.3"
uv run python train.py --model "$BEST" --num-layers 3   --run-name "${BEST}_L3"

# 3. what a wider fixed window buys the MLP
for C in 5 10 20; do
    echo "== mlp context $C"
    uv run python train.py --model mlp --context "$C" --run-name "mlp_ctx$C"
done

echo "Done. Results in runs/<name>/, samples in results/sequences/"
