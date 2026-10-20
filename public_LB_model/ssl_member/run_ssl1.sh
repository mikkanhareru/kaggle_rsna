#!/bin/bash
# S2 bounded MAE trial. Recipe FROZEN in competition_details/preregistered/2026-09-19_ssl_mae_trial.md
set -euo pipefail
cd /import/fox/will/graduation_thesis/RSNA_kaggle
PY=~/anaconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -gt 5000 ]; then echo "!! GPU busy (${USED} MiB) -- refusing to co-schedule"; exit 1; fi
N=$(ls ssl_cache_336_130mm/train/*.pt | wc -l)
if [ "$N" -lt 3500 ]; then echo "!! SSL cache holds only ${N} studies"; exit 1; fi
# cold corpus runs at 253 img/s vs 1817 warm: the 7x gap is I/O, so warm it first.
t=$(date +%s); find ssl_cache_336_130mm/train -name '*.pt' -print0 | xargs -0 -P 8 -n 40 cat > /dev/null
echo "cache pre-read $(( $(date +%s) - t )) s"
echo "=== S2 MAE trial start $(date -u) ==="
$PY -m rsna.train.pretrain_mae --steps 300000 --batch-size 256 --workers 16 \
    --lr 5e-5 --dec-lr-mult 10 --lr-warmup 6000 --save-every 50000 --tag ssl1
echo "=== S2 MAE trial done $(date -u) ==="
