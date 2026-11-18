#!/usr/bin/env bash
# Evaluate a checkpoint:  GPU=0 bash scripts/evaluate.sh <arm> <step> <val|test|ood|all>
set -euo pipefail
ARM=${1:?arm}; STEP=${2:?step}; MODE=${3:-all}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}; GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs}
DATA_DIR=${DATA_DIR:-$REPO/data/vlaa_subset_5400}
RESULTS=${RESULTS:-$REPO/results}
CKPT=${CKPT:-$OUTPUT_DIR/checkpoints/ft_$ARM/global_step_$STEP}
MERGED=$OUTPUT_DIR/merged/${ARM}_step$STEP
OOD_SETS=${OOD_SETS:-slake vqarad mmstar mmmu_pro mathvista mathvision blink realworldqa}
: "${VERL_DIR:?set VERL_DIR to the patched verl checkout}"
export VERL_DIR PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$RESULTS/eval" "$RESULTS/ood" "$OUTPUT_DIR/merged"

"$PYTHON" - "$REPO/scripts/eval" "$CKPT" "$MERGED" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from eval_adapter_vllm import merge_fullft_checkpoint
merge_fullft_checkpoint(Path(sys.argv[2]), Path(sys.argv[3]), reuse=True)
PY

indomain() {
    "$PYTHON" "$REPO/scripts/eval/eval_adapter_vllm.py" --no-adapter --base-model "$MERGED" \
        --test-parquet "$DATA_DIR/$1.parquet" --output "$2" --save-all-responses \
        --max-new-tokens 4096 --gpu "$GPU" --gpu-memory-utilization 0.85
}
case "$MODE" in val|all) indomain val "$RESULTS/eval/${ARM}_step${STEP}__val.json" ;; esac
case "$MODE" in test|all) indomain test "$RESULTS/eval/${ARM}_S${STEP}__test.json" ;; esac
case "$MODE" in ood|all)
    for ds in $OOD_SETS; do
        "$PYTHON" "$REPO/scripts/eval/eval_ood_vllm.py" --no-adapter --base-model "$MERGED" \
            --dataset "$ds" --output "$RESULTS/ood/${ARM}_step${STEP}__${ds}_s0.json" \
            --arm-label "${ARM}_S$STEP" --seed 0 --max-new-tokens 4096 \
            --gpu "$GPU" --gpu-memory-utilization 0.85
    done ;;
esac
