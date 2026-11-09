#!/usr/bin/env bash
# MacPorts bash/libintl can abort in its forked child before Python starts.
# exec replaces the shell without forking; use macOS's bundled bash instead.
if [[ "$OSTYPE" == darwin* && "$BASH" != /bin/bash ]]; then
  exec /bin/bash "$0" "$@"
fi
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/.model_cache"
PYTHON="${PYTHON:-python}"
STAGE="${1:-help}"
DEVICE="${DEVICE:-cuda}"
BASE="${BASE_MODEL:-local_models/bge-base-en-v1.5}"
CE="${RERANKER_MODEL:-local_models/ms-marco-MiniLM-L6-v2}"
EXP="experiments/finetune"
DATA="$EXP/data"
OUT="$EXP/run"
mkdir -p "$EXP/results" "$EXP/logs"

if [[ "$STAGE" == "prepare" ]]; then
  options=()
  if [[ "${DEVELOPMENT_ONLY_TEST:-0}" == 1 ]]; then options+=(--development-only-test); fi
  if [[ -n "${RAW_DIR:-}" ]]; then options+=(--raw-dir "$RAW_DIR"); fi
  "$PYTHON" -m experiments.finetune.prepare --train-limit 0 "${options[@]}"
elif [[ "$STAGE" == "smoke" ]]; then
  for required in "$DATA/train.json" "artifacts/indexes/finetune/passages.jsonl" "$BASE/config.json"; do
    test -f "$required" || { echo "Missing input: $required. See docs/experiments.md for data preparation and local model setup." >&2; exit 1; }
  done
  "$PYTHON" -m pytest -q "tests/finetune/test_experiment.py"
  "$PYTHON" -m experiments.finetune.train --data "$DATA" --base-model "$BASE" \
    --device "$DEVICE" --threads "${THREADS:-1}" --layers 12 --steps 2 --batch-size 2 --checkpoint-every 2 --output "$EXP/smoke_gpu"
elif [[ "$STAGE" == "train" ]]; then
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Select an NVIDIA GPU image with a working CUDA Torch build"; print(torch.cuda.get_device_name(0))'
  test ! -e "$OUT/training_config.json" || { echo 'Training output already exists. Rename run/ before starting another independent run.'; exit 1; }
  "$PYTHON" -u -m experiments.finetune.train --data "$DATA" --base-model "$BASE" \
    --device "$DEVICE" --layers 12 --steps "${STEPS:-2000}" --batch-size "${BATCH_SIZE:-8}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-1000}" --output "$OUT" 2>&1 | tee "$EXP/logs/train.log"
elif [[ "$STAGE" == "select" ]]; then
  results=()
  for checkpoint in "$OUT"/checkpoint-*; do
    test -d "$checkpoint" || continue
    name="$(basename "$checkpoint")"
    result="$EXP/results/dev_$name.json"
    "$PYTHON" -u -m experiments.finetune.evaluate --model "$checkpoint" --device "$DEVICE" \
      --samples "$DATA/dev.json" --index-dir "$EXP/indexes/$name" --output "$result" \
      2>&1 | tee "$EXP/logs/dev_$name.log"
    results+=("$result")
  done
  test "${#results[@]}" -gt 0
  "$PYTHON" -m experiments.finetune.analyze select "${results[@]}" --output "$OUT/selection.json"
elif [[ "$STAGE" == "test" ]]; then
  test "${DEVELOPMENT_ONLY_TEST:-0}" == 1 || { echo "Historical test400 uses the official dummy file. Set DEVELOPMENT_ONLY_TEST=1 only for development reproduction." >&2; exit 1; }
  test -f "$OUT/selection.json"
  test ! -e "$EXP/results/final_comparison.json" || { echo 'Final test was already completed. Preserve the frozen result; do not tune using test scores.'; exit 1; }
  selected="$($PYTHON -c "import json; print(json.load(open('$OUT/selection.json'))['selected']['model'])")"
  "$PYTHON" -u -m experiments.finetune.evaluate --model "$BASE" --device "$DEVICE" \
    --samples "$DATA/test.json" --index-dir "$EXP/indexes/base" --rerank --reranker-model "$CE" \
    --output "$EXP/results/base_test.json" 2>&1 | tee "$EXP/logs/base_test.log"
  "$PYTHON" -u -m experiments.finetune.evaluate --model "$selected" --device "$DEVICE" \
    --samples "$DATA/test.json" --index-dir "$EXP/indexes/selected" \
    --output "$EXP/results/tuned_test.json" 2>&1 | tee "$EXP/logs/tuned_test.log"
  "$PYTHON" -m experiments.finetune.analyze compare --base "$EXP/results/base_test.json" \
    --tuned "$EXP/results/tuned_test.json" --samples "$DATA/test.json" \
    --output "$EXP/results/final_comparison.json"
elif [[ "$STAGE" == "export" ]]; then
  test -f "$EXP/results/final_comparison.json"
  selected="$($PYTHON -c "import json; print(json.load(open('$OUT/selection.json'))['selected']['model'])")"
  tar -czf autodl_results.tar.gz "$EXP/results" "$EXP/logs" "$OUT/selection.json" \
    "$OUT/training_config.json" "$OUT/training_summary.json" "$OUT/loss_history.json" "$selected" \
    "$EXP/indexes/selected" "$EXP/data/audit.json"
  echo "Download $PWD/autodl_results.tar.gz"
else
  echo 'Usage: bash experiments/finetune/run.sh prepare|smoke|train|select|test|export'
  echo 'Prepare requires raw data; later stages require local pretrained models. Run stages in that order. Train defaults: 2000 steps, batch 8, full BGE fine-tuning.'
fi
