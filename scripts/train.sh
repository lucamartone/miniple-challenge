#!/usr/bin/env bash
# Default 3B-token single-GPU recipe, initialized from scratch. Pass --resume PATH to continue a checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-$PWD/.venv/bin/python}"
out_dir="${OUT_DIR:-runs/baseline_3b}"
data_dir="${DATA_DIR:-data/ultrafineweb_3b}"
gpu_id="${GPU_ID:-0}"
stage1_tokens="${STAGE1_TOKENS:-2100000000}"
decay_tokens="${DECAY_TOKENS:-900000000}"
mix_start_tokens="${MIX_START_TOKENS:-1500000000}"
warmup_tokens="${WARMUP_TOKENS:-10000000}"
adamw_lr="${ADAMW_LR:-6e-4}"
muon_lr="${MUON_LR:-8e-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "$out_dir"
exec 9>"$out_dir/job.lock"
flock -n 9 || { echo "A job already owns $out_dir" >&2; exit 1; }
resume=0
if (( $# > 0 )); then
  if (( $# == 2 )) && [[ "$1" == --resume ]]; then
    resume=1
    [[ -f "$2" ]] || { echo "Checkpoint not found: $2" >&2; exit 1; }
  elif (( $# == 1 )) && [[ "$1" == --resume=* ]]; then
    resume=1
    [[ -f "${1#--resume=}" ]] || { echo "Checkpoint not found: ${1#--resume=}" >&2; exit 1; }
  else
    echo "Usage: GPU_ID=0 OUT_DIR=runs/my_run bash scripts/train.sh [--resume CHECKPOINT]" >&2
    exit 2
  fi
fi
if (( ! resume )) && [[ -e "$out_dir/metrics.jsonl" || -e "$out_dir/latest.pt" ]]; then
  echo "Output directory already contains training. Use OUT_DIR for a fresh run or --resume PATH." >&2
  exit 1
fi
trap 'code=$?; if (( code != 0 )); then printf "failed (exit %s)\n" "$code" >"$out_dir/status"; fi' EXIT

for reference in baselines/validation/val_l2.bin baselines/validation/tokenizer.json; do
  [[ -f "$reference" ]] || { echo "Missing frozen course reference: $reference" >&2; exit 1; }
done
if [[ ! -f "$data_dir/metadata.json" ]]; then
  printf '%s\n' preparing_data >"$out_dir/status"
  prepare_args=(--out-dir "$data_dir"
    --stage1-tokens "$stage1_tokens" --decay-tokens "$decay_tokens"
    --mix-start-tokens "$mix_start_tokens" --val-tokens 5000000
    --validation-reference baselines/validation/val_l2.bin
    --tokenizer-json baselines/validation/tokenizer.json)
  "$python_bin" -u prepare_data.py "${prepare_args[@]}" \
    2>&1 | tee -a "$out_dir/prepare.log"
fi
printf '%s\n' validating_data >"$out_dir/status"
CUDA_VISIBLE_DEVICES="" "$python_bin" -u train.py --data-dir "$data_dir" --validate-data-only \
  2>&1 | tee -a "$out_dir/prepare.log"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu_id" --query-gpu=uuid --format=csv,noheader)"
printf '%s\n' waiting_for_gpu >"$out_dir/status"
while [[ -n "$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader)" ]]; do
  echo "$(date -Is) Waiting for GPU $gpu_id" >>"$out_dir/job.log"
  sleep 60
done
printf '%s\n' training >"$out_dir/status"
"$python_bin" -u train.py --data-dir "$data_dir" --out-dir "$out_dir" \
  --stage1-tokens "$stage1_tokens" --decay-tokens "$decay_tokens" \
  --mix-start-tokens "$mix_start_tokens" \
  --micro-batch 20 --grad-accum 4 --seq-len 2048 \
  --lr "$adamw_lr" --muon-lr "$muon_lr" --warmup-tokens "$warmup_tokens" \
  --attn flash --loss-backend liger --dtype bf16 \
  --sampled-softmax pr360 --full-softmax-at-decay \
  --compile --compile-mode max-autotune-no-cudagraphs \
  --log-interval 10 --eval-interval 250 --eval-sequences 128 --save-interval 1000 --seed 1337 \
  "$@" 2>&1 | tee -a "$out_dir/train.log"
printf '%s\n' complete >"$out_dir/status"
