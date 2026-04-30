#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/exp"

GPU_LIST="0"
SEEDS=()
DEFAULT_SEEDS=(3 312 712 644 867)

usage() {
  echo "用法："
  echo "  bash run.sh [seed1 seed2 ...] [--gpus 0,1,2,3]"
  echo "示例："
  echo "  bash run.sh 312 3 69 290 --gpus 0,1,2,3"
  echo "  bash run.sh --gpus 0"
  echo ""
  echo "默认：不传 seed 时使用: ${DEFAULT_SEEDS[*]}，不传 --gpus 时使用: ${GPU_LIST}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_LIST="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      SEEDS+=("$1")
      shift 1
      ;;
  esac
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=("${DEFAULT_SEEDS[@]}")
fi

IFS=', ' read -r -a GPU_ARR <<< "$GPU_LIST"
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "[run.sh] --gpus 解析失败：GPU_LIST='$GPU_LIST'"
  exit 1
fi

MAX_CONCURRENT="${#GPU_ARR[@]}"

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate gopt

cd "$ROOT_DIR"

WHISPER_FEAT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/data/whisper_feature/feature_aligned/whisper_block25_features"
WHISPER_STAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)/data/whisper_feature.npz"

echo "[run.sh] ROOT_DIR=$ROOT_DIR"
echo "[run.sh] EXP_ROOT=$EXP_ROOT"
echo "[run.sh] SEEDS=${SEEDS[*]}"
echo "[run.sh] GPU_LIST=$GPU_LIST (MAX_CONCURRENT=$MAX_CONCURRENT)"

active_jobs() {
  jobs -p 2>/dev/null | wc -l
}

seed_idx=0
for seed in "${SEEDS[@]}"; do
  gpu="${GPU_ARR[$((seed_idx % ${#GPU_ARR[@]}))]}"

  exp_dir="$EXP_ROOT/seed${seed}"
  mkdir -p "$exp_dir"
  log_path="$exp_dir/train.log"
  rm -f "$log_path"

  echo "[run.sh] Starting seed=$seed on GPU=$gpu -> $log_path"

  (
    env CUDA_VISIBLE_DEVICES="$gpu" nohup python train.py \
      --data-train train \
      --data-eval test \
      --exp-dir "$exp_dir" \
      --seed "$seed" \
      --am librispeech \
      --lr 5e-4 \
      --n-epochs 80 \
      --batch-size 25 \
      --loss_w_phn 1.0 \
      --loss_w_word 1.0 \
      --loss_w_utt 1.0 \
      --loss_w_consist 0.2 \
      --noise 0.02 \
      --lr-scheduler tristage \
      --embed-dim 24 \
      --goptdepth 1 \
      --goptheads 1 \
      --feat-drop 0.1 \
      --conv-kernel 31 \
      --conv-dropout 0.1 \
      --dur-dim 1 \
      --energy-dim 7 \
      --word-aspect-fusion-layers 3 \
      --word-aspect-fusion-dropout 0.15 \
      --use-whisper-feat \
      --use-pca \
      --whisper-feat-root "$WHISPER_FEAT_ROOT" \
      --whisper-stat-path "$WHISPER_STAT_PATH" \
      > "$log_path" 2>&1
  ) &

  # 控制并发：同时最多跑 GPU 数量个任务
  while [[ $(active_jobs) -ge "$MAX_CONCURRENT" ]]; do
    sleep 2
  done

  seed_idx=$((seed_idx + 1))
done

wait
echo "[run.sh] 所有任务已结束。"

