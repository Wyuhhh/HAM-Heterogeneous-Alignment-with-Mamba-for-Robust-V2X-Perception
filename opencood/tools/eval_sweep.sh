#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/data1/wangyh/heal_framework/opencood/logs/airv2x_HEAL_collab_lidar/stage2_mamba_lossfix_2gpu_nohup_final__2026_03_23_07_58_52"
EPOCH=7
FUSION="late"

SCORES=(0.05 0.10 0.20 0.30)
NMSS=(0.10 0.20 0.30)

echo "model_dir=${MODEL_DIR}"
echo "epoch=${EPOCH}, fusion=${FUSION}"

for s in "${SCORES[@]}"; do
  for n in "${NMSS[@]}"; do
    echo "======================================================"
    echo "Running score_threshold=${s}, nms_thresh=${n}"
    python -u opencood/tools/inference.py \
      --model_dir "${MODEL_DIR}" \
      --fusion_method "${FUSION}" \
      --eval_epoch "${EPOCH}" \
      --save_pred \
      --score_threshold "${s}" \
      --nms_thresh "${n}" | tee "sweep_s${s}_n${n}.log"
  done
done

echo "Done. Check logs: sweep_s*_n*.log"
