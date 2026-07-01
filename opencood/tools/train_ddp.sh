#!/usr/bin/env bash
set -euo pipefail

# Simple DDP launcher with configurable GPUs and rendezvous endpoint.
# Usage examples:
#   GPUS=0,1,2,3 NPROC=4 MASTER_PORT=29519 bash opencood/tools/train_ddp.sh
#   GPUS=0,1 NPROC=2 bash opencood/tools/train_ddp.sh
#   CONFIG=opencood/hypes_yaml/airv2x/lidar/det/airv2x_heal/airv2x_HEAL_collab_lidar.yaml bash opencood/tools/train_ddp.sh

ROOT_DIR="/data1/wangyh/heal_framework"
CONFIG=${CONFIG:-"opencood/hypes_yaml/airv2x/lidar/det/airv2x_heal/airv2x_HEAL_collab_lidar.yaml"}
GPUS=${GPUS:-"0,1,2,3"}
NPROC=${NPROC:-4}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29511}

cd "${ROOT_DIR}"

# Print resolved settings for visibility
echo "Using CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${GPUS}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
  torchrun --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  opencood/tools/train_stamp.py \
  --hypes_yaml "${CONFIG}"
