#!/bin/bash
# Stage2 Collaborative Training with Mamba
# Uses fixed cls_loss + fixed psm reshape

CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29501 \
    opencood/tools/train_stamp.py \
    -y opencood/hypes_yaml/airv2x/lidar/det/airv2x_heal/airv2x_HEAL_collab_lidar_mamba.yaml \
    --vehicle_dir opencood/logs/airv2x_HEAL_vehicle_lidar/stage1_vehicle__2026_02_04_03_45_21 \
    --rsu_dir opencood/logs/airv2x_HEAL_rsu_lidar/stage1_rsu__2026_02_27_04_50_19 \
    --drone_dir opencood/logs/airv2x_HEAL_drone_lidar/stage1_drone__2026_03_05_02_10_33 \
    --tag stage2_mamba
