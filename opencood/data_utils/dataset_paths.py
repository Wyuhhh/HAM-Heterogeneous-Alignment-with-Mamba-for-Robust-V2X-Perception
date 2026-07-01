"""
Centralized dataset path configuration for AirV2X.

These values are used as fallbacks when no environment variables are set.
To override via env vars, set one of:
  - AIRV2X_TRAIN_DIR / AIRV2X_VAL_DIR / AIRV2X_TEST_DIR
  - OPENCOOD_TRAIN_DIR / OPENCOOD_VAL_DIR / OPENCOOD_TEST_DIR
  - AIRV2X_DATA_ROOT / OPENCOOD_DATA_ROOT / DATASET_ROOT
"""

# User-provided dataset directories
TRAIN_DIR = "/data1/wangyh/airdata/AirV2X-Perception/train/train/"
VAL_DIR = "/data1/wangyh/airdata/AirV2X-Perception/val/val/"
TEST_DIR = "/data/wangyh/test/"
