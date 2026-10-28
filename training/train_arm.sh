#!/usr/bin/env bash
# Train one Stage-1 ensemble member (3D video-ResNet + linear 18-way head).
# Single GPU; ~6-12 h on one A100 80 GB for 6 epochs.
#
#   ARM=r2p1d RAC_R3D_ARCH=r2plus1d_18 RAC_SEED=0 bash training/train_arm.sh
#
# The exact settings of every shipped member are in configs/stage1_arms.env.
# The best checkpoint (highest CT-RATE validation macro-AUROC) is written to
# runs/stage1_${ARM}/stage1_best.pt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM=${ARM:?set ARM, e.g. ARM=r2p1d}

# Preprocessed CT-RATE (see README "Data"). Expected layout under RAC_DATA_ROOT:
#   mps_ct_npz/{train,valid}/*.npz
#   multi_abnormality_labels/{train,valid}_predicted_labels.csv
#   ct_reports/{train,valid}_reports.csv
export RAC_DATA_ROOT=${RAC_DATA_ROOT:-$ROOT/data}

export RAC_SEED=${RAC_SEED:-0}
export RAC_KINETICS_REQUIRED=${RAC_KINETICS_REQUIRED:-1}
export RAC_STAGE1_RESULTS_DIR=${RAC_STAGE1_RESULTS_DIR:-$ROOT/runs/stage1_${ARM}}

export RAC_R3D_ARCH=${RAC_R3D_ARCH:-r3d_18}           # r3d_18 | mc3_18 | r2plus1d_18
export RAC_STAGE1_LOSS=${RAC_STAGE1_LOSS:-asl}        # asl | bce
export RAC_STAGE1_EPOCHS=${RAC_STAGE1_EPOCHS:-6}
export RAC_STAGE1_BATCH_SIZE=${RAC_STAGE1_BATCH_SIZE:-8}
export RAC_STAGE1_LR=${RAC_STAGE1_LR:-1e-4}
export RAC_NUM_WORKERS=${RAC_NUM_WORKERS:-6}

# Augmentation (identical for every shipped member except the HU-window jitter arm).
export RAC_ENABLE_ANATOMY_FLIPS=${RAC_ENABLE_ANATOMY_FLIPS:-1}
export RAC_RANDOM_CROP=${RAC_RANDOM_CROP:-1}
export RAC_INTENSITY_JITTER_HU=${RAC_INTENSITY_JITTER_HU:-30}
export RAC_GAUSSIAN_NOISE_SIGMA=${RAC_GAUSSIAN_NOISE_SIGMA:-0}
export RAC_HU_WINDOW_JITTER=${RAC_HU_WINDOW_JITTER:-0}
export RAC_HU_LUNG_LO=${RAC_HU_LUNG_LO:--1500}
export RAC_HU_LUNG_HI=${RAC_HU_LUNG_HI:-500}

# Optional extra public training data (RAD-ChestCT imgtrain), built by
# training/build_radchest_cache.py. Leave RAC_EXTRA_TRAIN empty for CT-RATE only.
export RAC_EXTRA_TRAIN=${RAC_EXTRA_TRAIN:-}
export RAC_EXTRA_REPEAT=${RAC_EXTRA_REPEAT:-1}

mkdir -p "$RAC_STAGE1_RESULTS_DIR"
env | grep -E "^RAC_" | sort > "$RAC_STAGE1_RESULTS_DIR/env.txt"
echo "[stage1] arm=$ARM arch=$RAC_R3D_ARCH loss=$RAC_STAGE1_LOSS seed=$RAC_SEED extra=${RAC_EXTRA_TRAIN:-none}"
cd "$ROOT"
python -u training/train_stage1.py
