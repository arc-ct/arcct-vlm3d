#syntax=docker/dockerfile:1.6
# VLM3D CT Abnormality Classification -- final leaderboard submission.
# 8-slot prob-mean ensemble of 6 Stage-1 video-ResNet classifiers + per-class
# batch-quantile calibration. Weights are NOT baked in: they are uploaded as a
# separate weights.zip and mounted at /weights (see README "Weights").
#
# Build from the repository root:
#   DOCKER_BUILDKIT=1 docker build -f classification/Dockerfile -t arcct-vlm3d-cls .
#
# CUDA 12.1 runtime base: torch 2.4.1+cu121 needs its libcudart to see the GPU;
# a plain python: image silently falls back to CPU.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# The platform validator rejects root images, but runs them as root anyway, so
# everything is installed system-wide and kept world-readable (a `pip install
# --user` under /home/user is invisible to root).
RUN useradd --create-home --uid 1000 user && \
    mkdir -p /opt/app/models /opt/app/repo /input /output && \
    chown -R user:user /opt/app /input /output

WORKDIR /opt/app/
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121
COPY classification/requirements.txt /opt/app/requirements.txt
COPY classification/constraints.txt /opt/app/constraints.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt -c /opt/app/constraints.txt

ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

COPY --chown=user:user src/ /opt/app/repo/
COPY --chown=user:user classification/process_stage1.py /opt/app/process_stage1.py
COPY --chown=user:user classification/entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh && chmod -R a+rX /opt/app
USER user

# Final submission configuration (identical to the scored image).
# STAGE1_MODELS lists 8 slots over 6 checkpoints: repeating a file is how the
# integer ensemble weights (rc_r2p1d x3, all others x1) are expressed.
# RAC_CAL_KVEC is the per-class k of the batch-quantile calibration
# (src/calibrate.py), in the challenge label order.
ENV RAC_REPO_ROOT=/opt/app/repo \
    STAGE1_CKPT=/opt/app/models/stage1_best.pt \
    STAGE1_MODELS=r2plus1d_18:stage1_rc_r2p1d.pt,r2plus1d_18:stage1_rc_r2p1d.pt,r2plus1d_18:stage1_rc_r2p1d.pt,r2plus1d_18:stage1_r2p1d.pt,r2plus1d_18:stage1_r2p1d_s1.pt,r3d_18:stage1_rc_r3d.pt,r3d_18:stage1_r3djit.pt,mc3_18:stage1_mc3_s1.pt \
    RAC_QFORMER_QUERIES=30 \
    RAC_QFORMER_DEPTH=4 \
    RAC_QFORMER_HEADS=8 \
    RAC_QFORMER_POOL=mean \
    RAC_USE_ANATOMY_QFORMER=1 \
    RAC_HU_LUNG_LO=-1500 \
    RAC_HU_LUNG_HI=500 \
    RAC_PROMPT_LEN=64 \
    RAC_TEMPERATURE=0.07 \
    RAC_EVAL_APPLY_EMA=0 \
    RAC_CAL_MODE=quantile \
    RAC_CAL_K=1.75 \
    RAC_CAL_KVEC=1.766,1.711,1.745,1.957,1.745,1.506,1.560,1.506,1.855,1.944,1.773,1.814,1.458,1.362,1.355,1.588,1.636,1.670 \
    RAC_NUM_WORKERS=3 \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output

ENTRYPOINT ["/opt/app/entrypoint.sh"]
