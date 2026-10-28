#syntax=docker/dockerfile:1.6
# VLM3D CT Report Generation -- final leaderboard submission (RG_MODE=hybrid).
# A 4-model Stage-1 ensemble predicts the 18 findings; the container then selects
# the CT-RATE training report whose label row best matches them (label-ranked
# retrieval), breaks ties by BLEU consensus (MBR), and edits the chosen report so
# its findings agree with the prediction. Weights and the report gallery are NOT
# baked in: they are mounted at /weights (see README "Weights").
#
# Build from the repository root:
#   DOCKER_BUILDKIT=1 docker build -f report_generation/Dockerfile -t arcct-vlm3d-rg .

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN useradd --create-home --uid 1000 user && \
    mkdir -p /opt/app/models /opt/app/repo /input /output && \
    chown -R user:user /opt/app /input /output

WORKDIR /opt/app/
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121
COPY report_generation/requirements.txt /opt/app/requirements.txt
COPY report_generation/constraints.txt /opt/app/constraints.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt -c /opt/app/constraints.txt

# The entrypoint builds the ARC-CT model at start-up (it also serves the
# retrieval/generate modes), which needs this text encoder. The evaluation
# runs offline, so it is cached at build time.
ENV HF_HOME=/opt/app/hf-cache
RUN python -c "from transformers import BertTokenizer, BertModel; \
    m='microsoft/BiomedVLP-CXR-BERT-specialized'; \
    BertTokenizer.from_pretrained(m, do_lower_case=True); \
    BertModel.from_pretrained(m); print('text encoder cached')"
ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

COPY --chown=user:user src/ /opt/app/repo/
# The report-generation image shipped its own revision of these two modules.
COPY --chown=user:user report_generation/predict.py report_generation/calibrate.py /opt/app/repo/
# Canonical per-finding sentences mined from CT-RATE training reports; used by
# the edit step to state a predicted finding the selected report omits.
COPY --chown=user:user report_generation/sentence_bank.json /opt/app/models/sentence_bank.json
COPY --chown=user:user report_generation/process_reportgen.py /opt/app/process_reportgen.py
COPY --chown=user:user report_generation/entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh && chmod -R a+rX /opt/app
USER user

# Final submission configuration (identical to the scored image).
ENV RAC_REPO_ROOT=/opt/app/repo \
    EVAL_CKPT=/opt/app/models/CTClip.best.pt \
    RG_MODELS_DIR=/opt/app/models \
    RG_MODE=hybrid \
    STAGE1_MODELS=r2plus1d_18:stage1_r2p1d.pt,r3d_18:stage1_r3djit.pt,r2plus1d_18:stage1_r2p1d_s1.pt,mc3_18:stage1_mc3.pt \
    RG_MBR_SLACK=0.10 \
    RG_MBR_CAP=60 \
    RG_BATCH=4 \
    RAC_READOUT=slice12 \
    RAC_CAL_K=1.25 \
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
    RAC_NUM_WORKERS=6 \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output

ENTRYPOINT ["/opt/app/entrypoint.sh"]
