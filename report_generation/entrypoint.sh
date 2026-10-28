#!/usr/bin/env bash
# Thin-container entrypoint: bridge /weights into the models dir, then run.
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/weights}"
MODELS_DIR="${RG_MODELS_DIR:-/opt/app/models}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"

for name in CTClip.best.pt bridge.best.pt train_reports.csv train_predicted_labels.csv sentence_bank.json stage1_r3d.pt stage1_r2p1d.pt stage1_mc3.pt stage1_r3djit.pt stage1_r2p1d_s1.pt Qwen2.5-3B-Instruct; do
    src="$WEIGHTS_DIR/$name"
    [ -e "$src" ] && ln -sfn "$src" "$MODELS_DIR/$name"
done

echo "[entrypoint] mode=${RG_MODE:-retrieval}"
ls -la "$WEIGHTS_DIR" 2>&1 | head -12 | sed 's/^/[entrypoint]   /' || true

exec python /opt/app/process_reportgen.py
