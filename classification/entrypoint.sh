#!/usr/bin/env bash
set -euo pipefail
WEIGHTS_DIR="${WEIGHTS_DIR:-/weights}"
MODELS_DIR="/opt/app/models"
mkdir -p "$MODELS_DIR" "${OUTPUT_DIR:-/output}"
# Symlink every checkpoint shipped in weights.zip; STAGE1_MODELS decides which
# ones are actually loaded, so the ensemble can change without touching this.
for f in "$WEIGHTS_DIR"/*.pt; do
  [ -e "$f" ] && ln -sfn "$f" "$MODELS_DIR/$(basename "$f")"
done
ls -la "$WEIGHTS_DIR" 2>&1 | head -8 | sed 's/^/[entrypoint]   /' || true
exec python /opt/app/process_stage1.py
