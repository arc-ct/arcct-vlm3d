#!/usr/bin/env bash
# Upload the challenge checkpoints to the Hugging Face Hub.
#
#   hf auth login
#   CLS=/path/to/classification_weights RG=/path/to/reportgen_weights ./hf/upload.sh
#
# CLS holds the 6 classification stage1_*.pt files; RG holds stage1_mc3.pt and
# CTClip.best.pt (the other report-generation members are shared with CLS).
set -euo pipefail
REPO=${REPO:-arc-ct/arcct-vlm3d}
CLS=${CLS:?set CLS}; RG=${RG:?set RG}

# private by default; flip with PUBLIC=1 once you are ready to release
if [ "${PUBLIC:-0}" = "1" ]; then VIS=--public; else VIS=--private; fi
hf repos create "$REPO" --type model $VIS || true
hf upload "$REPO" hf/README.md README.md --type model
for f in "$CLS"/stage1_{rc_r2p1d,r2p1d,r2p1d_s1,rc_r3d,r3djit,mc3_s1}.pt "$RG"/stage1_mc3.pt "$RG"/CTClip.best.pt; do
  echo "uploading $(basename "$f")"
  hf upload "$REPO" "$f" "$(basename "$f")" --type model
done
echo "done: https://huggingface.co/$REPO"
