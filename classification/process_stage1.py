#!/usr/bin/env python3
"""Task-2 entrypoint for the Stage-1 supervised classifier.

The challenge asks for 18 abnormality probabilities, not a vision-language
model, so the Stage-1 checkpoint -- image encoder + nn.Linear(512,18) + sigmoid
-- is a valid submission on its own and needs no text encoder, no prompts and no
Q-Former. Measured on a 2:1 in-domain:external mixture it beats the deployed
pipeline on every ranked metric (AUROC 0.8378 vs 0.8264, CRG 0.5400 vs 0.5333,
F1 0.5662 vs 0.5533).

Reads /input/*.nii.gz, writes /output/results.json and /output/predictions.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = os.environ.get("RAC_REPO_ROOT", "/opt/app/repo")
sys.path.insert(0, REPO)

import predict  # noqa: E402  sets the mandatory RAC_* defaults
from calibrate import calibrate_from_env  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from dataset import PATHOLOGIES  # noqa: E402
from model.image_encoder import RACImageEncoder  # noqa: E402

VOL_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
CKPT = os.environ.get("STAGE1_CKPT", "/opt/app/models/stage1_best.pt")
# STAGE1_MODELS ensembles several Stage-1 classifiers: "arch:file,arch:file,...".
# Probabilities are averaged (prob-mean beat rank-mean by 0.009 AUROC on the
# proxy). Decorrelated errors are what pay here: the small mc3_18 backbone is
# the WORST in-domain (0.8737) and the BEST external (0.7673), so it lifts the
# mixture even though it loses head-to-head.
MODELS_SPEC = os.environ.get("STAGE1_MODELS", "").strip()
# Relative filenames in STAGE1_MODELS resolve next to STAGE1_CKPT.
MODELS_DIR = Path(os.environ.get("RAC_MODELS_DIR", str(Path(CKPT).parent)))
WORKERS = int(os.environ.get("RAC_NUM_WORKERS", "6"))
SUFFIX = ".nii.gz"


def _prefetch(vols, meta, workers):
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        pending, it = deque(), iter(vols)
        for _ in range(max(1, workers) + 1):
            v = next(it, None)
            if v is None:
                break
            pending.append((v, pool.submit(predict.preprocess_one, str(v), meta)))
        while pending:
            v, fut = pending.popleft()
            yield v.name[: -len(SUFFIX)], fut
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(predict.preprocess_one, str(nxt), meta)))
    finally:
        pool.shutdown(wait=False)


def _host_report():
    """Log host/GPU limits. A submission killed by the platform's RAM cap returns
    no container output at all, so the sizing has to be visible from the start."""
    try:
        import resource
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        print(f"[stage1] host RAM {pages/1e9:.1f} GB, prefetch workers {WORKERS}", flush=True)
    except Exception:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[stage1] GPU {torch.cuda.get_device_name(0)} free {free/1e9:.1f}/{total/1e9:.1f} GB",
              flush=True)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _host_report()

    def _load_one(arch, path):
        """Build the backbone named by `arch` and load one Stage-1 checkpoint."""
        if arch in ("vjepa", "dino"):
            os.environ["RAC_ENCODER"] = arch
            from model.image_encoder_vjepa import build_image_encoder
            enc = build_image_encoder(pretrained_kinetics=False)
        else:
            os.environ["RAC_R3D_ARCH"] = arch
            enc = RACImageEncoder(pretrained_kinetics=False)
        clf = nn.Linear(RACImageEncoder.FEAT_DIM, len(PATHOLOGIES))
        pkg = torch.load(path, map_location="cpu")
        me, ue = enc.load_state_dict(pkg["image_encoder"], strict=False)
        ce, cu = clf.load_state_dict(pkg["classifier"], strict=False)
        # Fail loud: a silent mismatch would score a random head at ~0.5 AUROC.
        assert not me and not ue, f"{arch} encoder mismatch missing={len(me)} unexpected={len(ue)}"
        assert not ce and not cu, f"{arch} classifier mismatch"
        return enc.to(dev).eval(), clf.to(dev).eval()

    if MODELS_SPEC:
        members = []
        for item in MODELS_SPEC.split(","):
            arch, _, fn = item.strip().partition(":")
            path = fn if os.path.isabs(fn) else str(MODELS_DIR / fn)
            members.append((arch, _load_one(arch, path)))
            print(f"[stage1] loaded {arch} <- {os.path.basename(path)}", flush=True)
        print(f"[stage1] device={dev} ENSEMBLE of {len(members)}: "
              f"{[a for a, _ in members]}", flush=True)
    else:
        arch = os.environ.get("RAC_R3D_ARCH", "r3d_18")
        members = [(arch, _load_one(arch, CKPT))]
        print(f"[stage1] device={dev} single ckpt={CKPT}", flush=True)
    print(f"[stage1] {len(PATHOLOGIES)} classes", flush=True)

    meta_csv = os.environ.get("RAC_META_CSV", "")
    meta = predict._load_meta_csv(meta_csv) if meta_csv and os.path.isfile(meta_csv) else None

    vols = sorted(p for p in VOL_DIR.rglob("*")
                  if p.is_file() and p.name.lower().endswith(SUFFIX))
    if not vols:
        raise SystemExit(f"No {SUFFIX} volumes under {VOL_DIR}")
    print(f"[stage1] {len(vols)} volume(s)", flush=True)

    stems, raw = [], []
    with torch.no_grad():
        for i, (stem, fut) in enumerate(_prefetch(vols, meta, WORKERS)):
            ct = fut.result().unsqueeze(0).to(dev)
            p = np.mean([torch.sigmoid(c(e(ct)))[0].float().cpu().numpy()
                         for _, (e, c) in members], axis=0)
            stems.append(stem); raw.append(p)
            if (i + 1) % 25 == 0 or i + 1 == len(vols):
                print(f"[stage1] {i + 1}/{len(vols)}", flush=True)

    # Operating-point calibration: the scorer binarises at a hard 0.5 and both
    # CRG and macro-F1 come from that matrix. The map is strictly monotone, so
    # AUROC is unchanged.
    probs = calibrate_from_env(np.stack(raw))

    preds = [{"input_image_name": s,
              "probabilities": {n: float(v) for n, v in zip(PATHOLOGIES, row)}}
             for s, row in zip(stems, probs)]
    inner = {"name": "Generated probabilities", "type": "Abnormality Classification",
             "version": {"major": 1, "minor": 0}, "predictions": preds}
    payload = [{"outputs": [{"value": inner}]}]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("results.json", "predictions.json"):
        (OUT_DIR / name).write_text(json.dumps(payload, indent=2))
    print(f"[stage1] wrote {len(preds)} predictions", flush=True)


if __name__ == "__main__":
    main()
