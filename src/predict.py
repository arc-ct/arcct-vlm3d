#!/usr/bin/env python
"""VLM3D Task-2 (Multi-Abnormality Classification) inference entrypoint.

Reads chest-CT NIfTI volumes (.nii.gz) from an input directory and writes an
18-class per-scan abnormality prediction to an output directory. Reuses the
the model evaluation pipeline unchanged (build_model_for_eval / encode_prompts /
prompt_probs) and the training preprocessing (dataset._to_3ch_hwd /
dataset._pad_crop_hwd). Runs MASK-FREE, matching scripts/eval_radchest_fixed.py
(the validated external-transfer path), because the challenge test set has no
TotalSegmentator masks.

The 18 classes are emitted in dataset.PATHOLOGIES order, which is exactly the
challenge class order.
"""
from __future__ import annotations

import os
import sys

# --- Runtime config for the rcretrain_augmd_seed0 checkpoint. MUST be set
# before importing `eval`, which reads several of these at module-import time
# (e.g. QFORMER_NUM_QUERIES). Values come from the run's env.txt. ---
_DEFAULTS = {
    "RAC_QFORMER_QUERIES": "30",     # ckpt has 30 queries; the default 32 silently
                                     # drops the query tensor (strict=False) -> garbage.
    "RAC_QFORMER_DEPTH": "4",
    "RAC_QFORMER_HEADS": "8",
    "RAC_QFORMER_POOL": "mean",
    "RAC_USE_ANATOMY_QFORMER": "1",
    "RAC_HU_LUNG_LO": "-1500",       # training lung window (NOT the -1000/400 default)
    "RAC_HU_LUNG_HI": "500",
    "RAC_PROMPT_LEN": "64",
    "RAC_TEMPERATURE": "0.07",
    "RAC_EVAL_APPLY_EMA": "0",       # this ckpt has no EMA shadow
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# Repo root (parent of this VLM3D/ folder) so `import eval` etc. resolve.
REPO_ROOT = os.environ.get(
    "RAC_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO_ROOT)

import ast
import glob
import json
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from eval import build_model_for_eval, encode_prompts, prompt_probs  # noqa: E402
from dataset import PATHOLOGIES, _pad_crop_hwd, _to_3ch_hwd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import calibrate_from_env  # noqa: E402

INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
CKPT = os.environ.get("EVAL_CKPT", "/opt/ckpts/best.pt")
META_CSV = os.environ.get("RAC_META_CSV", "")          # optional CT-RATE-style sidecar
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", os.path.join(OUTPUT_DIR, "predictions.csv"))
OUTPUT_JSON = os.environ.get("OUTPUT_JSON", os.path.join(OUTPUT_DIR, "predictions.json"))
OUTPUT_BIN = os.environ.get("OUTPUT_BIN", os.path.join(OUTPUT_DIR, "predictions_binary.csv"))
THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", "0.5"))
MAX_VOLS = int(os.environ.get("RAC_MAX_VOLS", "0"))     # 0 = no limit (smoke-test knob)

TARGET_SPACING = (1.5, 1.5, 3.0)                        # NPZ_TARGET_SPACING (batched prep)
SPATIAL_HWD = (192, 192, 96)                            # RACDatasetV4.SPATIAL_HWD


def _load_meta_csv(path):
    import pandas as pd

    df = pd.read_csv(path)
    key = "VolumeName" if "VolumeName" in df.columns else df.columns[0]
    return {str(r[key]): r for _, r in df.iterrows()}


def _meta_row(meta, name):
    if meta is None:
        return None
    row = meta.get(name)
    if row is None:
        row = meta.get(name.replace(".nii.gz", ""))
    return row


def load_hu_and_spacing(nii_path, meta=None):
    """Return (HU volume float32 in native (i,j,k) axis order, spacing (sx,sy,sz) mm).

    Two paths:
      * Sidecar CSV (meta given, row found) -> apply RescaleSlope/Intercept +
        XYSpacing/ZSpacing exactly like the offline prep. REQUIRED for the Hugging
        Face CT-RATE release, whose headers carry no rescale and bogus (1,1,1)
        spacing (intercepts vary, e.g. -1024/-8192). Used for local testing.
      * Otherwise -> nibabel get_fdata() (applies the header's scl_slope/inter to
        give HU) + header get_zooms() spacing. This is how the challenge platform
        ships volumes (proper HU + real spacing, same as the example's SimpleITK
        read), so the container needs no CSV at submission time.
    """
    nii = nib.load(str(nii_path))
    hdr = nii.header
    name = os.path.basename(nii_path)
    row = _meta_row(meta, name)

    if row is not None:
        raw = np.asanyarray(nii.dataobj).astype(np.float32)  # raw stored values, no auto-rescale
        hu = raw * float(row.get("RescaleSlope", 1.0)) + float(row.get("RescaleIntercept", 0.0))
        try:
            xy_val = float(ast.literal_eval(str(row.get("XYSpacing", "[1.0,1.0]")))[0])
        except Exception:
            xy_val = 1.0
        try:
            z_val = float(row.get("ZSpacing", 1.0))
        except Exception:
            z_val = 1.0
        return hu, (xy_val, xy_val, z_val)

    # dtype=np.float32 matters: plain get_fdata() materialises float64 first, which is
    # 2.2 GB for a 1024x1024x259 volume and, times the prefetch threads, OOM-kills the
    # 32 GB submission host before any Python error can be reported.
    hu = nii.get_fdata(dtype=np.float32)  # header rescale applied -> HU on proper NIfTI
    zooms = [float(z) for z in hdr.get_zooms()[:3]]
    if not all(z > 0 for z in zooms) or all(abs(z - 1.0) < 1e-6 for z in zooms):
        print(f"[predict] WARN {name}: header spacing looks bogus {zooms}; if this is an "
              f"HF-format CT-RATE volume, set RAC_META_CSV to its metadata CSV.", file=sys.stderr)
    return hu, (zooms[0], zooms[1], zooms[2])


def resample_to_ctrate(vol, spacing):
    """Trilinear resample to TARGET_SPACING."""
    target = tuple(
        max(1, int(round(s * so / st)))
        for s, so, st in zip(vol.shape, spacing, TARGET_SPACING)
    )
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=target, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def preprocess_one(nii_path, meta=None):
    """Raw NIfTI -> model input tensor (3, 96, 192, 192), reproducing the training cache."""
    hu, spacing = load_hu_and_spacing(nii_path, meta)          # native (i,j,k) HU
    vol_r = resample_to_ctrate(hu, spacing)                    # resampled, native axes
    # These two transposes cancel (identity) but mirror the two code paths exactly:
    #   batched prep: transpose(2,0,1) -> (D,H,W) cache arr_0
    #   _load_npz_hwd: transpose(1,2,0) -> (H,W,D)
    vol_dhw = np.transpose(vol_r, (2, 0, 1))
    arr_hwd = np.transpose(vol_dhw, (1, 2, 0))
    # Pad value is in raw HU here. Must match the training run's RAC_PAD_HU
    # (0 = water for every checkpoint trained so far; -1000 = air for the
    # robustness retrains). The checkpoint records it under config["preproc"].
    arr_hwd = _pad_crop_hwd(arr_hwd, SPATIAL_HWD,
                            pad_value=float(os.environ.get("RAC_PAD_HU", "0")))
    ct3 = _to_3ch_hwd(arr_hwd)                                 # (3,192,192,96), 3 HU windows
    ct = torch.from_numpy(ct3).permute(0, 3, 1, 2).contiguous()  # (3,96,192,192)
    return ct


@torch.no_grad()
def mask_free_probs(clip, qf, ct, pos_embs, neg_embs, device):
    """18 per-class probs, mask-free (matches scripts/eval_radchest_fixed.py:232-245)."""
    feat_map = clip.visual_transformer.forward_spatial(ct)
    B = ct.shape[0]
    if getattr(clip, "use_anatomy_qformer", False):
        zmask = torch.zeros(B, ct.shape[2], ct.shape[3], ct.shape[4], device=device)
        hmask = torch.zeros(B, dtype=torch.bool, device=device)
        pooled = qf(feat_map, zmask, hmask)
    else:
        pooled = qf(feat_map)
    img_lat = F.normalize(pooled, dim=-1)
    return prompt_probs(img_lat, pos_embs, neg_embs).float().cpu().numpy()


@torch.no_grad()
def slice_probs(clip, qf, ct, pos_embs, neg_embs, device):
    """Slice Top-12 readout, mask-free (port of scripts/eval_radchest_slice.py:70-85).

    Scores each of the T=12 feature-grid slabs against the class prompts and
    averages, instead of pooling features first. "Top-12" is a plain mean: K is
    clamped to T, so K=12 selects nothing. Measured +0.020 external macro-AUROC
    and +0.014 in-domain over the global readout, mask-free in both cases.
    """
    from eval import TEMPERATURE  # module-level constant, re-read here for clarity

    feat = clip.visual_transformer.forward_spatial(ct)          # (B,512,T,Hp,Wp)
    B, _, T = feat.shape[0], feat.shape[1], feat.shape[2]
    A, P = len(qf.anatomy_labels), len(qf.pathology_labels)
    zeros = torch.zeros(B, ct.shape[2], ct.shape[3], ct.shape[4], device=device)
    no_mask = torch.zeros(B, dtype=torch.bool, device=device)

    per_slice = torch.zeros(B, T, P, device=device)
    for t in range(T):
        _, tokens = qf(feat[:, :, t:t + 1], zeros, no_mask, return_tokens=True)
        pt = F.normalize(tokens[:, A:A + P, :], dim=-1)
        sp = (pt * pos_embs.unsqueeze(0)).sum(-1)
        sn = (pt * neg_embs.unsqueeze(0)).sum(-1)
        per_slice[:, t, :] = F.softmax(torch.stack([sp, sn], -1) / TEMPERATURE, -1)[..., 0]
    return per_slice.mean(1).float().cpu().numpy()


def readout_probs(clip, qf, ct, pos_embs, neg_embs, device):
    """Dispatch on RAC_READOUT: 'slice12' (default) or 'global'."""
    mode = os.environ.get("RAC_READOUT", "slice12")
    if mode == "global":
        return mask_free_probs(clip, qf, ct, pos_embs, neg_embs, device)
    if mode == "slice12":
        return slice_probs(clip, qf, ct, pos_embs, neg_embs, device)
    raise ValueError(f"unknown RAC_READOUT={mode!r} (expected 'slice12' or 'global')")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] device={device} ckpt={CKPT}", flush=True)

    clip, tokenizer = build_model_for_eval(CKPT)
    clip.to(device).eval()
    qf = getattr(clip, "qformer_module", None)
    if qf is None:
        raise RuntimeError("Checkpoint has no qformer_module; wrong checkpoint?")
    qf.to(device).eval()
    # Guard against the silent 30-vs-32 query mismatch.
    n_q = qf.qformer.queries.shape[0] if hasattr(qf, "qformer") else qf.queries.shape[0]
    print(f"[predict] q-former queries loaded = {n_q} (expect 30)", flush=True)
    assert n_q == int(os.environ["RAC_QFORMER_QUERIES"]), (
        f"query count {n_q} != RAC_QFORMER_QUERIES {os.environ['RAC_QFORMER_QUERIES']}")

    pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)

    meta = _load_meta_csv(META_CSV) if META_CSV and os.path.isfile(META_CSV) else None
    if meta is not None:
        print(f"[predict] using sidecar metadata: {META_CSV} ({len(meta)} rows)", flush=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.nii.gz"), recursive=True))
    if MAX_VOLS > 0:
        files = files[:MAX_VOLS]
    if not files:
        raise RuntimeError(f"No .nii.gz volumes found under {INPUT_DIR}")
    print(f"[predict] {len(files)} volume(s) to score", flush=True)

    names, probs_all = [], []
    for i, f in enumerate(files):
        # Fail loud: a bad volume / OOM should crash, not be silently skipped.
        ct = preprocess_one(f, meta).unsqueeze(0).to(device)
        p = readout_probs(clip, qf, ct, pos_embs, neg_embs, device)[0]
        names.append(os.path.basename(f))
        probs_all.append(p)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"[predict] {i + 1}/{len(files)}", flush=True)

    probs = calibrate_from_env(np.stack(probs_all)).astype(np.float32)
    _write_outputs(names, probs)
    print(f"[predict] wrote {OUTPUT_CSV} / {OUTPUT_JSON} / {OUTPUT_BIN}", flush=True)


def _write_outputs(names, probs):
    import pandas as pd

    df = pd.DataFrame(probs, columns=PATHOLOGIES)
    df.insert(0, "VolumeName", names)
    df.to_csv(OUTPUT_CSV, index=False)

    (pd.DataFrame((probs >= THRESHOLD).astype(int), columns=PATHOLOGIES)
       .assign(VolumeName=names)
       .set_index("VolumeName").reset_index()
       .to_csv(OUTPUT_BIN, index=False))

    obj = {n: {p: float(v) for p, v in zip(PATHOLOGIES, row)}
           for n, row in zip(names, probs)}
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(obj, fh)


if __name__ == "__main__":
    main()
