"""Cache RAD-ChestCT volumes in the CT-RATE training format.

Task-2's hidden test set is ~half external (Boston University), and that half is
where we are weakest (0.88 in-domain vs 0.75 external macro-AUROC). The rules
allow any publicly available data, so RAD-ChestCT (Duke, public) can go into
Stage-1 training as external-domain supervision.

Training reads NPZ with key ``arr_0`` shaped (D, H, W) in HU/1000, so the raw
RAD-ChestCT volumes are resampled to CT-RATE spacing, padded/cropped to
(192, 192, 96) and re-oriented EXACTLY as the RAD-ChestCT evaluation path in the ARC-CT repository
does (transpose H,W then flip H) before being written back in (D, H, W) order.
Anything that diverges here would train on a different orientation than we
evaluate on.

Only the imgtrain split is cached: imgvalid + imgtest stay untouched so the
external proxy remains honest after this data enters training.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUB / "src"))
from dataset import _pad_crop_hwd  # noqa: E402

NPZ_DIR = os.environ.get("RAC_RADCHEST_NPZ", "data/rad_chest_ct/extracted")
OUT_DIR = Path(os.environ.get("RAC_RADCHEST_CACHE", "data/radchest_cache_192"))
SPLIT = os.environ.get("RAC_RADCHEST_SPLIT", "imgtrain")
WORKERS = int(os.environ.get("RAC_CACHE_WORKERS", "24"))
CT_SP = (1.5, 1.5, 3.0)
RC = 0.8
HWD = (192, 192, 96)


def resample(ct: np.ndarray) -> np.ndarray:
    d, h, w = ct.shape
    t = torch.from_numpy(ct.astype(np.float32))[None, None]
    sz = (max(1, round(d * RC / CT_SP[2])), max(1, round(h * RC / CT_SP[1])),
          max(1, round(w * RC / CT_SP[0])))
    return F.interpolate(t, size=sz, mode="trilinear", align_corners=False).squeeze().numpy()


def one(acc: str) -> tuple[str, str]:
    dst = OUT_DIR / f"{acc}.npz"
    if dst.exists():
        return acc, "skip"
    src = os.path.join(NPZ_DIR, f"{acc}.npz")
    if not os.path.exists(src):
        return acc, "missing"
    hu = resample(np.load(src)["ct"].astype(np.float32))          # (D, H, W) HU
    hwd = _pad_crop_hwd(np.transpose(hu, (1, 2, 0)), HWD)          # (H, W, D)
    dhw = np.transpose(hwd, (2, 0, 1))                             # (D, H, W)
    dhw = np.flip(np.transpose(dhw, (0, 2, 1)), axis=1)            # transpose H,W then flip H
    np.savez_compressed(dst, arr_0=np.ascontiguousarray(dhw / 1000.0, dtype=np.float16))
    return acc, "ok"


def main() -> None:
    labels = pd.read_csv(f"{NPZ_DIR}/{SPLIT}_Abnormality_and_Location_Labels.csv", index_col=0)
    accs = [str(a) for a in labels.index]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[rc-cache] split={SPLIT} n={len(accs)} out={OUT_DIR} workers={WORKERS}", flush=True)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, a) for a in accs]
        for i, f in enumerate(as_completed(futs), 1):
            _, st = f.result()
            counts[st] = counts.get(st, 0) + 1
            if i % 100 == 0:
                print(f"[rc-cache] {i}/{len(accs)} {counts}", flush=True)
    print(f"[rc-cache] DONE {counts}", flush=True)


if __name__ == "__main__":
    main()
