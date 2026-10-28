"""Dataset and anatomy helpers for the final the model TIA recipe.

The executable pipeline uses the local CT-RATE NPZ cache under
``/data/data``. Volumes are stored as HU / 1000 and
are cropped to a paper-faithful CT-RATE spatial grid:

    input cache: [H, W, D] = [192, 192, 96]
    model input: [C, D, H, W] = [3, 96, 192, 192]

The three channels are clinical HU windows: lung, soft tissue, and bone. Fine
TotalSegmentator masks are loaded with the same spatial crop and returned in
model order as [1, D, H, W].
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
import monai.transforms as mt


PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
N_PATH = len(PATHOLOGIES)


FINE_LABEL_NAMES = {
    0: "background",
    1: "lung_upper_lobe_left",
    2: "lung_lower_lobe_left",
    3: "lung_upper_lobe_right",
    4: "lung_middle_lobe_right",
    5: "lung_lower_lobe_right",
    6: "trachea",
    7: "heart",
    8: "aorta",
    9: "vessels",
    10: "esophagus",
}
N_FINE = 10


PATHOLOGY_FINE_ORGANS = {
    "Medical material": [],
    "Arterial wall calcification": [8],
    "Cardiomegaly": [7],
    "Pericardial effusion": [7],
    "Coronary artery wall calcification": [7, 8],
    "Hiatal hernia": [10],
    "Lymphadenopathy": [9],
    "Emphysema": [1, 3],
    "Atelectasis": [1, 2, 3, 4, 5],
    "Lung nodule": [1, 2, 3, 4, 5],
    "Lung opacity": [1, 2, 3, 4, 5],
    "Pulmonary fibrotic sequela": [2, 5],
    "Pleural effusion": [2, 5],
    "Mosaic attenuation pattern": [1, 2, 3, 4, 5],
    "Peribronchial thickening": [1, 2, 3, 4, 5, 6],
    "Consolidation": [1, 2, 3, 4, 5],
    "Bronchiectasis": [1, 2, 3, 4, 5],
    "Interlobular septal thickening": [1, 2, 3, 4, 5],
}

# R-Map v2 (T1.15 from FINAL_PIPELINE_2026-06-13.md §3.5). Env-gated, default OFF
# so v10/v11/v12/v13/v14 scripts are unaffected. Three under-routed classes get
# anatomically-faithful organ unions:
#   Lymphadenopathy: [9] -> []        (distributed mediastinal/hilar/axillary)
#   Emphysema:       [1, 3] -> [1-5]  (bilateral, all lobes)
#   Pulm fibrotic:   [2, 5] -> [1-5]  (UIP/NSIP not exclusively basal)
if os.environ.get("RAC_PATHOLOGY_MAP_V2", "0") == "1":
    PATHOLOGY_FINE_ORGANS["Lymphadenopathy"] = []
    PATHOLOGY_FINE_ORGANS["Emphysema"] = [1, 2, 3, 4, 5]
    PATHOLOGY_FINE_ORGANS["Pulmonary fibrotic sequela"] = [1, 2, 3, 4, 5]
    print(f"[RAC] PATHOLOGY_MAP_V2 ENABLED (R-Map): Lymph->[], Emphysema->[1-5], Pulm fibrotic->[1-5]")

# Built only from classes that are actually in PATHOLOGIES. A site that drops adult
# classes (pediatric: no coronary/arterial calcification, no hiatal hernia) leaves them
# in PATHOLOGY_FINE_ORGANS, and a bare PATHOLOGIES.index() would raise ValueError at
# import time -- before anything else has a chance to report a clearer error.
ORGAN_TO_PATHIDX: dict[int, list[int]] = {}
_PATH_INDEX = {_n: _i for _i, _n in enumerate(PATHOLOGIES)}
for _pname, _organs in PATHOLOGY_FINE_ORGANS.items():
    _pidx = _PATH_INDEX.get(_pname)
    if _pidx is None:
        continue  # mapped class not in this site's schema
    for _organ in _organs:
        ORGAN_TO_PATHIDX.setdefault(_organ, []).append(_pidx)


def pathology_organ_lists(pathologies=None, verbose=True):
    """Per-pathology anatomy-region lists for the Q-Former, safe for a new schema.

    ``PATHOLOGY_FINE_ORGANS[name]`` raises KeyError for any class a new site adds, so
    every caller building an AnatomyQFormer must go through this. An unmapped class gets
    ``[]``, which means "no anatomical restriction" -- the query attends everywhere, the
    same treatment Lymphadenopathy gets under RAC_PATHOLOGY_MAP_V2. That is a sane
    default for diffuse or non-anatomical findings (post-surgical change, medical
    material) and a poor one for focal findings that do have a home region, so the
    unmapped list is printed rather than silently accepted.
    """
    names = list(PATHOLOGIES if pathologies is None else pathologies)
    unmapped = [n for n in names if n not in PATHOLOGY_FINE_ORGANS]
    if unmapped and verbose:
        print(f"[RAC] {len(unmapped)} pathology class(es) have no PATHOLOGY_FINE_ORGANS entry "
              f"and will use unrestricted (global) attention: {', '.join(unmapped)}. "
              f"Add region ids 1-10 in dataset.PATHOLOGY_FINE_ORGANS if they are focal.",
              flush=True)
    return [PATHOLOGY_FINE_ORGANS.get(n, []) for n in names]


REGION_KEYWORDS = {
    1: ["left upper", "upper left", "left lung upper", "lingula", "left apic", "left upper lobe"],
    2: ["left lower", "lower left", "left lung lower", "left base", "left basal", "left hemidiaphragm", "left lower lobe"],
    3: ["right upper", "upper right", "right lung upper", "right apic", "right upper lobe"],
    4: ["right middle", "middle lobe", "right middle lobe"],
    5: ["right lower", "lower right", "right lung lower", "right base", "right basal", "right hemidiaphragm", "right lower lobe"],
    6: ["trachea", "carina", "main bronch", "tracheal"],
    7: ["heart", "cardiac", "pericardial", "cardiomegal", "atrial", "ventricle", "myocardial", "pericardium"],
    8: ["aorta", "aortic", "coronary"],
    9: ["pulmonary artery", "pulmonary vein", "vascular", "mediastinal vessel", "lymph", "lymphadenopathy", "lymph node", "subclavian", "brachiocephalic"],
    10: ["esophagus", "esophageal", "hiatal hernia", "gastroesophageal"],
}
LUNG_GENERAL_KEYWORDS = [
    "lung", "pulmonary", "pleural", "bronch", "emphysema", "consolidation",
    "atelectasis", "nodule", "opacity", "fibrosis", "fibrotic", "interlobular",
    "mosaic", "septal", "air space", "airspace", "lobe",
]


def localize_findings(findings_text: str, organ_label: int, max_chars: int = 400) -> str:
    """Fallback rule-based organ sentence extraction."""
    sentences = re.split(r"(?<=[.!?])\s+", findings_text.strip())
    keywords = REGION_KEYWORDS.get(organ_label, [])
    relevant = []
    for sent in sentences:
        s = sent.lower()
        if any(k in s for k in keywords):
            relevant.append(sent)
        elif organ_label in range(1, 6) and any(k in s for k in LUNG_GENERAL_KEYWORDS):
            relevant.append(sent)
    text = " ".join(relevant) if relevant else findings_text
    return text[:max_chars]


def _coerce_region_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, dict):
        for key in ("text", "sentence", "sentences", "findings"):
            if key in value:
                return _coerce_region_text(value[key])
    return str(value)


def load_region_cache(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        cache = json.load(f)
    print(f"[RAC] Loaded region cache: {path} ({len(cache):,} top-level keys)")
    return cache


def region_text_from_cache(cache: dict[str, Any], accession: str, organ_label: int, findings_text: str) -> str:
    """Return cached LLM organ text if present, otherwise keyword fallback."""
    keys = (
        f"{accession}|{organ_label}",
        f"{accession}:{organ_label}",
        f"{accession.replace('.nii.gz', '')}|{organ_label}",
        f"{accession.replace('.nii.gz', '')}:{organ_label}",
    )
    for key in keys:
        text = _coerce_region_text(cache.get(key))
        if text.strip():
            return text

    nested = cache.get(accession) or cache.get(accession.replace(".nii.gz", ""))
    if isinstance(nested, dict):
        for key in (str(organ_label), organ_label, FINE_LABEL_NAMES.get(organ_label, "")):
            text = _coerce_region_text(nested.get(key))
            if text.strip():
                return text

    return localize_findings(findings_text, organ_label)


def _npz_to_mask_path(npz_path: str, mask_root: str) -> str:
    p = Path(npz_path)
    try:
        idx = list(p.parts).index("mps_ct_npz")
        rel = os.path.join(*p.parts[idx + 2:])
    except ValueError:
        rel = p.name
    return os.path.join(mask_root, rel.replace(".npz", ".nii.gz"))


def _pad_crop_hwd(arr: np.ndarray, target_shape: tuple[int, int, int],
                  pad_value: float = 0.0) -> np.ndarray:
    """Pad/crop to target_shape. ``pad_value`` is in the units of ``arr``.

    Callers on raw HU should pass air (-1000) rather than the 0 default, which is
    water and produces a border no real scan has. Mask callers MUST keep 0.0: a
    negative pad rounds to -1 and wraps to 255 under ``.astype(np.uint8)``.
    """
    if arr.shape == target_shape:
        return arr
    t = mt.Compose([
        mt.ResizeWithPadOrCrop(spatial_size=(240, 240, 120), mode="constant", value=pad_value),
        mt.CenterSpatialCrop(roi_size=target_shape),
    ])
    return np.asarray(t(arr[np.newaxis].astype(np.float32))[0])


def _load_fine_mask(mask_path: str, target_shape=(192, 192, 96), fail_fast=True):
    try:
        img = nib.load(mask_path)
        data = np.round(img.get_fdata()).astype(np.uint8)
        if data.ndim == 4:
            data = data[..., 0]
        if data.shape != target_shape:
            data = np.round(_pad_crop_hwd(data, target_shape)).astype(np.uint8)
        return data
    except Exception:
        if fail_fast:
            raise
        return None


def _window_hu(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    hu = np.clip(hu, lo, hi)
    return ((hu - lo) / max(hi - lo, 1.0)).astype(np.float32)


def _channel_fill_values(hu_value: float) -> np.ndarray:
    """Where `hu_value` lands in each of the 3 normalized window channels."""
    windows = [
        (float(os.environ.get("RAC_HU_LUNG_LO", "-1000")), float(os.environ.get("RAC_HU_LUNG_HI", "400"))),
        (float(os.environ.get("RAC_HU_SOFT_LO", "-160")), float(os.environ.get("RAC_HU_SOFT_HI", "240"))),
        (float(os.environ.get("RAC_HU_BONE_LO", "300")), float(os.environ.get("RAC_HU_BONE_HI", "2000"))),
    ]
    return np.array([np.clip((hu_value - lo) / max(hi - lo, 1.0), 0.0, 1.0)
                     for lo, hi in windows], dtype=np.float32)


def _jitter_window(lo: float, hi: float, jitter: float) -> tuple:
    """Randomly perturb one HU window's level and width.

    Level is shifted by up to ``jitter`` x the window width; width is scaled by
    ``exp(U(-jitter, +jitter))``. Sampling the display window per iteration stops
    the encoder from binding to CT-RATE's exact intensity code book, which is the
    failure our external HU measurements point at (CT-RATE has 16.3% of voxels
    below -1000 HU and a max of 3071, while RAD-ChestCT ships clipped to
    [-1000, 1000], so the lung channel can never fall below 0.25 externally).
    """
    if jitter <= 0.0:
        return lo, hi
    centre = 0.5 * (lo + hi)
    width = max(hi - lo, 1.0)
    centre += float(np.random.uniform(-jitter, jitter)) * width
    width *= float(np.exp(np.random.uniform(-jitter, jitter)))
    return centre - 0.5 * width, centre + 0.5 * width


def _to_3ch_hwd(hu: np.ndarray, window_jitter: float = 0.0) -> np.ndarray:
    # Optional global HU clamp, applied BEFORE windowing so train and every eval
    # path share it. CT-RATE volumes carry scanner-dependent out-of-FOV fill
    # values (-1024 / -1844 / -2362 / -8192 across scans) and metal up to
    # ~21000 HU, while public external releases (RAD-ChestCT) are clipped to
    # [-1000, 1000]. Clamping makes the intensity code book domain-invariant.
    # Default is off (+/-inf) so existing runs are bit-identical.
    floor = float(os.environ.get("RAC_HU_FLOOR", "-inf"))
    ceil = float(os.environ.get("RAC_HU_CEIL", "inf"))
    if floor != float("-inf") or ceil != float("inf"):
        hu = np.clip(hu, floor, ceil)
    lung_lo = float(os.environ.get("RAC_HU_LUNG_LO", "-1000"))
    lung_hi = float(os.environ.get("RAC_HU_LUNG_HI", "400"))
    soft_lo = float(os.environ.get("RAC_HU_SOFT_LO", "-160"))
    soft_hi = float(os.environ.get("RAC_HU_SOFT_HI", "240"))
    bone_lo = float(os.environ.get("RAC_HU_BONE_LO", "300"))
    bone_hi = float(os.environ.get("RAC_HU_BONE_HI", "2000"))
    if window_jitter > 0.0:
        lung_lo, lung_hi = _jitter_window(lung_lo, lung_hi, window_jitter)
        soft_lo, soft_hi = _jitter_window(soft_lo, soft_hi, window_jitter)
        bone_lo, bone_hi = _jitter_window(bone_lo, bone_hi, window_jitter)
    return np.stack([
        _window_hu(hu, lung_lo, lung_hi),
        _window_hu(hu, soft_lo, soft_hi),
        _window_hu(hu, bone_lo, bone_hi),
    ], axis=0)


class RACDatasetV4(Dataset):
    """CT-RATE NPZ dataset with 3-channel HU windows and optional TS masks."""

    SPATIAL_HWD = (192, 192, 96)
    PAD_HWD = (240, 240, 120)

    def __init__(
        self,
        data_folder,
        reports_csv,
        labels_csv,
        mask_root=None,
        region_cache_path="",
        is_train=True,
        limit=0,
        fail_fast=True,
        volume_list_path="",
    ):
        self.data_folder = data_folder
        self.mask_root = mask_root
        self.is_train = is_train
        self.fail_fast = fail_fast
        self.region_cache = load_region_cache(region_cache_path)
        self.volume_list = None
        if volume_list_path and os.path.isfile(volume_list_path):
            with open(volume_list_path) as f:
                self.volume_list = {ln.strip() for ln in f if ln.strip()}
            print(f"[CTRATE] volume_list filter active: {len(self.volume_list)} vols ({volume_list_path})", flush=True)
        self.enable_flips = is_train and os.environ.get("RAC_ENABLE_ANATOMY_FLIPS", "0") == "1"
        self.require_mask = is_train and os.environ.get("RAC_REQUIRE_MASK", "0") == "1"
        # Stage-2 augmentation pack (train-only, defaults off).
        self.enable_random_crop = is_train and os.environ.get("RAC_RANDOM_CROP", "0") == "1"
        self.intensity_jitter_hu = float(os.environ.get("RAC_INTENSITY_JITTER_HU", "0")) if is_train else 0.0
        self.gaussian_noise_sigma = float(os.environ.get("RAC_GAUSSIAN_NOISE_SIGMA", "0")) if is_train else 0.0
        # Mask-dropout (train-only): with prob p, treat a mask-carrying sample as
        # maskless so the mask-free inference path (used on external data) is trained
        # and the model does not overfit to oracle TS-mask routing. Default off.
        self.mask_dropout = float(os.environ.get("RAC_MASK_DROPOUT", "0")) if is_train else 0.0
        # External-release simulation (train-only, default off): public external
        # cohorts ship HU clipped to a narrow range (RAD-ChestCT: [-1000, 1000]),
        # which flattens exactly the high-HU evidence our worst-transferring
        # classes rely on (arterial/coronary calcification, medical material).
        # With prob p, clip a training sample the same way so the model learns
        # both intensity regimes rather than only CT-RATE's.
        # Per-sample HU window jitter (train-only, default off). See _jitter_window.
        self.hu_window_jitter = float(os.environ.get("RAC_HU_WINDOW_JITTER", "0")) if is_train else 0.0
        self.hu_clip_sim_prob = float(os.environ.get("RAC_HU_CLIP_SIM_PROB", "0")) if is_train else 0.0
        self.hu_clip_sim_hi = (float(os.environ.get("RAC_HU_CLIP_SIM_HI_LO", "900")),
                               float(os.environ.get("RAC_HU_CLIP_SIM_HI_HI", "2000")))

        if not os.path.isdir(data_folder):
            raise FileNotFoundError(f"Missing data folder: {data_folder}")
        if not os.path.isfile(reports_csv):
            raise FileNotFoundError(f"Missing reports CSV: {reports_csv}")
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")

        df_rep = pd.read_csv(reports_csv)
        self.acc2text = {}
        self.acc2findings = {}
        self.acc2impressions = {}
        for _, row in df_rep.iterrows():
            acc = row["VolumeName"]
            findings = str(row.get("Findings_EN", "") or "")
            impressions = str(row.get("Impressions_EN", "") or "")
            # T1.10 + §3.3 option B: impression-first concat (env-gated, default OFF).
            # When ON, the diagnostic-conclusion phrase enters the first tokens of the
            # 256/384 budget instead of getting truncated in the tail. Targets the four
            # weakest v10 classes (Pulm fibrotic 0.6905, Lung nodule 0.7374, Lymph
            # 0.7875, Mosaic 0.8017) whose pathognomonic language lives in impressions.
            if os.environ.get("RAC_IMPRESSION_FIRST", "0") == "1":
                concat = (impressions + " " + findings).strip()
            else:
                concat = (findings + " " + impressions).strip()
            text = re.sub(r"[\"'()]", "", concat)
            self.acc2text[acc] = text
            self.acc2findings[acc] = re.sub(r"[\"'()]", "", findings)
            self.acc2impressions[acc] = re.sub(r"[\"'()]", "", impressions)

        df_lab = pd.read_csv(labels_csv)
        self.acc2label = {}
        for _, row in df_lab.iterrows():
            acc = row["VolumeName"]
            vals = [float(row.get(c, 0.0) or 0.0) for c in PATHOLOGIES]
            self.acc2label[acc] = np.asarray(vals, dtype=np.float32)

        skipped_report = skipped_label = skipped_nomask = 0
        self.samples = []
        stop_scan = False
        for pf in tqdm.tqdm(sorted(glob.glob(os.path.join(data_folder, "*"))),
                            desc=f"Scan {os.path.basename(data_folder)}", leave=False):
            if stop_scan:
                break
            for af in sorted(glob.glob(os.path.join(pf, "*"))):
                if stop_scan:
                    break
                for npz in sorted(glob.glob(os.path.join(af, "*.npz"))):
                    acc = os.path.basename(npz).replace(".npz", ".nii.gz")
                    if self.volume_list is not None and acc not in self.volume_list:
                        continue
                    if acc not in self.acc2text:
                        skipped_report += 1
                        continue
                    if acc not in self.acc2label:
                        skipped_label += 1
                        continue
                    mask_path = None
                    if mask_root:
                        mp = _npz_to_mask_path(npz, mask_root)
                        if os.path.exists(mp):
                            mask_path = mp
                    if self.require_mask and mask_path is None:
                        skipped_nomask += 1
                        continue
                    self.samples.append((
                        npz,
                        acc,
                        self.acc2text[acc],
                        self.acc2findings.get(acc, ""),
                        self.acc2label[acc],
                        mask_path,
                    ))
                    if limit and limit > 0 and len(self.samples) >= limit:
                        stop_scan = True
                        break

        n_masked = sum(1 for s in self.samples if s[5])
        print(
            f"[RAC] {os.path.basename(data_folder)}: {len(self.samples):,} samples, "
            f"{n_masked:,} with masks, skipped_report={skipped_report:,}, "
            f"skipped_label={skipped_label:,}, skipped_nomask={skipped_nomask:,} "
            f"(require_mask={self.require_mask})"
        )
        if not self.samples:
            raise RuntimeError(f"No usable samples found in {data_folder}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_npz_hwd(npz_path: str) -> np.ndarray:
        arr = np.load(npz_path)["arr_0"]
        arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        # arr is HU/1000 here, so the pad value has to be scaled the same way.
        pad_hu = float(os.environ.get("RAC_PAD_HU", "0"))
        arr = _pad_crop_hwd(arr, RACDatasetV4.SPATIAL_HWD, pad_value=pad_hu / 1000.0)
        return arr * 1000.0

    def _random_crop_192_from_240(self, ct_chwd: torch.Tensor, mask_hwd: torch.Tensor):
        """Pad to PAD_HWD then take a random SPATIAL_HWD window (paired image+mask).

        ct_chwd: (C, H, W, D) in SPATIAL_HWD geometry.
        mask_hwd: (H, W, D) uint8 in SPATIAL_HWD geometry.
        Returns the same shapes after random crop within the padded volume.
        """
        if not self.enable_random_crop:
            return ct_chwd, mask_hwd
        Ht, Wt, Dt = self.SPATIAL_HWD
        Hp, Wp, Dp = self.PAD_HWD
        # Center-pad ct and mask into PAD_HWD.
        pad_h = (Hp - Ht) // 2
        pad_w = (Wp - Wt) // 2
        pad_d = (Dp - Dt) // 2
        # The border is filled in ALREADY-WINDOWED [0,1] space, so a plain zero
        # means each channel's own window floor: -1500 HU lung / -160 HU soft /
        # +300 HU bone -- a physically impossible border pasted on every training
        # sample. With RAC_PAD_HU set, fill with that HU value per channel instead.
        fill = _channel_fill_values(float(os.environ.get("RAC_PAD_HU", "0"))) \
            if os.environ.get("RAC_PAD_HU") else None
        if fill is not None and ct_chwd.shape[0] == 3:
            ct_padded = torch.empty(ct_chwd.shape[0], Hp, Wp, Dp, dtype=ct_chwd.dtype)
            for c in range(3):
                ct_padded[c].fill_(float(fill[c]))
        else:
            ct_padded = torch.zeros(ct_chwd.shape[0], Hp, Wp, Dp, dtype=ct_chwd.dtype)
        ct_padded[:, pad_h:pad_h + Ht, pad_w:pad_w + Wt, pad_d:pad_d + Dt] = ct_chwd
        mask_padded = torch.zeros(Hp, Wp, Dp, dtype=mask_hwd.dtype)
        mask_padded[pad_h:pad_h + Ht, pad_w:pad_w + Wt, pad_d:pad_d + Dt] = mask_hwd
        # Sample top-left corner uniformly inside the padded volume.
        h0 = int(np.random.randint(0, Hp - Ht + 1))
        w0 = int(np.random.randint(0, Wp - Wt + 1))
        d0 = int(np.random.randint(0, Dp - Dt + 1))
        ct_out = ct_padded[:, h0:h0 + Ht, w0:w0 + Wt, d0:d0 + Dt].contiguous()
        mask_out = mask_padded[h0:h0 + Ht, w0:w0 + Wt, d0:d0 + Dt].contiguous()
        return ct_out, mask_out

    def _intensity_jitter(self, ct_chwd: torch.Tensor, hu_delta: float) -> torch.Tensor:
        """Per-channel HU shift drawn uniform[-hu_delta, +hu_delta].

        The cached volumes are stored as HU windowed to [0, 1] per channel, so
        shifting by raw HU units would over-shift the normalized signal. By
        default (v10 behavior) we rescale hu_delta by 1400 HU globally
        (lung-window width) — which under-jitters bone (1700 HU) and over-
        jitters soft tissue (400 HU).

        When ``RAC_JITTER_PER_CHANNEL=1`` (T1.7 fix from
        ``FINAL_PIPELINE_2026-06-13.md`` §1), each channel is rescaled by its
        own window width: lung=1400, soft=400, bone=1700. Env-gated, default
        OFF so v10/v11/v12/v13/v14 scripts produce bit-identical jitter to
        before this fix.
        """
        if hu_delta <= 0.0:
            return ct_chwd
        C = ct_chwd.shape[0]
        if os.environ.get("RAC_JITTER_PER_CHANNEL", "0") == "1" and C == 3:
            channel_widths = np.array([1400.0, 400.0, 1700.0], dtype=np.float32)
            scales = hu_delta / channel_widths
        else:
            scales = np.full((C,), hu_delta / 1400.0, dtype=np.float32)
        shifts = np.random.uniform(-scales, scales, size=(C,)).astype(np.float32)
        shifts_t = torch.from_numpy(shifts).view(C, 1, 1, 1)
        return (ct_chwd + shifts_t).clamp_(0.0, 1.0)

    def _gaussian_noise(self, ct_chwd: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return ct_chwd
        noise = np.random.normal(0.0, sigma, size=ct_chwd.shape).astype(np.float32)
        return (ct_chwd + torch.from_numpy(noise)).clamp_(0.0, 1.0)

    def _maybe_joint_flip(self, ct_chwd: torch.Tensor, mask_hwd: torch.Tensor):
        if not self.enable_flips:
            return ct_chwd, mask_hwd
        if random.random() < 0.2:
            ct_chwd = torch.flip(ct_chwd, dims=(1,))
            mask_hwd = torch.flip(mask_hwd, dims=(0,))
        if random.random() < 0.2:
            ct_chwd = torch.flip(ct_chwd, dims=(2,))
            mask_hwd = torch.flip(mask_hwd, dims=(1,))
        return ct_chwd, mask_hwd

    def _hu_domain_sim(self, hu_hwd: np.ndarray) -> np.ndarray:
        """Simulate an external release's HU clipping (see hu_clip_sim_prob)."""
        if self.hu_clip_sim_prob <= 0.0 or random.random() >= self.hu_clip_sim_prob:
            return hu_hwd
        hi = float(np.random.uniform(*self.hu_clip_sim_hi))
        return np.clip(hu_hwd, -1000.0, hi)

    def __getitem__(self, idx):
        npz_path, accession, text, findings, label, mask_path = self.samples[idx]
        try:
            hu_hwd = self._hu_domain_sim(self._load_npz_hwd(npz_path))
            ct_chwd = torch.from_numpy(_to_3ch_hwd(hu_hwd, self.hu_window_jitter))
        except Exception:
            if self.fail_fast:
                raise
            ct_chwd = torch.zeros(3, *self.SPATIAL_HWD, dtype=torch.float32)

        mask_hwd = torch.zeros(self.SPATIAL_HWD, dtype=torch.uint8)
        has_mask = False
        if mask_path:
            m = _load_fine_mask(mask_path, self.SPATIAL_HWD, fail_fast=self.fail_fast)
            if m is not None and m.max() > 0:
                mask_hwd = torch.from_numpy(m.astype(np.uint8))
                has_mask = True

        # Mask-dropout: randomly hide an available mask during training so the
        # mask-free path is exercised (improves external/maskless generalization).
        if has_mask and self.mask_dropout > 0.0 and random.random() < self.mask_dropout:
            mask_hwd = torch.zeros(self.SPATIAL_HWD, dtype=torch.uint8)
            has_mask = False

        # Stage-2 augmentation pack (train-only; helpers no-op when disabled).
        ct_chwd, mask_hwd = self._random_crop_192_from_240(ct_chwd, mask_hwd)
        ct_chwd = self._intensity_jitter(ct_chwd, self.intensity_jitter_hu)
        ct_chwd = self._gaussian_noise(ct_chwd, self.gaussian_noise_sigma)
        ct_chwd, mask_hwd = self._maybe_joint_flip(ct_chwd, mask_hwd)
        ct = ct_chwd.permute(0, 3, 1, 2).contiguous()
        mask_fine = mask_hwd.permute(2, 0, 1).contiguous().unsqueeze(0)
        return (
            ct,
            text,
            findings,
            torch.tensor(label, dtype=torch.float32),
            mask_fine,
            torch.tensor(has_mask, dtype=torch.bool),
            accession,
        )


def rac_collate(batch):
    cts, texts, findings, labels, masks, has_masks, accessions = zip(*batch)
    return (
        torch.stack(cts),
        list(texts),
        list(findings),
        torch.stack(labels),
        torch.stack(masks),
        torch.stack(has_masks),
        list(accessions),
    )


class RadChestStage1Dataset(Dataset):
    """RAD-ChestCT volumes as extra external-domain supervision for Stage-1.

    Task-2's hidden test set is roughly half external, and that half is where we
    lose the most (0.88 in-domain vs 0.75 external macro-AUROC). The challenge
    rules allow any publicly available data, and RAD-ChestCT is public, so its
    imgtrain split is trained on directly. Volumes come from the cache written by
    ``scripts/build_radchest_cache.py`` (already resampled, cropped and oriented
    like the eval path), so loading reuses ``RACDatasetV4._load_npz_hwd``.

    RAD-ChestCT does not label every CT-RATE pathology (e.g. hiatal hernia is
    absent). Unlabelled cells must not be taught as negatives, so ``keep_matrix``
    exposes a per-volume mask for the loss, in the same {accession: keep-row}
    form ``RAC_LABEL_MASK`` uses.
    """

    SPATIAL_HWD = RACDatasetV4.SPATIAL_HWD
    PAD_HWD = RACDatasetV4.PAD_HWD

    def __init__(self, cache_dir: str, labels_csv: str, split_csv: str = "",
                 is_train: bool = True, fail_fast: bool = True, limit: int = 0):
        import pandas as pd

        self.cache_dir = cache_dir
        self.is_train = is_train
        self.fail_fast = fail_fast
        df = pd.read_csv(labels_csv)
        if split_csv:
            keep = {str(a) for a in pd.read_csv(split_csv, index_col=0).index}
            df = df[df["NoteAcc_DEID"].astype(str).isin(keep)]
        self.samples = []
        self.keep = {}
        for _, r in df.iterrows():
            acc = str(r["NoteAcc_DEID"])
            path = os.path.join(cache_dir, f"{acc}.npz")
            if not os.path.isfile(path):
                continue
            lab = np.array([0.0 if pd.isna(r[c]) else float(r[c]) for c in PATHOLOGIES],
                           dtype=np.float32)
            keep_row = np.array([0.0 if pd.isna(r[c]) else 1.0 for c in PATHOLOGIES],
                                dtype=np.float32)
            self.samples.append((path, acc, lab))
            self.keep[acc] = keep_row
            if limit and len(self.samples) >= limit:
                break
        dropped = 1.0 - float(np.mean([v.mean() for v in self.keep.values()])) if self.keep else 0.0
        print(f"[radchest-train] {len(self.samples)} volumes from {cache_dir} "
              f"({100*dropped:.1f}% of label cells unlabelled -> dropped from the loss)",
              flush=True)

        self.enable_random_crop = os.environ.get("RAC_RANDOM_CROP", "0") == "1" and is_train
        self.intensity_jitter_hu = float(os.environ.get("RAC_INTENSITY_JITTER_HU", "0")) if is_train else 0.0
        self.gaussian_noise_sigma = float(os.environ.get("RAC_GAUSSIAN_NOISE_SIGMA", "0")) if is_train else 0.0
        self.hu_window_jitter = float(os.environ.get("RAC_HU_WINDOW_JITTER", "0")) if is_train else 0.0
        self.enable_flips = os.environ.get("RAC_ENABLE_ANATOMY_FLIPS", "0") == "1" and is_train
        # Reuse RACDatasetV4's augmentation helpers verbatim so both corpora get
        # the identical pipeline; only the volume source differs.
        for fn in ("_random_crop_192_from_240", "_intensity_jitter", "_gaussian_noise",
                   "_maybe_joint_flip"):
            setattr(self, fn, getattr(RACDatasetV4, fn).__get__(self, RadChestStage1Dataset))

    def keep_matrix(self) -> dict:
        return self.keep

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, acc, label = self.samples[idx]
        try:
            hu_hwd = RACDatasetV4._load_npz_hwd(path)
            ct_chwd = torch.from_numpy(_to_3ch_hwd(hu_hwd, self.hu_window_jitter))
        except Exception:
            if self.fail_fast:
                raise
            ct_chwd = torch.zeros(3, *self.SPATIAL_HWD, dtype=torch.float32)
        mask_hwd = torch.zeros(self.SPATIAL_HWD, dtype=torch.uint8)
        ct_chwd, mask_hwd = self._random_crop_192_from_240(ct_chwd, mask_hwd)
        ct_chwd = self._intensity_jitter(ct_chwd, self.intensity_jitter_hu)
        ct_chwd = self._gaussian_noise(ct_chwd, self.gaussian_noise_sigma)
        ct_chwd, mask_hwd = self._maybe_joint_flip(ct_chwd, mask_hwd)
        ct = ct_chwd.permute(0, 3, 1, 2).contiguous()
        mask_fine = mask_hwd.permute(2, 0, 1).contiguous().unsqueeze(0)
        return (ct, "", "", torch.tensor(label, dtype=torch.float32), mask_fine,
                torch.tensor(False, dtype=torch.bool), acc)
