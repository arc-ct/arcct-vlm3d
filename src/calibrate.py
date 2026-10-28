"""Operating-point calibration for the VLM3D Task-2 submission.

Why this exists
---------------
The challenge's scorer binarises the probabilities WE emit at a hard 0.5 and
computes both non-AUROC ranked metrics from that binary matrix
(`abnclass_evaluation/json_to_csv.py:45`, `evaluation.py:106`). CRG weights a
false negative by `r = (X-A)/(2A)` and a false positive by 1, so its optimal
operating point is `p > prevalence` (~0.19), not 0.5.

Our raw scores are a 2-way pos/neg softmax at a fixed temperature (`eval.py:209`)
and are not calibrated. Measured on the shipped checkpoint: on CT-RATE valid we
fire on 13% of cells (true prevalence 19%), while on external RadChest we fire on
**75%** -- the score distribution shifts up bodily under domain shift. An absolute
threshold fitted in-domain is therefore wrong out-of-domain (optimum 0.29 vs 0.55).

Fix: pick each class's operating point as a QUANTILE of that class's scores over
the whole test batch (prior/rate matching), then apply a strictly monotone
piecewise-linear map that sends it to 0.5. Quantiles self-correct under a bodily
shift; the map preserves the per-class ranking, so macro AUROC is unchanged.

Measured (shipped checkpoint, Slice-12 readout, k=1.6, vs raw@0.5):
    CT-RATE valid : CRG 0.4231 -> 0.5737,  macro-F1 0.3778 -> 0.5992
    RadChest ext  : CRG 0.4126 -> 0.4644,  macro-F1 0.3049 -> 0.4791

k=1.6 is the flat optimum of (mean CRG + mean macro-F1) over both proxies; the
curve is shallow between 1.4 and 1.8, with larger k favouring the external half.
"""
from __future__ import annotations

import os

import numpy as np

# Per-class prevalence on the CT-RATE validation split
# (multi_abnormality_labels/valid_predicted_labels.csv, n=3002). Order is the
# challenge label order == dataset.PATHOLOGIES.
PATHOLOGY_PRIOR = {
    "Medical material": 0.1023,
    "Arterial wall calcification": 0.2845,
    "Cardiomegaly": 0.1053,
    "Pericardial effusion": 0.0723,
    "Coronary artery wall calcification": 0.2525,
    "Hiatal hernia": 0.1389,
    "Lymphadenopathy": 0.2575,
    "Emphysema": 0.1969,
    "Atelectasis": 0.2345,
    "Lung nodule": 0.4500,
    "Lung opacity": 0.3921,
    "Pulmonary fibrotic sequela": 0.2738,
    "Pleural effusion": 0.1233,
    "Mosaic attenuation pattern": 0.0819,
    "Peribronchial thickening": 0.1159,
    "Consolidation": 0.1925,
    "Bronchiectasis": 0.1089,
    "Interlobular septal thickening": 0.0819,
}

# Per-class F1-optimal thresholds fitted on CT-RATE valid with the shipped
# checkpoint (rcretrain_augmd_seed0). Only used by mode="absolute" and as the
# small-batch fallback, because they do not transfer under domain shift.
ABSOLUTE_THRESHOLDS = {
    "Medical material": 0.49,
    "Arterial wall calcification": 0.37,
    "Cardiomegaly": 0.39,
    "Pericardial effusion": 0.39,
    "Coronary artery wall calcification": 0.39,
    "Hiatal hernia": 0.37,
    "Lymphadenopathy": 0.25,
    "Emphysema": 0.27,
    "Atelectasis": 0.36,
    "Lung nodule": 0.36,
    "Lung opacity": 0.35,
    "Pulmonary fibrotic sequela": 0.12,
    "Pleural effusion": 0.41,
    "Mosaic attenuation pattern": 0.39,
    "Peribronchial thickening": 0.32,
    "Consolidation": 0.52,
    "Bronchiectasis": 0.26,
    "Interlobular septal thickening": 0.34,
}

PRIOR_VEC = np.array(list(PATHOLOGY_PRIOR.values()), dtype=float)
ABS_VEC = np.array(list(ABSOLUTE_THRESHOLDS.values()), dtype=float)

# Small-batch shrinkage toward the in-domain absolute thresholds. DISABLED by
# default (MIN_N=1): resampling the saved predictions shows the batch quantile is
# already stable at n=20, while the shrinkage drags the external score down badly
# (RadChest CRG at n=50: pure quantile 0.4343 vs blended 0.3749 vs raw 0.3917),
# because the absolute thresholds do not transfer. Raise RAC_CAL_MIN_N only if a
# future test set is known to be in-domain and tiny.
MIN_N = int(os.environ.get("RAC_CAL_MIN_N", "1"))
EPS = 1e-4


def parse_k(k, c: int) -> np.ndarray:
    """k may be a scalar or a per-class vector (list/array/comma-separated string).

    A single global k spends the firing budget across classes strictly in
    proportion to the CT-RATE prior, which is not where CRG and macro-F1 want it:
    the optimum per class also depends on how well that class is separated. A
    per-class k vector keeps the rule rate-based (so it still self-corrects under
    the bodily score shift that motivated quantile calibration) while letting the
    allocation differ. Measured on held-out halves of the mixed proxy, vs global
    k=1.75: CRG +0.0027, macro-F1 +0.0150.
    """
    if isinstance(k, str):
        k = [float(x) for x in k.replace(",", " ").split()]
    k = np.asarray(k, dtype=float).ravel()
    if k.size == 1:
        return np.full(c, float(k[0]))
    # Fail loud: a truncated or over-long vector would silently mis-threshold.
    if k.size != c:
        raise ValueError(f"per-class k has {k.size} entries, expected {c}")
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError(f"per-class k must be finite and positive, got {k!r}")
    return k


def thresholds(probs: np.ndarray, mode: str = "quantile", k=1.6,
               min_n: int = MIN_N) -> np.ndarray:
    """Per-class operating points for a whole test batch, shape (C,)."""
    n, c = probs.shape
    abs_t = ABS_VEC[:c]
    if mode == "absolute":
        return np.clip(abs_t, EPS, 1 - EPS)
    if mode != "quantile":
        raise ValueError(f"unknown calibration mode {mode!r}")

    target_rate = np.clip(PRIOR_VEC[:c] * parse_k(k, c), 0.01, 0.95)
    q = np.array([np.quantile(probs[:, i], 1.0 - target_rate[i]) for i in range(c)])
    if n < min_n:                       # small batch -> shrink toward in-domain fit
        w = n / float(min_n)
        q = w * q + (1.0 - w) * abs_t
    return np.clip(q, EPS, 1 - EPS)


def apply_thresholds(probs: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Strictly monotone piecewise-linear map sending thr[c] -> 0.5.

    Per-class ranking (hence per-class AUROC) is preserved exactly.
    """
    t = thr[None, :]
    lo = 0.5 * probs / t
    hi = 0.5 + 0.5 * (probs - t) / (1.0 - t)
    return np.clip(np.where(probs < t, lo, hi), 0.0, 1.0)


def calibrate_probs(probs: np.ndarray, mode: str = "quantile", k=1.6,
                    min_n: int = MIN_N) -> np.ndarray:
    """Calibrate a (N, C) probability matrix. mode='off' returns it unchanged."""
    probs = np.asarray(probs, dtype=float)
    if mode == "off":
        return probs
    if probs.ndim != 2:
        raise ValueError(f"expected (N, C), got {probs.shape}")
    return apply_thresholds(probs, thresholds(probs, mode=mode, k=k, min_n=min_n))


def calibrate_from_env(probs: np.ndarray) -> np.ndarray:
    """Container entry point: RAC_CAL_MODE (default quantile) + RAC_CAL_K (1.6)."""
    mode = os.environ.get("RAC_CAL_MODE", "quantile")
    # RAC_CAL_KVEC (18 comma-separated values) overrides the scalar RAC_CAL_K.
    kvec = os.environ.get("RAC_CAL_KVEC", "").strip()
    k = kvec if kvec else float(os.environ.get("RAC_CAL_K", "1.6"))
    out = calibrate_probs(probs, mode=mode, k=k)
    print(f"[calibrate] mode={mode} k={k} n={probs.shape[0]} "
          f"pos_rate {float((probs >= 0.5).mean()):.3f} -> {float((out >= 0.5).mean()):.3f}",
          flush=True)
    return out
