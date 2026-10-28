#!/usr/bin/env python
"""Score saved predictions with the VLM3D challenge's OWN metrics.

The platform ranks by mean rank over three metrics (`.forithmus/config.json`):
CRG (primary), macro AUROC, macro F1. Two of the three are computed from a HARD
0.5 binarisation of the probabilities we emit -- see the challenge eval container
the official evaluator (github.com/forithmus/VLM3D-Dockers, `ct_challenges/abnclass_evaluation/`):

    evaluation.py:106   main(thresh=0.5)
    json_to_csv.py:45   {lab: int(v >= thresh) ...}
    crg_score.py:66-69  r = (X-A)/(2A); U = (X-A)/2; s = r*TP - r*FN - FP; CRG = U/(2U-s)
    calc_scores.py:70   precision/recall/f1/accuracy from the BINARY csv, AUROC from probs

So the operating point of our emitted probabilities is a first-class modelling
decision, not a reporting detail. This script reproduces those formulas on any
saved `predictions.npz` (keys: pred, true, optional mask) so that every local A/B
is judged the way the leaderboard judges it.

Usage
-----
    python evaluation/score_challenge.py --npz predictions.npz
    python evaluation/score_challenge.py --npz <a.npz>,<b.npz> --cal quantile --k 1.5
    python evaluation/score_challenge.py --npz <npz> --sweep-k        # k sweep table
    python evaluation/score_challenge.py --npz <npz> --fit-absolute   # per-class thresholds

`--cal` applies `src/calibrate.py` to the probabilities BEFORE scoring, exactly
as the container will. AUROC is invariant to it (the map is strictly monotone);
the script asserts that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUB / "src"))

from calibrate import PATHOLOGY_PRIOR, calibrate_probs  # noqa: E402

# Challenge label order (abnclass_evaluation/crg_score.py:22-30).
LABEL_COLS = list(PATHOLOGY_PRIOR.keys())


def load_npz(path: str, readout: str = "auto"):
    """-> (probs (N,C), labels (N,C) with NaN where unlabelled).

    readout: 'auto' prefers the slab-scored Slice-12 probabilities when the file
    has them, 'global' forces the pooled-feature readout.
    """
    d = np.load(path, allow_pickle=True)
    if readout != "global" and "slice_probs" in d:
        key = "slice_probs"
    elif "pred" in d:
        key = "pred"
    else:
        key = "global_pred"
    p = np.asarray(d[key], dtype=float)
    y = np.asarray(d["true" if "true" in d else "labels"], dtype=float)
    if "mask" in d:                      # RadChest: NaN-labelled classes
        y = np.where(np.asarray(d["mask"]).astype(bool), y, np.nan)
    if p.ndim == 3:                      # slice_probs (N, T, C) -> slab-mean readout
        p = p.mean(axis=1)
    return p, y


def crg(y: np.ndarray, y_bin: np.ndarray) -> dict:
    """crg_score.py:55-69, restricted to labelled cells."""
    m = ~np.isnan(y)
    yt = np.where(m, y, 0)
    pb = y_bin & m
    TP = int(((pb == 1) & (yt == 1) & m).sum())
    FN = int(((pb == 0) & (yt == 1) & m).sum())
    FP = int(((pb == 1) & (yt == 0) & m).sum())
    X = int(m.sum())
    A = int(yt[m].sum())
    if A == 0:
        raise ValueError("no positive labels; CRG undefined")
    r = (X - A) / (2 * A)
    U = (X - A) / 2
    s = r * TP - r * FN - FP
    return {"TP": TP, "FN": FN, "FP": FP, "X": X, "A": A, "r": r, "U": U,
            "score_s": s, "CRG": U / (2 * U - s) if (2 * U - s) != 0 else 0.0}


def classification(y: np.ndarray, p: np.ndarray, y_bin: np.ndarray) -> dict:
    """calc_scores.py:69-95.

    The official scorer averages precision/recall/F1/accuracy over ALL 18 classes
    and only drops a class from the AUROC average when a single class is present
    (`except ValueError` -> None -> `np.nanmean`). We follow that, with one
    unavoidable deviation: our external proxy leaves some classes fully
    unlabelled (RadChest has no Hiatal hernia), and those cannot be scored at
    all, so they are excluded and counted in `n_classes_unlabelled`.
    """
    per, thresholded, aurocs, unlabelled = [], [], [], 0
    for i, col in enumerate(LABEL_COLS[: y.shape[1]]):
        m = ~np.isnan(y[:, i])
        if m.sum() == 0:
            per.append({"name": col, "auroc": None, "unlabelled": True})
            unlabelled += 1
            continue
        yt, yp = y[m, i].astype(int), y_bin[m, i].astype(int)
        rec = {"name": col,
               "precision": precision_score(yt, yp, zero_division=0),
               "recall": recall_score(yt, yp, zero_division=0),
               "f1": f1_score(yt, yp, zero_division=0),
               "accuracy": accuracy_score(yt, yp),
               "auroc": roc_auc_score(yt, p[m, i]) if len(np.unique(yt)) > 1 else None,
               "prevalence": float(yt.mean()),
               "pred_rate": float(yp.mean())}
        per.append(rec)
        thresholded.append(rec)
        if rec["auroc"] is not None:
            aurocs.append(rec["auroc"])
    macro = {k: float(np.mean([r[k] for r in thresholded]))
             for k in ("precision", "recall", "f1", "accuracy")}
    macro["auroc"] = float(np.mean(aurocs))
    return {"per_pathology": per, "macro": macro,
            "n_classes_scored": len(thresholded), "n_classes_unlabelled": unlabelled}


def score(p_raw: np.ndarray, y: np.ndarray, cal: str = "off", k: float = 1.5,
          thresh: float = 0.5) -> dict:
    p = calibrate_probs(p_raw, mode=cal, k=k) if cal != "off" else p_raw
    y_bin = p >= thresh
    out = classification(y, p, y_bin)
    out["crg"] = crg(y, y_bin)
    out["calibration"] = {"mode": cal, "k": k, "pos_rate": float(y_bin.mean())}
    if cal != "off":                      # monotone map must not move AUROC
        base = classification(y, p_raw, y_bin)["macro"]["auroc"]
        assert abs(base - out["macro"]["auroc"]) < 1e-9, "calibration changed AUROC!"
    return out


def fit_absolute(p: np.ndarray, y: np.ndarray) -> list:
    """Per-class F1-optimal absolute thresholds (for the `absolute` calib mode)."""
    ts = np.arange(0.02, 0.99, 0.01)
    out = []
    for i in range(p.shape[1]):
        m = ~np.isnan(y[:, i])
        if m.sum() == 0 or len(np.unique(y[m, i])) < 2:
            out.append(0.5)
            continue
        f1s = [f1_score(y[m, i].astype(int), (p[m, i] >= t).astype(int), zero_division=0)
               for t in ts]
        out.append(round(float(ts[int(np.argmax(f1s))]), 3))
    return out


def fmt(tag: str, r: dict) -> str:
    m, c = r["macro"], r["crg"]
    return (f"{tag:<34s} AUROC {m['auroc']:.4f}  CRG {c['CRG']:.4f}  F1 {m['f1']:.4f}  "
            f"P {m['precision']:.4f}  R {m['recall']:.4f}  Acc {m['accuracy']:.4f}  "
            f"posrate {r['calibration']['pos_rate']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="comma-separated predictions.npz paths (averaged if >1)")
    ap.add_argument("--cal", default="quantile", choices=["off", "quantile", "absolute"])
    ap.add_argument("--k", type=float, default=1.5)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--sweep-k", action="store_true")
    ap.add_argument("--fit-absolute", action="store_true")
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    paths = [q for q in args.npz.split(",") if q]
    ps, ys = zip(*[load_npz(q) for q in paths])
    if len({p.shape for p in ps}) != 1:
        raise SystemExit(f"shape mismatch across npz: {[p.shape for p in ps]}")
    p = np.mean(ps, axis=0)
    y = ys[0]
    tag = "+".join(Path(q).parts[-3] for q in paths)
    print(f"# {tag}  n={p.shape[0]} x {p.shape[1]}")

    if args.fit_absolute:
        thr = fit_absolute(p, y)
        print("per-class F1-optimal absolute thresholds:")
        print(json.dumps(dict(zip(LABEL_COLS, thr)), indent=2))
        return

    if args.sweep_k:
        print(fmt("raw (no calibration)", score(p, y, "off")))
        for k in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
            print(fmt(f"quantile k={k}", score(p, y, "quantile", k)))
        print(fmt("absolute (CT-RATE-fitted)", score(p, y, "absolute")))
        return

    res = score(p, y, args.cal, args.k, args.thresh)
    print(fmt(f"cal={args.cal} k={args.k}", res))
    if args.per_class:
        for r in res["per_pathology"]:
            if r.get("auroc") is None:
                print(f"  {r['name']:<36s} (not scored)")
                continue
            print(f"  {r['name']:<36s} AUROC {r['auroc']:.4f}  F1 {r['f1']:.4f}  "
                  f"prev {r['prevalence']:.3f}  pred {r['pred_rate']:.3f}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2))
        print("->", args.out_json)


if __name__ == "__main__":
    main()
