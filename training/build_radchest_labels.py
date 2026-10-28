"""Build CT-RATE-aligned 18-class labels for the FULL RAD-ChestCT cohort.

RAD-ChestCT has 3,630 preprocessed volumes (imgtrain 2286 + imgvalid 984 +
imgtest 360). This script maps its abnormality-by-location labels onto the 18
CT-RATE classes for all of them. Output: data/radchest_ctrate_aligned_full.csv,
the default RAC_EXTRA_LABELS of training/train_stage1.py.

The mapping follows the imgtest aligned-label CSV used in prior CT-RATE work: each
CT-RATE class = OR over all locations of the corresponding RAD-ChestCT
abnormality prefix(es). It reproduces the reference labels at 100% agreement on
all 360 non-NaN cells (validated below). RAD-ChestCT has no hiatal-hernia
finding, so that class stays NaN (excluded from the macro), as in the ARC-CT paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
D = Path(os.environ.get("RAC_RADCHEST_DIR", "data/rad_chest_ct/extracted"))

PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

# CT-RATE class -> RAD-ChestCT abnormality prefix(es); value = OR over all locations.
MAP = {
    "Medical material": ["catheter_or_port", "clip", "pacemaker_or_defib", "stent",
                          "hardware", "staple", "suture", "chest_tube", "gi_tube"],
    "Arterial wall calcification": ["atherosclerosis"],
    "Cardiomegaly": ["cardiomegaly"],
    "Pericardial effusion": ["pericardial_effusion"],
    "Coronary artery wall calcification": ["coronary_artery_disease"],
    "Hiatal hernia": [],  # not present in RAD-ChestCT -> NaN, excluded from macro
    "Lymphadenopathy": ["lymphadenopathy"],
    "Emphysema": ["emphysema"],
    "Atelectasis": ["atelectasis"],
    "Lung nodule": ["nodule"],
    "Lung opacity": ["opacity"],
    "Pulmonary fibrotic sequela": ["fibrosis"],
    "Pleural effusion": ["pleural_effusion"],
    "Mosaic attenuation pattern": ["air_trapping"],
    "Peribronchial thickening": ["bronchial_wall_thickening"],
    "Consolidation": ["consolidation"],
    "Bronchiectasis": ["bronchiectasis"],
    "Interlobular septal thickening": ["septal_thickening"],
}

SPLITS = ["imgtrain", "imgvalid", "imgtest"]


def _prefix_cols(cols, prefixes):
    want = set(prefixes)
    return [c for c in cols if c != "NoteAcc_DEID" and c.split("*")[0] in want]


def build_split(split: str) -> pd.DataFrame:
    nat = pd.read_csv(D / f"{split}_Abnormality_and_Location_Labels.csv")
    cols = list(nat.columns)
    out = pd.DataFrame({"NoteAcc_DEID": nat["NoteAcc_DEID"].values})
    for cls in PATHOLOGIES:
        prefixes = MAP[cls]
        if not prefixes:
            out[cls] = np.nan
            continue
        pc = _prefix_cols(cols, prefixes)
        if not pc:
            raise RuntimeError(f"[{split}] no native columns for {cls} prefixes={prefixes}")
        out[cls] = (nat[pc].values > 0.5).any(axis=1).astype(float)
    out["split"] = split
    return out


def validate_against_reference(built_imgtest: pd.DataFrame) -> None:
    ref_path = D / "imgtest_ctrate_aligned_labels.csv"
    if not ref_path.exists():
        print(f"  (no reference {ref_path.name}; skipping the imgtest consistency check)")
        return
    ref = pd.read_csv(ref_path).set_index("NoteAcc_DEID")
    got = built_imgtest.set_index("NoteAcc_DEID")
    ref = ref.loc[got.index]
    mism = 0
    for cls in PATHOLOGIES:
        if ref[cls].isna().all():
            continue
        m = ~ref[cls].isna().values
        if not np.array_equal(got[cls].values[m], ref[cls].values[m]):
            n = int((got[cls].values[m] != ref[cls].values[m]).sum())
            mism += n
            print(f"  MISMATCH {cls}: {n} cells")
    if mism == 0:
        print("  reference imgtest reproduced EXACTLY (0 mismatches on all non-NaN cells)")
    else:
        print(f"  WARNING: {mism} total mismatches vs reference")
        sys.exit(1)


def main():
    frames = [build_split(s) for s in SPLITS]
    validate_against_reference(next(f for f in frames if f["split"].iloc[0] == "imgtest"))
    full = pd.concat(frames, ignore_index=True)
    out_path = Path(__file__).resolve().parents[1] / "data" / "radchest_ctrate_aligned_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.drop(columns=["split"]).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(full)} scans)")
    print("positives per class (full cohort):")
    for cls in PATHOLOGIES:
        pos = int((full[cls] > 0.5).sum())
        nan = int(full[cls].isna().sum())
        print(f"  {cls:38s} pos={pos:5d}  nan={nan}")


if __name__ == "__main__":
    main()
