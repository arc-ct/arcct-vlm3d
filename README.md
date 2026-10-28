# arcct-vlm3d: our VLM3D 2026 challenge submissions (CT tasks)

This repository holds the code, configurations and weights for our submissions to the
[VLM3D Challenge 2026](https://research.forithmus.com) (team **metu-qedi**):

| Task | Final leaderboard entry | CRG | Other ranked metrics | Final rank |
|---|---|---|---|---|
| CT Abnormality Classification (VLM3D-CT-ABNORMALITY) | 3D video-ResNet ensemble + batch-quantile calibration | 0.5578 | AUROC 0.8670 · F1 0.6120 | 4th |
| CT Report Generation (VLM3D-CT-REPORT) | Label-ranked report retrieval + MBR consensus | 0.5238 | F1 0.6049 · BLEU 0.2556 | 5th |

Both entries are built on the same supervised **Stage-1** image classifiers. The
classification entry outputs their calibrated probabilities directly. The
report-generation entry uses them to choose and edit a real training report.

The vision-language model we started from, ARC-CT (TIA @ MICCAI 2026), has its own
repository: [arc-ct/arc-ct](https://github.com/arc-ct/arc-ct). Our earlier challenge
submissions used ARC-CT, and the report-generation container still loads its
checkpoint at start-up (see [Limitations](#limits-and-limitations)).

- [Team](#team)
- [Method](#method)
- [Data](#data)
- [Installation](#installation)
- [Usage: inference containers](#usage-inference-containers)
- [Training](#training)
- [Weights](#weights)
- [Submissions and configurations](#submissions-and-configurations)
- [Validation and results](#validation-and-results)
- [Limits and limitations](#limits-and-limitations)
- [References](#references)
- [License and terms of use](#license-and-terms-of-use)

## Team

| Member | Affiliation |
|---|---|
| Hüseyin Umut Işık | Department of Computer Engineering, Middle East Technical University (METU), Ankara, Türkiye |
| Mehmet Alp Özaydın | Department of Computer Engineering, Middle East Technical University (METU), Ankara, Türkiye |
| Sila Kurugol | Department of Computer Engineering, Middle East Technical University (METU), Ankara, Türkiye |
| Şeyda Ertekin | Department of Computer Engineering, Middle East Technical University (METU), Ankara, Türkiye |

## Method

### Stage-1 classifiers (shared by both tasks)

- **Backbones.** These are torchvision video ResNets (`r3d_18`, `mc3_18`, `r2plus1d_18`)
  initialised from **Kinetics-400** weights. The CT volume is fed as a 3-channel
  "video": each channel is the same volume under a different HU window (lung
  [-1500, 500], soft tissue [-160, 240], bone [300, 2000]), so the Kinetics RGB stem is
  reused unchanged. The features are globally pooled to 512-d, followed by
  `nn.Linear(512, 18)` and a sigmoid.
- **Input.** Volumes are read from NIfTI in Hounsfield units, resampled to
  1.5 × 1.5 × 3.0 mm and padded or cropped to 192 × 192 × 96 (H × W × D). Training
  takes a random 192² crop from a 240² pad.
- **Training.** We use asymmetric loss (γ⁻ = 4, γ⁺ = 1, clip 0.05), AdamW (lr 1e-4,
  weight decay 1e-4), a cosine schedule, batch 8, 6 epochs and AMP. The checkpoint with
  the best CT-RATE validation macro-AUROC is kept. Validation AUROC peaks at
  epoch 5–6 and falls with longer training. Augmentation: random crop,
  anatomy-consistent flips, ±30 HU intensity jitter, and (one member) random
  HU-window jitter of 0.15.
- **Diversity.** Members differ in architecture, seed, HU-window jitter, and whether
  RAD-ChestCT `imgtrain` is added to training (oversampled 3×; the RAD-ChestCT label
  cells that cannot be mapped to CT-RATE are masked out of the loss). Settings per
  member: [`configs/stage1_arms.env`](configs/stage1_arms.env).

### Calibration

The official scorer binarises our probabilities at 0.5, and CRG and F1 are computed
from that binary matrix, so the operating point matters as much as the ranking. Under
domain shift, raw scores move as a block: the raw positive rate was 13 % on CT-RATE
validation and 75 % on RAD-ChestCT. A fixed threshold therefore does not transfer.

[`src/calibrate.py`](src/calibrate.py) works per class over the **whole test batch**.
It places the operating point at the `1 − k_c · π_c` quantile of that class's scores,
where `π_c` is the CT-RATE validation prevalence. A strictly monotone piecewise-linear
map then sends that point to 0.5. Per-class ranking is unchanged, so **AUROC is exactly
preserved**.

### Classification entry (VLM3D-CT-ABNORMALITY)

This entry is a weighted probability mean of 6 Stage-1 checkpoints, expressed as
8 slots: `rc_r2p1d` ×3, plus `r2p1d`, `r2p1d_s1`, `rc_r3d`, `r3djit` and `mc3_s1` ×1
each. Calibration is applied after the mean, with a per-class `k_c` vector (≈1.36–1.96,
mean ≈1.7). Membership and weights were chosen by exhaustive subset search over all
trained arms on CT-RATE validation. Rank averaging and flip test-time augmentation
did not help.

### Report-generation entry (VLM3D-CT-REPORT): label-ranked retrieval

The official report evaluator runs a RadBERT classifier over each generated report to
extract the 18 labels, then scores CRG and F1 on them, and also scores BLEU against the
reference. CT-RATE's released `train_predicted_labels.csv` holds that same labeller's
output for every training report. So choosing a training report whose label row
matches our image prediction largely determines what the evaluator will extract from
it.

1. **Predict.** A 4-member Stage-1 ensemble (`r2p1d`, `r2p1d_s1`, `r3djit`, `mc3`)
   produces probabilities, calibrated over the batch with k = 1.25.
2. **Rank.** Each of the 46,397 training reports is scored by
   `Σ_c (p_c − 0.5) · y_c`, where `y_c` is the report's released label row. Since each
   class's operating point is already at 0.5, this is the expected CRG gain of
   asserting that label row.
3. **MBR tie-break.** Many reports share the same label row. Among candidates within
   0.10 of the best score (at most 60), we keep the report with the highest mean
   pairwise BLEU against the other candidates. This is the consensus report, and it
   raises BLEU without changing the label row.
4. **Edit.** We drop sentences that assert a finding we predict absent, and prepend a
   canonical sentence ([`report_generation/sentence_bank.json`](report_generation/sentence_bank.json))
   for any predicted finding the report does not mention. Positives go first because
   the evaluator truncates at 512 tokens.

Outputs are real radiology prose of the expected length, so n-gram metrics stay
competitive with generative models while the clinical metrics follow the classifier.

## Data

| Dataset | Use | Access |
|---|---|---|
| [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) (challenge-provided) | training (`train`, 47k volumes), validation and model selection (`valid`, 3,002 volumes with labels), report gallery (`train_reports.csv`, `train_predicted_labels.csv`) | Hugging Face, gated (accept the terms) |
| [RAD-ChestCT](https://zenodo.org/records/6406114) (public) | extra training data, **`imgtrain` only** (2,286 volumes), for two classification members; `imgvalid` + `imgtest` (1,344) held out as the external validation set | Zenodo |

No private or synthetic data were used. The labels are the ones released with each
dataset: CT-RATE's report-derived labels, and RAD-ChestCT's abnormality-by-location
labels mapped onto the 18 CT-RATE classes by
[`training/build_radchest_labels.py`](training/build_radchest_labels.py).

**Preprocessing for training.** Use `tools/prepare_data.py` from
[arc-ct/arc-ct](https://github.com/arc-ct/arc-ct#data) to convert CT-RATE NIfTI files
to npz (HU/1000, 1.5 × 1.5 × 3.0 mm). Stage-1 expects this layout under
`RAC_DATA_ROOT`:

```
mps_ct_npz/{train,valid}/*.npz
multi_abnormality_labels/{train,valid}_predicted_labels.csv
ct_reports/{train,valid}_reports.csv
fine_{train,valid}_masks_192/        # optional; the classifier does not use masks
```

RAD-ChestCT: `python training/build_radchest_labels.py` writes
`data/radchest_ctrate_aligned_full.csv`. Then
`python training/build_radchest_cache.py` (with `RAC_RADCHEST_NPZ` and
`RAC_RADCHEST_CACHE` set) writes the `imgtrain` volumes in the CT-RATE npz format.

**Inference needs no preprocessing.** The containers read raw `.nii.gz` and take HU and
spacing from the NIfTI header.

## Installation

Inference runs in Docker (CUDA 12.1, Python 3.10, torch 2.4.1). For training or local
evaluation outside Docker:

```bash
git clone https://github.com/arc-ct/arcct-vlm3d && cd arcct-vlm3d
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c classification/constraints.txt
```

## Usage: inference containers

Both containers follow the VLM3D platform contract. They read `/input/*.nii.gz`, read
weights from `/weights` (uploaded separately as `weights.zip` with files at the zip
root) and write JSON to `/output`. They run fully offline.

```bash
# build (from the repository root)
DOCKER_BUILDKIT=1 docker build -f classification/Dockerfile     -t arcct-vlm3d-cls .
DOCKER_BUILDKIT=1 docker build -f report_generation/Dockerfile  -t arcct-vlm3d-rg  .

# run
docker run --rm --gpus all --network=none \
  -v /path/to/volumes:/input:ro -v /path/to/weights_cls:/weights:ro -v /path/to/out:/output \
  arcct-vlm3d-cls
```

**Input.** One chest CT per file, `<case_id>.nii.gz`, with HU intensities and voxel
spacing in the header (the platform's format). The raw Hugging Face CT-RATE files have
empty headers. For local tests on them, mount CT-RATE's metadata CSV and set
`RAC_META_CSV=/path/to/validation_metadata.csv`.

**Classification output.** `/output/results.json` (and an identical
`predictions.json`):

```json
[{"outputs": [{"value": {
  "name": "Generated probabilities", "type": "Abnormality Classification",
  "version": {"major": 1, "minor": 0},
  "predictions": [
    {"input_image_name": "<case_id>",
     "probabilities": {"Medical material": 0.12, "Arterial wall calcification": 0.81, "...": 0.0}}
  ]}}]}]
```

`probabilities` holds the 18 calibrated scores in the challenge label order.
Values ≥ 0.5 mean "present" at our chosen operating point.

**Report-generation output.** `/output/predictions.json`:

```json
[{"outputs": [{"value": {"generated_reports": [
  {"input_image_name": "<case_id>", "report": "Findings: ... Impression: ..."}
]}}]}]
```

`<case_id>` is the file name without `.nii.gz`. For platform names of the form
`{batch}_{patient}_{series}`, it is the trailing UUID segment, matching the official
evaluator.

**Hardware.** One GPU with ≥ 16 GB (scored on an A100 40 GB). The classification run
takes a few seconds per volume. Report generation spends about 5 extra minutes
loading the gallery. Calibration is computed over the whole batch, so run each test
set in a single call (see [Limitations](#limits-and-limitations)).

## Training

```bash
# one ensemble member; settings for every shipped member are in configs/stage1_arms.env
ARM=r2p1d RAC_R3D_ARCH=r2plus1d_18 RAC_SEED=0 bash training/train_arm.sh

# a RAD-ChestCT-augmented member
ARM=rc_r2p1d RAC_R3D_ARCH=r2plus1d_18 RAC_SEED=0 \
RAC_EXTRA_TRAIN=data/radchest_cache_192 RAC_EXTRA_REPEAT=3 bash training/train_arm.sh
```

Each member trains on one A100 80 GB in about 6–12 h. The best checkpoint is written
to `runs/stage1_<ARM>/stage1_best.pt`. Rename it to `stage1_<ARM>.pt` to use it with the
container configs.

**Local scoring.** [`evaluation/score_challenge.py`](evaluation/score_challenge.py)
re-implements the official CRG / macro-AUROC / macro-F1 at the 0.5 threshold on a saved
`predictions.npz` (`pred`, `true`). With `--cal quantile --k <k>` it first applies the
same calibration as the container.

## Weights

All checkpoints are on the Hugging Face Hub at
[`arc-ct/arcct-vlm3d`](https://huggingface.co/arc-ct/arcct-vlm3d):

| File | Arch | Training data | Classification | Report gen |
|---|---|---|---|---|
| `stage1_rc_r2p1d.pt` | r2plus1d_18 | CT-RATE + RAD-ChestCT imgtrain | ✓ (×3) | |
| `stage1_r2p1d.pt` | r2plus1d_18 | CT-RATE | ✓ | ✓ |
| `stage1_r2p1d_s1.pt` | r2plus1d_18 | CT-RATE | ✓ | ✓ |
| `stage1_rc_r3d.pt` | r3d_18 | CT-RATE + RAD-ChestCT imgtrain | ✓ | |
| `stage1_r3djit.pt` | r3d_18 | CT-RATE (HU-window jitter) | ✓ | ✓ |
| `stage1_mc3_s1.pt` | mc3_18 | CT-RATE | ✓ | |
| `stage1_mc3.pt` | mc3_18 | CT-RATE | | ✓ |
| `CTClip.best.pt` | ARC-CT (loaded at start-up only) | CT-RATE | | ✓ |

Building the `/weights` folders exactly as they were uploaded:

```bash
hf download arc-ct/arcct-vlm3d --local-dir ckpt
# classification
mkdir -p weights_cls && cp ckpt/stage1_{rc_r2p1d,r2p1d,r2p1d_s1,rc_r3d,r3djit,mc3_s1}.pt weights_cls/
# report generation: + two unmodified CT-RATE files (gated; not redistributed here)
mkdir -p weights_rg && cp ckpt/stage1_{r2p1d,r2p1d_s1,r3djit,mc3}.pt ckpt/CTClip.best.pt weights_rg/
cp <CT-RATE>/dataset/radiology_text_reports/train_reports.csv            weights_rg/
cp <CT-RATE>/dataset/multi_abnormality_labels/train_predicted_labels.csv weights_rg/
(cd weights_cls && zip ../weights_cls.zip *) ; (cd weights_rg && zip ../weights_rg.zip *)
```

## Submissions and configurations

The exact configuration of each final entry is the `ENV` block of its Dockerfile.
These were checked against the environment of the scored images.

| | Classification | Report generation |
|---|---|---|
| Leaderboard entry | 2026-08-17, "Stage-1 8-member ensemble with per-class quantile calibration" | `fb3b4a42-8269-4bf7-aa21-1f9fbf363699` (2026-08-14) |
| Container | [`classification/Dockerfile`](classification/Dockerfile) | [`report_generation/Dockerfile`](report_generation/Dockerfile) (`RG_MODE=hybrid`) |
| Members | `STAGE1_MODELS` (8 slots, 6 files) | `STAGE1_MODELS` (4 files) |
| Calibration | `RAC_CAL_KVEC` per class (CT-RATE validation priors) | `RAC_CAL_K=1.25` |
| Selection | none | `RG_MBR_SLACK=0.10`, `RG_MBR_CAP=60`, edit step on |
| Weights | `weights_cls.zip` (6 files, 0.7 GB) | `weights_rg.zip` (7 files, 1.8 GB) |

Running each container from this repository on three CT-RATE validation volumes
reproduces the outputs of the scored images (classification probabilities agree to
1e-7).

### Submission history

Earlier entries, and the change each one tested. Metrics are leaderboard values, where
the platform reported them.

**Classification** (CRG / AUROC / F1)

| Date | Change | Result |
|---|---|---|
| 07-28 | ARC-CT vision-language model, slice-wise prompt readout, batch-quantile calibration | 0.5348 / 0.8479 / 0.5909 |
| 08-12 | Single Stage-1 `r3d_18` classifier instead of the VLM (no text path) | ≈0.540 / 0.852 / 0.600 |
| 08-13 | 3-member Stage-1 ensemble (r3d, r2plus1d, mc3) | 0.5539 / 0.8612 / 0.6074 |
| 08-14 – 08-15 | 4–5 member ensembles, first RAD-ChestCT-trained members | CRG 0.5544–0.5566 |
| **08-17** | **Final 8-slot ensemble + per-class calibration** | **0.5578 / 0.8670 / 0.6120** |
| 08-27 | Same ensemble, more aggressive per-class k aimed at F1 | CRG 0.5326 (not kept) |

**Report generation** (CRG / F1 / BLEU)

| Date | Change | Result |
|---|---|---|
| 08-07 | Top-1 image–text cosine retrieval with ARC-CT | 0.4381 / 0.4521 / 0.2048 |
| 08-13 | Label-ranked retrieval driven by ARC-CT probabilities | 0.4985 / 0.5734 / 0.2364 |
| **08-14** | **Stage-1 ensemble probabilities + MBR consensus + edit step** | **0.5238 / 0.6049 / 0.2556** |

We also trained a generative arm: ARC-CT Q-Former tokens → linear bridge → frozen
Qwen2.5-3B-Instruct. It scored below retrieval on validation on every metric (CRG
0.3822, F1 0.2466, BLEU 0.2867) and is not part of the final entry.

## Validation and results

**Splits.** Model selection used the official CT-RATE validation split (n = 3,002).
RAD-ChestCT `imgvalid` + `imgtest` (n = 1,344, never trained on) served as the external
check. Every candidate was scored with the official metric formulas
([`evaluation/score_challenge.py`](evaluation/score_challenge.py)). For report
generation we ran a local copy of the official evaluator, including its RadBERT
labeller.

**Classification, final ensemble**

| | CRG | macro-AUROC | macro-F1 |
|---|---|---|---|
| CT-RATE validation (proxy) | 0.5862 | 0.8890 | 0.6132 |
| Leaderboard test | 0.5578 | 0.8670 | 0.6120 |

The ensemble was the best of all 2,036 subsets of the 11 trained arms on validation.
Single members reach validation AUROC between 0.8737 (`mc3`) and 0.8863 (`rc_r2p1d`).
On held-out RAD-ChestCT, the CT-RATE-only members reach 0.739–0.762 macro-AUROC.
Calibration alone moved an earlier model from CRG 0.4765 → 0.5567 on CT-RATE
validation, at unchanged AUROC.

**Report generation, label-ranked retrieval on CT-RATE validation (official labeller)**

| Selection | CRG | F1 | BLEU |
|---|---|---|---|
| Random gallery report | 0.3514 | 0.1945 | — |
| Image–text cosine top-1 | 0.4508 | 0.4612 | 0.269 |
| Label-ranked (single model) | 0.5426 | 0.6176 | 0.335 |
| Upper bound for that model: emit its predicted labels directly | 0.5481 | 0.6240 | — |
| Final configuration (Stage-1 ensemble + MBR + edit) | 0.5641 | 0.6287 | 0.357 |

Leaderboard (final): CRG 0.5238, F1 0.6049, BLEU 0.2556 (ROUGE-L 0.2895, METEOR
0.2457, accuracy 0.8313).

## Limits and limitations

- **Domain shift.** External performance is clearly lower: macro-AUROC is about 0.75
  on held-out RAD-ChestCT versus about 0.89 on CT-RATE. Adding RAD-ChestCT training
  data helps the external site but slightly lowers in-domain AUROC (0.8890 → 0.8862 when
  more external data was added). Validation-to-test drops were about −0.03 CRG and
  −0.02 AUROC.
- **Batch-dependent calibration.** Thresholds are quantiles over the submitted batch.
  This assumes a reasonably large batch (we measured it stable down to about 20 cases)
  whose class prevalence is close to CT-RATE's. Single-case inference, very small
  batches, or strongly skewed case mixes (e.g. a screening cohort) will be thresholded
  poorly. AUROC is unaffected.
- **Operating-point transfer.** Shifting thresholds toward more positives cost about
  twice as much CRG on the test set as on validation (0.5578 → 0.5326). Validation is
  optimistic for any "predict more" change.
- **Report generation retrieves rather than writes.** Every report comes from another
  patient, edited only at the sentence level. Measurements, sizes, laterality and
  findings outside the 18 classes can be wrong. The method optimises the evaluator's
  RadBERT label space, so its clinical scores depend on that labeller's behaviour.
  Text quality beyond the 18 labels is not optimised.
- **Report-generation start-up.** The container builds the ARC-CT model at start-up
  (`CTClip.best.pt` + CXR-BERT), although the final mode does not use it for
  prediction. This costs memory and start-up time, not accuracy.
- **Labels.** Training labels are automatically extracted from reports (CT-RATE) or
  mapped from another label schema (RAD-ChestCT; hiatal hernia is unlabelled there),
  so label noise transfers into the models.
- **Not tested on:** non-chest CT, paediatric patients, contrast-phase-specific
  protocols, scanners or institutions outside CT-RATE (Istanbul) and RAD-ChestCT
  (Duke), and non-English reports. No radiologist review of the outputs was carried
  out. **This is research code, not a medical device.**

## References

- Hamamci et al. *Developing Generalist Foundation Models from a Multimodal Dataset for
  3D Computed Tomography* (CT-RATE / CT-CLIP), 2024.
- Draelos et al. *Machine-learning-based multiple abnormality prediction with
  large-scale chest computed tomography volumes* (RAD-ChestCT), Medical Image
  Analysis, 2021.
- Tran et al. *A Closer Look at Spatiotemporal Convolutions for Action Recognition*
  (R3D, MC3, R(2+1)D), CVPR 2018.
- Ridnik et al. *Asymmetric Loss for Multi-Label Classification*, ICCV 2021.
- Kumar & Byrne. *Minimum Bayes-Risk Decoding for Statistical Machine Translation*,
  NAACL 2004.
- Yan et al. *RadBERT: Adapting Transformer-based Language Models to Radiology*, 2022
  (evaluator labeller).
- Işık, Özaydın, Kurugol, Ertekin. *ARC-CT: Anatomy-Routed Contrastive Vision-Language
  Learning for 3D Chest CT*, TIA @ MICCAI 2026.
  [Code](https://github.com/arc-ct/arc-ct) · [Paper](https://openreview.net/forum?id=t6mhjZ4R70)

```bibtex
@inproceedings{isik2026arcct,
  title     = {{ARC}-{CT}: Anatomy-Routed Contrastive Vision-Language Learning for 3D Chest {CT}},
  author    = {Huseyin Umut Isik and Mehmet Alp Ozaydin and Sila Kurugol and Seyda Ertekin},
  booktitle = {The Third International Workshop on Thoracic Image Analysis at MICCAI 2026},
  year      = {2026},
  url       = {https://openreview.net/forum?id=t6mhjZ4R70}
}
```

## License and terms of use

Code: MIT ([`LICENSE`](LICENSE)). The weights were trained on CT-RATE (CC BY-NC-SA 4.0)
and RAD-ChestCT and are released for **non-commercial research use**, in line with
those datasets' terms. CT-RATE and RAD-ChestCT data must be obtained from their
original sources under their own licenses; this repository does not redistribute them.
The software is not intended for clinical use.
