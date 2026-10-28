---
license: cc-by-nc-sa-4.0
tags:
  - medical-imaging
  - computed-tomography
  - multi-label-classification
  - report-generation
datasets:
  - ibrahimhamamci/CT-RATE
library_name: pytorch
---

# arcct-vlm3d: VLM3D 2026 CT challenge weights (team metu-qedi)

Checkpoints for the final VLM3D 2026 submissions in
[arc-ct/arcct-vlm3d](https://github.com/arc-ct/arcct-vlm3d): CT Abnormality
Classification (CRG 0.5578 · AUROC 0.8670 · F1 0.6120) and CT Report Generation
(CRG 0.5238 · F1 0.6049 · BLEU 0.2556).

| file | arch | training data | classification | report gen |
|---|---|---|---|---|
| `stage1_rc_r2p1d.pt` | r2plus1d_18 | CT-RATE + RAD-ChestCT imgtrain | ✓ (×3) | |
| `stage1_r2p1d.pt` | r2plus1d_18 | CT-RATE | ✓ | ✓ |
| `stage1_r2p1d_s1.pt` | r2plus1d_18 | CT-RATE | ✓ | ✓ |
| `stage1_rc_r3d.pt` | r3d_18 | CT-RATE + RAD-ChestCT imgtrain | ✓ | |
| `stage1_r3djit.pt` | r3d_18 | CT-RATE (HU-window jitter) | ✓ | ✓ |
| `stage1_mc3_s1.pt` | mc3_18 | CT-RATE | ✓ | |
| `stage1_mc3.pt` | mc3_18 | CT-RATE | | ✓ |
| `CTClip.best.pt` | ARC-CT (loaded at start-up) | CT-RATE | | ✓ |

Each `stage1_*.pt` is a dict with `image_encoder` (torchvision video ResNet,
Kinetics-400 initialised) and `classifier` (`nn.Linear(512, 18)`, sigmoid outputs in
the challenge label order). Usage, preprocessing and container configs: see the GitHub
README.

Non-commercial research use only (trained on CT-RATE, CC BY-NC-SA 4.0, and
RAD-ChestCT). Not a medical device.
