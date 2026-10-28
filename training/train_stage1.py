"""Stage-1 supervised 3-channel vision pretraining for RAC-CLIP.

This trains the R3D-18 image encoder on the 18 CT-RATE abnormality labels and
saves weights that ``train.py`` can load through ``RAC_STAGE1_CKPT``.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", f"{HERE}/.cache/torch")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from dataset import PATHOLOGIES, N_PATH, RACDatasetV4, RadChestStage1Dataset, rac_collate  # noqa: E402
from model.image_encoder import RACImageEncoder  # noqa: E402


DATA_ROOT = os.environ.get("RAC_DATA_ROOT", "data")
DATA_TRAIN = f"{DATA_ROOT}/mps_ct_npz/train"
DATA_VALID = f"{DATA_ROOT}/mps_ct_npz/valid"
MASK_TRAIN = os.environ.get("RAC_MASK_TRAIN", f"{DATA_ROOT}/fine_train_masks_192")
MASK_VALID = os.environ.get("RAC_MASK_VALID", f"{DATA_ROOT}/fine_valid_masks_192")
REPORTS_TRAIN = f"{DATA_ROOT}/ct_reports/train_reports.csv"
REPORTS_VALID = f"{DATA_ROOT}/ct_reports/valid_reports.csv"
LABELS_TRAIN = os.environ.get("RAC_LABELS_TRAIN", f"{DATA_ROOT}/multi_abnormality_labels/train_predicted_labels.csv")
LABELS_VALID = os.environ.get("RAC_LABELS_VALID", f"{DATA_ROOT}/multi_abnormality_labels/valid_predicted_labels.csv")
RESULTS_DIR = os.environ.get("RAC_STAGE1_RESULTS_DIR", f"{HERE}/runs/stage1_seed{os.environ.get('RAC_SEED', '0')}")

EPOCHS = int(os.environ.get("RAC_STAGE1_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("RAC_STAGE1_BATCH_SIZE", "8"))
LR = float(os.environ.get("RAC_STAGE1_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("RAC_STAGE1_WEIGHT_DECAY", "1e-4"))
NUM_WORKERS = int(os.environ.get("RAC_NUM_WORKERS", "6"))
TRAIN_LIMIT = int(os.environ.get("RAC_TRAIN_LIMIT", "0"))
VAL_LIMIT = int(os.environ.get("RAC_VAL_LIMIT", "0"))
FAIL_FAST = os.environ.get("RAC_FAIL_FAST", "1") == "1"
USE_AMP = os.environ.get("RAC_AMP", "1") == "1"
KINETICS_PRETRAINED = os.environ.get("RAC_KINETICS_PRETRAINED", "1") == "1"
MAX_GRAD_NORM = float(os.environ.get("RAC_MAX_GRAD_NORM", "1.0"))
LOSS_NAME = os.environ.get("RAC_STAGE1_LOSS", "asl").lower()
# Partial-label masking. CT-RATE labels are binary, so a pathology the report
# explicitly says could NOT be assessed is still recorded as a negative -- e.g.
# non-contrast CT cannot assess mediastinal nodes, which makes 35.6% of
# Lymphadenopathy negatives unreliable. RAC_LABEL_MASK points at an npz of
# {names, keep}; masked cells are dropped from the loss instead of being taught
# as negatives. Only NEGATIVES are dropped: an explicit positive is trustworthy.
# Default empty -> identical behaviour to every previous run.
LABEL_MASK = os.environ.get("RAC_LABEL_MASK", "").strip()

# Extra external-domain training corpus. RAC_EXTRA_TRAIN points at the
# RAD-ChestCT cache built by training/build_radchest_cache.py; the challenge rules
# allow any publicly available data, and the hidden test set is roughly half
# external, which is the half we lose on. RAC_EXTRA_REPEAT oversamples it, since
# 2,286 external volumes against 46,394 CT-RATE ones is only 5% of batches.
EXTRA_TRAIN = os.environ.get("RAC_EXTRA_TRAIN", "").strip()
EXTRA_LABELS = os.environ.get("RAC_EXTRA_LABELS", "data/radchest_ctrate_aligned_full.csv").strip()
EXTRA_SPLIT_CSV = os.environ.get("RAC_EXTRA_SPLIT_CSV", "").strip()
EXTRA_REPEAT = int(os.environ.get("RAC_EXTRA_REPEAT", "1"))
EXTRA_LIMIT = int(os.environ.get("RAC_EXTRA_LIMIT", "0"))

_seed_env = os.environ.get("RAC_SEED", "").strip()
if _seed_env:
    SEED = int(_seed_env)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    _GEN = torch.Generator()
    _GEN.manual_seed(SEED)

    def _worker(wid):
        ws = (SEED + wid) & 0xFFFFFFFF
        np.random.seed(ws)
        random.seed(ws)
else:
    _GEN = None
    _worker = None


def _loader_kwargs():
    return {"generator": _GEN, "worker_init_fn": _worker} if _GEN else {}


def _amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return torch.bfloat16 if bf16_ok else torch.float16


def amp_context(device):
    if USE_AMP and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=_amp_dtype())
    return nullcontext()


# ---------------------------------------------------------------------------
# Clinical metadata fusion (RAC_META_FUSION=1, default off).
#
# Every arm so far has seen only voxels, so ensembling them re-arranges the same
# information and has stalled at +0.0005 AUROC. Age and sex are complete in the
# CT-RATE metadata (100%) and the indication line carries real content in 49% of
# studies ("Chronic renal failure", "shortness of breath", "covid"). These are
# priors the image cannot supply -- emphysema and coronary calcification rise
# sharply with age, and the indication often names the finding being looked for.
#
# Metadata is looked up by accession in the training loop, exactly like
# RAC_LABEL_MASK, so dataset.py stays untouched.
META_FUSION = os.environ.get("RAC_META_FUSION", "0") == "1"
META_DROPOUT = float(os.environ.get("RAC_META_DROPOUT", "0.5"))
META_DIM = 0
_META_LUT = {}
if META_FUSION:
    import csv as _csv, json as _json, re as _re
    _vocab = _json.load(open(os.environ.get(
        "RAC_META_VOCAB", "data/indication_vocab.json")))
    _vidx = {t: i for i, t in enumerate(_vocab)}
    META_DIM = 3 + len(_vocab)          # age, sex, has_indication, bag-of-words
    for _split in ("train", "valid"):
        _f = os.environ.get(f"RAC_META_{_split.upper()}",
                            f"data/meta_{_split}.csv")
        if not os.path.isfile(_f):
            continue
        with open(_f) as _fh:
            for _r in _csv.DictReader(_fh):
                _v = np.zeros(META_DIM, dtype=np.float32)
                try:
                    _v[0] = float(_r["age"]) / 100.0        # ~[0.18, 1.02]
                except (TypeError, ValueError):
                    _v[0] = 0.49                            # cohort mean when absent
                try:
                    _v[1] = float(_r["sex"])
                except (TypeError, ValueError):
                    _v[1] = 0.5
                _ind = (_r.get("indication") or "").lower()
                _v[2] = 1.0 if _ind.strip() else 0.0
                for _t in _re.findall(r"[a-z]{3,}", _ind):
                    _j = _vidx.get(_t)
                    if _j is not None:
                        _v[3 + _j] = 1.0
                _META_LUT[_r["VolumeName"]] = _v
    print(f"[stage1] metadata fusion ON: {len(_META_LUT):,} volumes, dim={META_DIM}", flush=True)


def meta_batch(accessions, device):
    """Look up the metadata vector for each accession; zeros when unknown."""
    z = np.zeros(META_DIM, dtype=np.float32)
    return torch.from_numpy(np.stack([_META_LUT.get(a, z) for a in accessions])).to(device)


class Stage1Model(nn.Module):
    def __init__(self):
        super().__init__()
        # RAC_ENCODER: 'r3d' (default, unchanged) or 'vjepa'. Lazy import keeps
        # the default path free of any transformers>=4.52 dependency.
        if os.environ.get("RAC_ENCODER", "r3d").lower() in ("r3d", "r3d18", ""):
            self.image_encoder = RACImageEncoder(pretrained_kinetics=KINETICS_PRETRAINED)
        else:
            from model.image_encoder_vjepa import build_image_encoder
            self.image_encoder = build_image_encoder(pretrained_kinetics=KINETICS_PRETRAINED)
        # Self-supervised (SimMIM-3D) init overrides Kinetics: better cross-scanner
        # transfer, in-domain recovered by this supervised stage.
        _ssl = os.environ.get("RAC_SSL_INIT", "").strip()
        if _ssl:
            sd = torch.load(_ssl, map_location="cpu")
            miss, unexp = self.image_encoder.load_state_dict(sd, strict=False)
            print(f"[stage1] loaded SSL init {_ssl}  missing={len(miss)} unexpected={len(unexp)}", flush=True)
        if META_FUSION:
            # small MLP on the metadata vector, concatenated with the pooled image
            # feature before the linear head. Kept narrow (64) so the image feature
            # still dominates and the head cannot simply memorise demographics.
            self.meta_mlp = nn.Sequential(
                nn.Linear(META_DIM, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(inplace=True))
            self.classifier = nn.Linear(RACImageEncoder.FEAT_DIM + 64, N_PATH)
        else:
            self.classifier = nn.Linear(RACImageEncoder.FEAT_DIM, N_PATH)

    def forward(self, x, meta=None):
        feat = self.image_encoder(x)
        if META_FUSION:
            if meta is None:
                meta = torch.zeros(feat.shape[0], META_DIM, device=feat.device)
            elif self.training and META_DROPOUT > 0:
                # The challenge feeds the container volumes only -- no age, sex or
                # indication at inference. Training with metadata always present
                # would therefore guarantee a train/serve mismatch. Dropping the
                # whole vector for a fraction of samples forces the image path to
                # stand alone, so the deployed model is valid with zeros while the
                # metadata still acts as a training-time auxiliary signal.
                keep = (torch.rand(meta.shape[0], 1, device=meta.device) >= META_DROPOUT).float()
                meta = meta * keep
            feat = torch.cat([feat, self.meta_mlp(meta)], dim=1)
        return self.classifier(feat)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, weight=None):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pos = probs
        neg = 1.0 - probs
        if self.clip and self.clip > 0:
            neg = (neg + self.clip).clamp(max=1.0)
        loss_pos = targets * torch.log(pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(neg.clamp(min=self.eps))
        pt = pos * targets + neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = (loss_pos + loss_neg) * torch.pow(1.0 - pt, gamma)
        if weight is not None:
            return -(loss * weight).sum() / weight.sum().clamp(min=1.0)
        return -loss.mean()


def _aucs(pred, true):
    vals = []
    per = {}
    for j, name in enumerate(PATHOLOGIES):
        if len(np.unique(true[:, j])) > 1:
            v = float(roc_auc_score(true[:, j], pred[:, j]))
            vals.append(v)
            per[name] = v
        else:
            per[name] = float("nan")
    return float(np.nanmean(vals)) if vals else 0.0, per


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for ct, _, _, labels, _, _, accessions in tqdm.tqdm(loader, desc="Stage1-val", leave=False):
        ct = ct.to(device, non_blocking=True)
        with amp_context(device):
            logits = model(ct, meta_batch(accessions, ct.device) if META_FUSION else None)
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(labels.numpy())
    model.train()
    return _aucs(np.concatenate(preds), np.concatenate(trues))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage1] device={device} results_dir={RESULTS_DIR}")
    print(f"[stage1] epochs={EPOCHS} batch={BATCH_SIZE} lr={LR} loss={LOSS_NAME}")

    train_ds = RACDatasetV4(
        DATA_TRAIN,
        REPORTS_TRAIN,
        LABELS_TRAIN,
        mask_root=MASK_TRAIN if os.path.isdir(MASK_TRAIN) else None,
        is_train=True,
        limit=TRAIN_LIMIT,
        fail_fast=FAIL_FAST,
    )
    val_ds = RACDatasetV4(
        DATA_VALID,
        REPORTS_VALID,
        LABELS_VALID,
        mask_root=MASK_VALID if os.path.isdir(MASK_VALID) else None,
        is_train=False,
        limit=VAL_LIMIT,
        fail_fast=FAIL_FAST,
    )
    extra_keep = None
    if EXTRA_TRAIN:
        extra_ds = RadChestStage1Dataset(EXTRA_TRAIN, EXTRA_LABELS, EXTRA_SPLIT_CSV,
                                         is_train=True, fail_fast=FAIL_FAST, limit=EXTRA_LIMIT)
        extra_keep = extra_ds.keep_matrix()
        train_ds = ConcatDataset([train_ds] + [extra_ds] * EXTRA_REPEAT)
        print(f"[stage1] train = {len(train_ds)} samples "
              f"(CT-RATE + {EXTRA_REPEAT}x {len(extra_ds)} RAD-ChestCT)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=rac_collate, drop_last=True, pin_memory=True, persistent_workers=NUM_WORKERS > 0, **_loader_kwargs())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=rac_collate, pin_memory=True, **_loader_kwargs())

    keep_lut = None
    if extra_keep:
        # RAD-ChestCT leaves some CT-RATE pathologies unlabelled; those cells are
        # dropped from the loss rather than taught as negatives.
        keep_lut = dict(extra_keep)
    if LABEL_MASK:
        z = np.load(LABEL_MASK, allow_pickle=True)
        keep_lut = dict(keep_lut or {})
        keep_lut.update({str(n): row for n, row in zip(z["names"], z["keep"])})
        dropped = float((z["keep"] == 0).mean())
        print(f"[stage1] label mask: {LABEL_MASK} ({len(keep_lut)} vols, "
              f"{100*dropped:.2f}% of cells dropped from the loss)", flush=True)

    model = Stage1Model().to(device)
    criterion = AsymmetricLoss() if LOSS_NAME == "asl" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS * len(train_loader), 1), eta_min=LR * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda" and _amp_dtype() == torch.float16)

    # Resume from latest saved epoch checkpoint if available
    start_epoch = 1
    best_auc = 0.0
    for ep in range(EPOCHS, 0, -1):
        ckpt_path = f"{RESULTS_DIR}/stage1_epoch{ep}.pt"
        if os.path.isfile(ckpt_path):
            pkg = torch.load(ckpt_path, map_location=device)
            model.image_encoder.load_state_dict(pkg["image_encoder"])
            model.classifier.load_state_dict(pkg["classifier"])
            best_auc = pkg.get("best_auc", 0.0)
            start_epoch = ep + 1
            print(f"[stage1] resumed from epoch={ep} best_auc={best_auc:.4f}")
            break

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"Stage1 epoch {epoch}/{EPOCHS}")
        for ct, _, _, labels, _, _, accessions in pbar:
            ct = ct.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            w = None
            if keep_lut is not None:
                w = torch.as_tensor(
                    np.stack([keep_lut.get(str(a), np.ones(N_PATH, dtype=np.float32))
                              for a in accessions]),
                    device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(ct, meta_batch(accessions, ct.device) if META_FUSION else None)
                loss = criterion(logits, labels, w) if w is not None else criterion(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        mean_auc, per_auc = validate(model, val_loader, device)
        print(f"[stage1] epoch={epoch} loss={np.mean(losses):.4f} val_auc={mean_auc:.4f} best={best_auc:.4f} t={(time.time()-t0)/60:.1f}m")
        pkg = {
            "image_encoder": model.image_encoder.state_dict(),
            "classifier": model.classifier.state_dict(),
            "epoch": epoch,
            "best_auc": max(best_auc, mean_auc),
            "per_auc": per_auc,
            "config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "loss": LOSS_NAME},
            # save the metadata branch too; without it a checkpoint cannot be
            # rebuilt exactly (inference only needs the zero-input constant, but
            # fine-tuning or ablations do need the weights)
            **({"meta_mlp": model.meta_mlp.state_dict()} if META_FUSION else {}),
        }
        torch.save(pkg, f"{RESULTS_DIR}/stage1_epoch{epoch}.pt")
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(pkg, f"{RESULTS_DIR}/stage1_best.pt")
            with open(f"{RESULTS_DIR}/stage1_best_auc.txt", "w") as f:
                for name in PATHOLOGIES:
                    f.write(f"{name:45s}: {per_auc[name]:.4f}\n")
                f.write(f"{'Mean AUC':45s}: {mean_auc:.4f}\n")

    print(f"[stage1] Done. best_val_auc={best_auc:.4f} ckpt={RESULTS_DIR}/stage1_best.pt")


if __name__ == "__main__":
    main()
