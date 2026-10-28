"""Q-Former checkpoint surgery for a changed pathology schema.

The anatomy Q-Former lays its query tokens out in one contiguous parameter,
``qformer.queries`` of shape ``(n_anatomy + n_pathology + n_global, dim)``, in that
order (see ``AnatomyQFormer.__init__``). It is the ONLY tensor in the module whose
shape depends on the label schema -- everything else is a function of ``dim``.

So adapting a checkpoint to a new pathology list is a row remap:

    rows [0, n_anatomy)                 anatomy   -> copied verbatim (anatomy is anatomy)
    rows [n_anatomy, n_anatomy+n_path)  pathology -> copied BY NAME, or freshly
                                                     initialised if the class is new
    trailing n_global rows              global    -> copied verbatim

``load_state_dict(..., strict=False)`` will not save you here: it tolerates missing
and unexpected KEYS but raises on a shape MISMATCH, so an unremapped load fails loudly.
That is the good case. The bad case is ``RAC_WARM_START_CKPT``, which historically did
not load this module at all -- see ``remap_qformer_state``'s use in ``train.py``.
"""
from __future__ import annotations

import torch


#: The 18 CT-RATE classes, in the order every checkpoint in ``artifacts/runs`` was
#: trained with. Deliberately duplicated from ``dataset.PATHOLOGIES`` rather than
#: imported: at a new site ``dataset.PATHOLOGIES`` becomes the *destination* schema,
#: and the source schema of an old checkpoint must not move with it.
CTRATE_PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

QUERY_KEY = "qformer.queries"


class SchemaMismatch(RuntimeError):
    """The checkpoint's query block cannot be reconciled with the target module."""


def remap_qformer_state(
    state: dict,
    dst_pathologies: list[str],
    src_pathologies: list[str] | None = None,
    n_anatomy: int = 10,
    n_global: int | None = None,
    generator: torch.Generator | None = None,
    reinit: set[str] | list[str] | None = None,
) -> tuple[dict, dict]:
    """Return ``(new_state, report)`` with ``qformer.queries`` remapped.

    ``src_pathologies`` defaults to the 18 CT-RATE classes. ``n_global`` defaults to
    whatever the source checkpoint had. New classes are drawn from a normal with the
    *source block's* std rather than the module default, so a fresh row sits on the
    same scale as its trained neighbours instead of being 5x too large or too small.

    ``reinit`` forces fresh rows for classes that survive by NAME but change MEANING at
    the new site -- e.g. pediatric "Emphysema" is bullae/pneumatocele, not smoking-related
    destruction, and "Mosaic attenuation" is air trapping rather than chronic PE. Copying
    the adult row there transfers a prior you do not want; the detector may still fire but
    its calibration is wrong. Cheaper to relearn the row than to unlearn it.

    The returned state is a shallow copy; only the query tensor is replaced.
    """
    if QUERY_KEY not in state:
        raise SchemaMismatch(
            f"state dict has no {QUERY_KEY!r} -- is this a qformer_module payload? "
            f"got keys like {list(state)[:3]}"
        )
    src_pathologies = list(src_pathologies if src_pathologies is not None else CTRATE_PATHOLOGIES)
    dst_pathologies = list(dst_pathologies)

    src_q = state[QUERY_KEY]
    total_src, dim = src_q.shape
    src_global = total_src - n_anatomy - len(src_pathologies)
    if src_global < 0:
        raise SchemaMismatch(
            f"{QUERY_KEY} has {total_src} rows but the declared source schema needs at least "
            f"{n_anatomy} anatomy + {len(src_pathologies)} pathology rows. "
            f"Check n_anatomy and src_pathologies against the checkpoint."
        )
    dst_global = src_global if n_global is None else int(n_global)
    total_dst = n_anatomy + len(dst_pathologies) + dst_global

    # Pathology rows sit between the anatomy block and the trailing global block.
    reinit = set(reinit or ())
    unknown_reinit = reinit - set(dst_pathologies)
    if unknown_reinit:
        raise SchemaMismatch(
            f"--reinit names classes that are not in the new schema: {sorted(unknown_reinit)}"
        )
    src_index = {name: n_anatomy + i for i, name in enumerate(src_pathologies)}
    src_std = float(src_q.std())

    dst_q = torch.empty(total_dst, dim, dtype=src_q.dtype)
    torch.nn.init.normal_(dst_q, mean=0.0, std=src_std, generator=generator)

    dst_q[:n_anatomy] = src_q[:n_anatomy]

    kept, added = [], []
    for i, name in enumerate(dst_pathologies):
        row = n_anatomy + i
        if name in src_index and name not in reinit:
            dst_q[row] = src_q[src_index[name]]
            kept.append(name)
        else:
            added.append(name)  # already holds the fresh draw from the init above

    # Copy as many global rows as both sides have; any extra stay freshly initialised.
    n_copy_global = min(src_global, dst_global)
    if n_copy_global:
        dst_q[total_dst - dst_global: total_dst - dst_global + n_copy_global] = \
            src_q[total_src - src_global: total_src - src_global + n_copy_global]

    new_state = dict(state)
    new_state[QUERY_KEY] = dst_q
    report = {
        "src_rows": total_src,
        "dst_rows": total_dst,
        "n_anatomy": n_anatomy,
        "src_global": src_global,
        "dst_global": dst_global,
        "kept": kept,
        "added": added,
        "dropped": [n for n in src_pathologies if n not in set(dst_pathologies)],
        "reinit": sorted(reinit & set(src_index)),
        "src_std": src_std,
    }
    return new_state, report


def format_report(report: dict) -> str:
    warn = ""
    if report["dst_global"] != report["src_global"]:
        warn = (
            f"\n[qformer-surgery]   WARNING: global-query count changed "
            f"{report['src_global']} -> {report['dst_global']}. This is almost always an "
            f"unset RAC_QFORMER_QUERIES: it is an absolute total, so shrinking the pathology "
            f"list while leaving it at the old value silently inflates the global block. "
            f"Set RAC_QFORMER_QUERIES = 10 + n_pathologies + {report['src_global']} "
            f"(= {10 + len(report['kept']) + len(report['added']) + report['src_global']}) "
            f"to keep the original {report['src_global']}."
        )
    return (
        f"[qformer-surgery] {report['src_rows']} -> {report['dst_rows']} query rows "
        f"({report['n_anatomy']} anatomy + {len(report['kept']) + len(report['added'])} pathology "
        f"+ {report['dst_global']} global)\n"
        f"[qformer-surgery]   kept {len(report['kept'])}: {', '.join(report['kept']) or '-'}\n"
        f"[qformer-surgery]   dropped {len(report['dropped'])}: {', '.join(report['dropped']) or '-'}\n"
        f"[qformer-surgery]   new (fresh init, std={report['src_std']:.4f}) "
        f"{len(report['added'])}: {', '.join(report['added']) or '-'}"
        + (f"\n[qformer-surgery]   force-reinit (same name, different disease) "
           f"{len(report['reinit'])}: {', '.join(report['reinit'])}" if report.get("reinit") else "")
        + warn
    )


def load_qformer_from_pkg(
    qformer_module,
    pkg: dict,
    dst_pathologies: list[str],
    src_pathologies: list[str] | None = None,
    n_anatomy: int | None = None,
    verbose: bool = True,
) -> dict:
    """Load ``pkg['qformer_module']`` into ``qformer_module``, remapping if needed.

    Handles both :class:`AnatomyQFormer` (query key ``qformer.queries``, rows split
    anatomy/pathology/global) and a plain :class:`QFormer` (key ``queries``, no schema
    structure, so a row-count change cannot be remapped and is an error).

    Raises ``SchemaMismatch`` if the payload has no Q-Former state -- loading a trained
    backbone next to a randomly initialised Q-Former is never what the caller wanted,
    and it is invisible without this check.
    """
    if "qformer_module" not in pkg:
        raise SchemaMismatch(
            "checkpoint has no 'qformer_module' key but this run builds a Q-Former. "
            "Continuing would train a randomly initialised Q-Former (38.2M params) on a "
            "warm-started backbone, with no error at any point. Use a checkpoint that "
            "has one, or set RAC_ALLOW_FRESH_QFORMER=1 if that is genuinely intended."
        )
    state = pkg["qformer_module"]
    is_anatomy = hasattr(qformer_module, "anatomy_labels")
    if is_anatomy:
        target_rows = qformer_module.qformer.queries.shape[0]
        if n_anatomy is None:
            n_anatomy = len(qformer_module.anatomy_labels)
        key = QUERY_KEY
    else:
        target_rows = qformer_module.queries.shape[0]
        key = "queries"

    src_rows = state[key].shape[0] if key in state else None
    if src_rows is None:
        raise SchemaMismatch(
            f"checkpoint Q-Former state has no {key!r}; it looks like the other Q-Former "
            f"flavour (got keys like {list(state)[:3]}). Check RAC_USE_ANATOMY_QFORMER."
        )

    remapped = False
    if src_rows != target_rows:
        if not is_anatomy:
            raise SchemaMismatch(
                f"plain QFormer query count changed ({src_rows} -> {target_rows}) and has no "
                f"anatomy/pathology/global structure to remap along. Keep RAC_QFORMER_QUERIES "
                f"at {src_rows}, or switch to the anatomy Q-Former."
            )
        state, report = remap_qformer_state(
            state, dst_pathologies, src_pathologies=src_pathologies, n_anatomy=n_anatomy,
            n_global=target_rows - n_anatomy - len(dst_pathologies),
        )
        remapped = True
        if verbose:
            print(format_report(report), flush=True)

    missing, unexpected = qformer_module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SchemaMismatch(
            f"Q-Former state does not match the built module "
            f"(missing={len(missing)} unexpected={len(unexpected)}). "
            f"Check RAC_QFORMER_QUERIES/DEPTH/HEADS against the checkpoint. "
            f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}"
        )
    if verbose:
        print(f"[RAC] Q-Former restored from checkpoint ({target_rows} query rows)", flush=True)
    return {"rows": target_rows, "remapped": remapped}
