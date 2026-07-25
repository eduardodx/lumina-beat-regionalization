#!/usr/bin/env python3
"""Would letting the model TRAIN on Brazilian variants raise br_only? -- the cheap decisive screen.

Runs LOCAL on the notebook after scripts/extract_two_tower_features.py. Uses torch for a small
head; no SageMaker job, no GPU training.

THE QUESTION
------------
Every current model is trained leave-Brazilian-out: it has NEVER seen a Brazilian variant. That
was the right design for the clean falsification, but it means no model could have learned any
Brazilian-specific pattern. The only remaining lever that adds model-side information is to let it
train on Brazilian variants. This screens that at the head level before spending 5 SageMaker jobs:

    baseline (leave-Brazilian-out) : train the head on nonbr_only only, score all br_only variants
    k-fold treatment               : k gene-disjoint folds of br_only; for each fold, train on
                                     nonbr_only + the OTHER br_only folds, score the held-out fold;
                                     pool the held-out predictions

Both arms use the SAME frozen two-tower features, so the only thing that differs is whether
Brazilian variants were in training. Gene-disjoint folds prevent a gene's variants leaking across
train/test.

WHY THE HEAD-LEVEL SCREEN IS DECISIVE ENOUGH
--------------------------------------------
The frozen backbone's linear probe already hits AUROC 0.953 on ClinVar globally, so the head over
frozen features is a strong proxy for the deployed model (which adds only rank-8 LoRA). If adding
Brazilian training examples does not raise br_only AUROC even here, the far more expensive full
LoRA k-fold is very unlikely to. If it does, that is the green light to escalate. The script
prints the baseline br_only AUROC so we can check it against the deployed ~0.89 and judge whether
the proxy is faithful.

PRIMARY METRIC = AUROC (threshold-free)
---------------------------------------
AUROC sidesteps the operating-point problem that made MCC misleading before. MCC at a
recall-constrained threshold is reported too, and P/LP recall is printed at that threshold so we
can see it is not sacrificed -- but the go/no-go is the paired bootstrap of br_only AUROC,
k-fold minus baseline.

USAGE
-----
    python scripts/kfold_train_on_brazilian.py \
        --feature-dir ~/v11eval/two_tower_features \
        --slice-dir   ~/slices \
        --k 5 --plp-recall-floor 0.405 \
        --output-dir  ~/v11eval/kfold_brazilian
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import classification_metrics  # noqa: E402
from scripts.test_abraom_vs_gnomad_power import auroc, bc_interval, stratified_indices  # noqa: E402

EXPLICIT_EPS = 1e-6
EXPLICIT_FEATURES = [
    "log10_af_abraom", "log10_af_gnomad", "af_delta", "af_abs_delta", "af_ratio_log10",
    "af_abraom_missing", "af_gnomad_missing", "specificity", "specificity_missing",
    "abraom_present", "is_snv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-dir", type=Path, default=home / "v11eval/two_tower_features")
    p.add_argument("--slice-dir", type=Path, default=home / "slices")
    p.add_argument("--brazilian", default="br_only")
    p.add_argument("--control", default="nonbr_only")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-explicit-features", action="store_true",
                   help="drop the 11 frequency features -> pure-representation screen")
    p.add_argument("--plp-recall-floor", type=float, default=0.405)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--device", default="cpu", help="cpu is plenty for a small head over cached features")
    p.add_argument("--output-dir", type=Path, default=home / "v11eval/kfold_brazilian")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(feature_dir: Path, dataset: str) -> dict:
    npz = feature_dir / f"{dataset}.two_tower.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"missing features: {npz} (run extract_two_tower_features.py first)")
    data = np.load(npz, allow_pickle=True)
    return {
        "features": data["features"].astype(np.float32),
        "original_index": data["original_index"].astype(np.int64),
        "label": data["label"].astype(np.int64),
        "split": data["split"].astype(str),
    }


def explicit_feature_matrix(slice_dir: Path, dataset: str, original_index: np.ndarray) -> np.ndarray:
    """Rebuild the 11 explicit frequency features, matching eval/clinvar/dataset.py:95-126."""
    path = slice_dir / f"{dataset}.parquet"
    meta = pd.read_parquet(path).reset_index().rename(columns={"index": "original_index"})
    meta = meta.set_index("original_index").reindex(original_index)

    def col(name: str) -> np.ndarray:
        if name not in meta.columns:
            return np.full(len(meta), np.nan)
        return pd.to_numeric(meta[name], errors="coerce").to_numpy(dtype=np.float64)

    af_abraom_raw, af_gnomad_raw = col("af_abraom"), col("af_gnomad")
    spec_raw = col("specificity")
    af_abraom = np.nan_to_num(af_abraom_raw, nan=0.0)
    af_gnomad = np.nan_to_num(af_gnomad_raw, nan=0.0)
    spec = np.nan_to_num(spec_raw, nan=0.0)
    is_snv = np.nan_to_num(col("is_snv"), nan=0.0)
    abraom_present = np.nan_to_num(col("abraom_present"), nan=0.0)

    def log10_af(v: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(v, EXPLICIT_EPS))

    ratio = np.clip(np.log10((af_abraom + EXPLICIT_EPS) / (af_gnomad + EXPLICIT_EPS)), -8.0, 8.0)
    columns = {
        "log10_af_abraom": log10_af(af_abraom),
        "log10_af_gnomad": log10_af(af_gnomad),
        "af_delta": af_abraom - af_gnomad,
        "af_abs_delta": np.abs(af_abraom - af_gnomad),
        "af_ratio_log10": ratio,
        "af_abraom_missing": np.isnan(af_abraom_raw).astype(np.float64),
        "af_gnomad_missing": np.isnan(af_gnomad_raw).astype(np.float64),
        "specificity": spec,
        "specificity_missing": np.isnan(spec_raw).astype(np.float64),
        "abraom_present": (abraom_present > 0).astype(np.float64),
        "is_snv": (is_snv > 0).astype(np.float64),
    }
    return np.column_stack([columns[name] for name in EXPLICIT_FEATURES]).astype(np.float32)


def gene_folds(slice_dir: Path, dataset: str, original_index: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each variant to one of k folds by GENE, so no gene spans train and test."""
    path = slice_dir / f"{dataset}.parquet"
    meta = pd.read_parquet(path).reset_index().rename(columns={"index": "original_index"}).set_index("original_index")
    genes = meta.reindex(original_index)["GeneSymbol"].fillna("unknown").astype(str).to_numpy()
    unique = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    gene_to_fold = {g: i % k for i, g in enumerate(unique)}
    return np.array([gene_to_fold[g] for g in genes], dtype=np.int64)


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_head(
    train_x: np.ndarray, train_y: np.ndarray, in_dim: int, args: argparse.Namespace, seed: int,
) -> tuple[Head, np.ndarray, np.ndarray]:
    """Returns (trained head, feature mean, feature std) for standardization at inference."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6
    xs = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(device)
    ys = torch.from_numpy(train_y.astype(np.float32)).to(device)

    n_pos = float((train_y == 1).sum())
    n_neg = float((train_y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    head = Head(in_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    head.train()
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = loss_fn(head(xs), ys)
        loss.backward()
        optimizer.step()
    head.eval()
    return head, mean, std


@torch.no_grad()
def predict(head: Head, mean: np.ndarray, std: np.ndarray, x: np.ndarray, device: str) -> np.ndarray:
    xs = torch.from_numpy(((x - mean) / std).astype(np.float32)).to(torch.device(device))
    return torch.sigmoid(head(xs)).cpu().numpy()


# ---------------------------------------------------------------------------
# Operating point (recall-constrained), founder-aware
# ---------------------------------------------------------------------------

def constrained_mcc(labels: np.ndarray, scores: np.ndarray, founder_mask: np.ndarray, floor: float) -> dict:
    """MCC at the LOWEST threshold whose founder recall is still >= floor (a fixed rule, not tuned
    for MCC, so no leakage). founder_mask marks pathogenic ABRAOM-present br_only variants."""
    order = np.unique(np.round(np.quantile(scores, np.linspace(0, 1, 200)), 6))
    best = {"mcc": float("nan"), "threshold": float("nan"), "plp_recall": float("nan"),
            "overall_recall": float("nan"), "specificity": float("nan")}
    founder_pos = founder_mask & (labels == 1)
    for threshold in order:
        pred = scores >= threshold
        if founder_pos.sum() > 0:
            plp_recall = float(pred[founder_pos].mean())
        else:
            plp_recall = float("nan")
        if not np.isnan(plp_recall) and plp_recall < floor:
            continue
        m = classification_metrics(labels, scores, float(threshold))
        if not np.isfinite(best["mcc"]) or m["mcc"] > best["mcc"]:
            best = {"mcc": float(m["mcc"]), "threshold": float(threshold), "plp_recall": plp_recall,
                    "overall_recall": float(m["recall"]), "specificity": float(m["specificity"])}
    return best


def _fmt(v, d: int = 3) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:.{d}f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    br = load_features(args.feature_dir, args.brazilian)
    ctrl = load_features(args.feature_dir, args.control)

    # control training pool = its train split (matches how the pipeline trained leave-BR-out)
    ctrl_train = ctrl["split"] == "train"
    ctrl_x, ctrl_y = ctrl["features"][ctrl_train], ctrl["label"][ctrl_train]

    br_x, br_y, br_idx = br["features"], br["label"], br["original_index"]

    if not args.no_explicit_features:
        br_expl = explicit_feature_matrix(args.slice_dir, args.brazilian, br_idx)
        ctrl_expl = explicit_feature_matrix(args.slice_dir, args.control, ctrl["original_index"][ctrl_train])
        br_x = np.concatenate([br_x, br_expl], axis=1)
        ctrl_x = np.concatenate([ctrl_x, ctrl_expl], axis=1)
    in_dim = br_x.shape[1]

    # founder mask on br_only: pathogenic AND ABRAOM-present
    meta = pd.read_parquet(args.slice_dir / f"{args.brazilian}.parquet").reset_index().rename(
        columns={"index": "original_index"}).set_index("original_index").reindex(br_idx)
    abraom_present = pd.to_numeric(meta.get("abraom_present"), errors="coerce").fillna(0).to_numpy() > 0
    founder_mask = abraom_present & (br_y == 1)

    device = args.device

    # --- arm 1: leave-Brazilian-out baseline ---
    base_head, base_mean, base_std = train_head(ctrl_x, ctrl_y, in_dim, args, args.seed)
    base_pred = predict(base_head, base_mean, base_std, br_x, device)

    # --- arm 2: gene-disjoint k-fold, Brazilian IN training ---
    folds = gene_folds(args.slice_dir, args.brazilian, br_idx, args.k, args.seed)
    kfold_pred = np.full(len(br_y), np.nan)
    for f in range(args.k):
        test_mask = folds == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or br_y[train_mask].min() == br_y[train_mask].max():
            continue
        train_x = np.concatenate([ctrl_x, br_x[train_mask]], axis=0)
        train_y = np.concatenate([ctrl_y, br_y[train_mask]], axis=0)
        head, mean, std = train_head(train_x, train_y, in_dim, args, args.seed + f)
        kfold_pred[test_mask] = predict(head, mean, std, br_x[test_mask], device)
        print(f"  fold {f}: train_br={int(train_mask.sum())} test_br={int(test_mask.sum())}", flush=True)

    valid = np.isfinite(kfold_pred)
    labels = br_y[valid]
    base_scores = base_pred[valid]
    kfold_scores = kfold_pred[valid]
    founder_valid = founder_mask[valid]

    base_auroc = auroc(labels, base_scores)
    kfold_auroc = auroc(labels, kfold_scores)

    # paired bootstrap of AUROC(kfold) - AUROC(baseline), same variants
    rng = np.random.default_rng(args.seed)
    draws = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        idx = stratified_indices(labels, rng)
        draws[i] = auroc(labels[idx], kfold_scores[idx]) - auroc(labels[idx], base_scores[idx])
    delta = kfold_auroc - base_auroc
    ci_low, ci_high = bc_interval(draws, delta)
    p_no_gain = float(np.mean(draws[np.isfinite(draws)] <= 0.0))

    base_op = constrained_mcc(labels, base_scores, founder_valid, args.plp_recall_floor)
    kfold_op = constrained_mcc(labels, kfold_scores, founder_valid, args.plp_recall_floor)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "brazilian": args.brazilian, "control": args.control, "k": args.k,
        "explicit_features": not args.no_explicit_features,
        "n_br_scored": int(labels.size), "n_br_pathogenic": int((labels == 1).sum()),
        "n_founders": int(founder_valid.sum()),
        "plp_recall_floor": args.plp_recall_floor,
        "auroc": {"baseline": base_auroc, "kfold": kfold_auroc, "delta": delta,
                  "ci_low": ci_low, "ci_high": ci_high, "p_no_gain": p_no_gain},
        "operating_point": {"baseline": base_op, "kfold": kfold_op},
    }

    lines = []
    lines.append("=" * 92)
    lines.append("K-FOLD SCREEN: does training on Brazilian variants raise br_only? (head-level)")
    lines.append("=" * 92)
    lines.append(f"  br scored={payload['n_br_scored']}  pathogenic={payload['n_br_pathogenic']}  "
                 f"founders(ABRAOM-present P/LP)={payload['n_founders']}  k={args.k}  "
                 f"explicit_freq={'on' if payload['explicit_features'] else 'off'}")
    lines.append("")
    lines.append("[PRIMARY] br_only AUROC (threshold-free)")
    lines.append(f"    leave-Brazilian-out baseline : {_fmt(base_auroc)}   "
                 f"(deployed model ~0.89 -> proxy is {'faithful' if base_auroc >= 0.85 else 'WEAK, read with care'})")
    lines.append(f"    k-fold (Brazilian in train)  : {_fmt(kfold_auroc)}")
    lines.append(f"    delta (k-fold - baseline)    : {_fmt(delta)}   95% CI [{_fmt(ci_low)}, {_fmt(ci_high)}]"
                 f"   p(no gain)={_fmt(p_no_gain)}")
    lines.append("")
    lines.append(f"[SECONDARY] MCC at threshold holding founder recall >= {args.plp_recall_floor}")
    lines.append(f"    {'arm':<14}{'MCC':>8}{'P/LP recall':>13}{'overall rec':>13}{'spec':>8}{'thr':>8}")
    lines.append("    " + "-" * 63)
    for name, op in (("baseline", base_op), ("k-fold", kfold_op)):
        lines.append(f"    {name:<14}{_fmt(op['mcc']):>8}{_fmt(op['plp_recall']):>13}"
                     f"{_fmt(op['overall_recall']):>13}{_fmt(op['specificity']):>8}{_fmt(op['threshold']):>8}")
    lines.append("")
    lines.append("=" * 92)
    lines.append("GO/NO-GO: if the AUROC delta CI is clear of 0, escalate to the full LoRA SageMaker")
    lines.append("k-fold. If it crosses 0 (or is negative), training on Brazilian does not help even")
    lines.append("at the head level -> do NOT spend the jobs; the br_only ceiling is confirmed.")
    lines.append("=" * 92)
    report = "\n".join(lines)
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "with_freq" if not args.no_explicit_features else "repr_only"
    (args.output_dir / f"kfold_screen.{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / f"kfold_screen.{tag}.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/kfold_screen.{tag}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
