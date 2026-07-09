#!/usr/bin/env python3
"""Evaluate ABRAOM regionalization by masked REF/ALT likelihood deltas."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.checkpoints import load_lumina_backbone_from_checkpoint  # noqa: E402
from src.constants import DNA_VOCAB, MASK_ID, PAD_ID, SNV_ALT_TO_INDEX, SNV_BASES  # noqa: E402
from src.dataset import encode_dna  # noqa: E402

PAIR_COLUMNS = [
    "pair_id",
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "context_start",
    "context_end",
    "variant_offset",
    "ref_sequence",
    "af_abraom",
    "af_gnomad",
    "specificity",
    "specificity_bin",
    "split",
    "n_fraction",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--pairs-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def read_pairs(path: Path, *, split: str | None, limit: int | None) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=PAIR_COLUMNS)
    rows = table.to_pylist()
    if split:
        rows = [row for row in rows if row["split"] == split]
    if limit is not None:
        rows = rows[: max(0, limit)]
    if not rows:
        raise ValueError(f"No ABRAOM REF/ALT pairs selected from {path}.")
    return rows


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_masked_contexts(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded: list[list[int]] = []
    for row in rows:
        seq = str(row["ref_sequence"]).upper()
        offset = int(row["variant_offset"])
        if offset < 0 or offset >= len(seq):
            raise ValueError(f"variant_offset out of bounds for pair_id={row['pair_id']}: {offset}")
        ref = str(row["ref"]).upper()
        if seq[offset] != ref:
            raise ValueError(
                f"REF mismatch in eval pair_id={row['pair_id']}: sequence={seq[offset]!r}, ref={ref!r}"
            )
        ids = encode_dna(seq)
        ids[offset] = MASK_ID
        encoded.append(ids)
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device)
    attention_mask = (input_ids != PAD_ID).long()
    return input_ids, attention_mask


def base_log_prob(log_probs: torch.Tensor, base: str) -> torch.Tensor:
    base = base.upper()
    if log_probs.shape[-1] == len(SNV_BASES):
        return log_probs[..., SNV_ALT_TO_INDEX[base]]
    return log_probs[..., DNA_VOCAB[base]]


@torch.no_grad()
def score_rows(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    label: str,
    progress_every: int,
) -> list[float]:
    model.eval()
    scores: list[float] = []
    total_batches = math.ceil(len(rows) / batch_size)
    for batch_index, batch_rows in enumerate(batched(rows, batch_size), start=1):
        input_ids, attention_mask = encode_masked_contexts(batch_rows, device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["mlm_logits"]
        offsets = torch.tensor([int(row["variant_offset"]) for row in batch_rows], dtype=torch.long, device=device)
        batch_indices = torch.arange(len(batch_rows), dtype=torch.long, device=device)
        site_log_probs = torch.log_softmax(logits[batch_indices, offsets], dim=-1)
        ref_logp = torch.stack(
            [base_log_prob(site_log_probs[index], str(row["ref"])) for index, row in enumerate(batch_rows)]
        )
        alt_logp = torch.stack(
            [base_log_prob(site_log_probs[index], str(row["alt"])) for index, row in enumerate(batch_rows)]
        )
        scores.extend((alt_logp - ref_logp).detach().cpu().float().tolist())
        should_log_progress = (
            progress_every > 0
            and (batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches)
        )
        if should_log_progress:
            print(
                f"[{label}] scored_batches={batch_index}/{total_batches} "
                f"scored_rows={len(scores)}/{len(rows)}",
                flush=True,
            )
    return scores


def frequency_bin(value: float) -> str:
    bins = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    labels = ["0", "(0,0.005]", "(0.005,0.01]", "(0.01,0.05]", "(0.05,0.1]", "(0.1,0.5]", "(0.5,1]"]
    if value <= 0.0:
        return labels[0]
    for idx in range(1, len(bins)):
        if value <= bins[idx]:
            return labels[idx]
    return ">1"


def summarize(rows: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"overall": rows} if group_key is None else {}
    if group_key is not None:
        for row in rows:
            groups.setdefault(str(row[group_key]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items()):
        base_scores = np.asarray([row["base_score"] for row in group_rows], dtype=np.float64)
        candidate_scores = np.asarray([row["candidate_score"] for row in group_rows], dtype=np.float64)
        delta_scores = np.asarray([row["delta_score"] for row in group_rows], dtype=np.float64)
        summaries.append(
            {
                "group": group,
                "n": len(group_rows),
                "mean_base_score": float(np.mean(base_scores)),
                "mean_candidate_score": float(np.mean(candidate_scores)),
                "mean_delta_score": float(np.mean(delta_scores)),
                "median_delta_score": float(np.median(delta_scores)),
                "sem_delta_score": float(np.std(delta_scores, ddof=1) / math.sqrt(len(delta_scores)))
                if len(delta_scores) > 1
                else 0.0,
            }
        )
    return summaries


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ABRAOM Regionalization Evaluation",
        "",
        f"base_checkpoint: `{summary['base_checkpoint']}`",
        f"candidate_checkpoint: `{summary['candidate_checkpoint']}`",
        f"pairs_evaluated: {summary['n_pairs']}",
        "",
        "## Overall",
        "",
        "| n | mean_base_score | mean_candidate_score | mean_delta_score | median_delta_score |",
        "|---:|---:|---:|---:|---:|",
    ]
    overall = summary["overall"][0]
    lines.append(
        "| {n} | {mean_base_score:.6f} | {mean_candidate_score:.6f} | "
        "{mean_delta_score:.6f} | {median_delta_score:.6f} |".format(**overall)
    )
    lines.extend(
        [
            "",
            "## By Specificity Bin",
            "",
            "| group | n | mean_delta_score | sem_delta_score |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["by_specificity_bin"]:
        lines.append("| {group} | {n} | {mean_delta_score:.6f} | {sem_delta_score:.6f} |".format(**row))
    lines.extend(
        [
            "",
            "## By ABRAOM AF Bin",
            "",
            "| group | n | mean_delta_score | sem_delta_score |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["by_af_abraom_bin"]:
        lines.append("| {group} | {n} | {mean_delta_score:.6f} | {sem_delta_score:.6f} |".format(**row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_pairs(Path(args.pairs_parquet), split=args.split, limit=args.limit)
    base_model, _base_model_config, _base_config = load_lumina_backbone_from_checkpoint(
        args.base_checkpoint,
        requested_model_key="beat-v10",
        device=device,
    )
    candidate_model, _candidate_model_config, _candidate_config = load_lumina_backbone_from_checkpoint(
        args.candidate_checkpoint,
        requested_model_key="beat-v10",
        device=device,
    )

    base_scores = score_rows(
        base_model,
        rows,
        device=device,
        batch_size=args.batch_size,
        label="base",
        progress_every=args.progress_every,
    )
    candidate_scores = score_rows(
        candidate_model,
        rows,
        device=device,
        batch_size=args.batch_size,
        label="candidate",
        progress_every=args.progress_every,
    )
    for row, base_score, candidate_score in zip(rows, base_scores, candidate_scores, strict=True):
        row["base_score"] = float(base_score)
        row["candidate_score"] = float(candidate_score)
        row["delta_score"] = float(candidate_score - base_score)
        row["af_abraom_bin"] = frequency_bin(float(row["af_abraom"]))
        row["af_gnomad_bin"] = frequency_bin(float(row["af_gnomad"]))

    summary = {
        "base_checkpoint": args.base_checkpoint,
        "candidate_checkpoint": args.candidate_checkpoint,
        "pairs_parquet": args.pairs_parquet,
        "split": args.split,
        "n_pairs": len(rows),
        "overall": summarize(rows),
        "by_specificity_bin": summarize(rows, "specificity_bin"),
        "by_af_abraom_bin": summarize(rows, "af_abraom_bin"),
        "by_af_gnomad_bin": summarize(rows, "af_gnomad_bin"),
        "by_chrom": summarize(rows, "chrom"),
    }

    pq.write_table(pa.Table.from_pylist(rows), output_dir / "eval_by_variant.parquet")
    pq.write_table(pa.Table.from_pylist(summary["by_specificity_bin"]), output_dir / "eval_by_specificity_bin.parquet")
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(output_dir / "EVAL_REPORT.md", summary)
    print(json.dumps(summary["overall"][0], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
