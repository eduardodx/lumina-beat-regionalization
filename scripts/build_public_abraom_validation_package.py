#!/usr/bin/env python3
"""Build a public-evidence validation package for ABRAOM regionalization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import classification_metrics  # noqa: E402

DEFAULT_REVIEW_QUEUE = Path(
    "artifacts/clinvar_regional_eval/regional_signal_validation_next_step/"
    "critical_error_audit/combined_manual_review_queue.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/clinvar_regional_eval/public_abraom_validation")
DEFAULT_MODEL_ROOT = Path("artifacts/clinvar_regional_eval")
DEFAULT_M5_V2_DIR = Path("artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned")
DEFAULT_M5_V3_DIR = Path("artifacts/clinvar_regional_eval/m5_v3_safety_calibrated")

MODEL_SPECS = {
    "M0": {
        "root": DEFAULT_MODEL_ROOT / "m0_nonbr_beatv10_v1_sagemaker",
        "kind": "raw",
    },
    "M7_dynamic_scrambled": {
        "root": DEFAULT_MODEL_ROOT / "m7_dynamic_scrambled_nonbr_beatv10_v1_sagemaker",
        "kind": "raw",
    },
    "M5_v2_calibrated": {
        "root": DEFAULT_M5_V2_DIR,
        "kind": "calibrated",
    },
    "M5_v3_safety": {
        "root": DEFAULT_M5_V3_DIR,
        "kind": "calibrated",
    },
}

PUBLIC_QUEUE_COLUMNS = [
    "public_review_status",
    "public_evidence_decision",
    "public_evidence_strength",
    "manual_curation_status",
    "manual_public_classification",
    "manual_public_source_url_or_pmid",
    "manual_evidence_note",
    "clinvar_variation_url",
    "clinvar_search_url",
    "review_question",
    "audit_type",
    "priority_tier",
    "priority_score",
    "failure_category",
    "variant_key",
    "GeneSymbol",
    "Chromosome",
    "Start",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "variant_type",
    "label",
    "prediction_used",
    "score_used",
    "threshold_used",
    "score_margin",
    "molecular_probability",
    "regional_discount",
    "effective_regional_discount",
    "af_abraom",
    "af_gnomad",
    "specificity",
    "specificity_bin",
    "source_review_status",
    "regional_clinical_significance_values",
    "clinvar_variation_ids",
    "max_review_status_rank_aggregate",
    "has_brazilian_submitter",
    "has_non_brazilian_submitter",
    "recommended_action",
]

MANUAL_FIELD_DEFAULTS = {
    "manual_curation_status": "pending",
    "manual_public_classification": "",
    "manual_public_source_url_or_pmid": "",
    "manual_evidence_note": "",
}
MANUAL_EXCLUDED_STATUSES = {"exclude", "excluded", "reject", "rejected", "invalid", "remove", "removed"}
READY_REVIEW_STATUSES = {"local_public_supports_label", "manual_public_supports_label"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--m5-v2-dir", type=Path, default=DEFAULT_M5_V2_DIR)
    parser.add_argument("--m5-v3-dir", type=Path, default=DEFAULT_M5_V3_DIR)
    parser.add_argument("--priority-tiers", nargs="*", default=["P0_manual_review", "P1_manual_review"])
    return parser.parse_args(argv)


def read_threshold(model_dir: Path) -> float:
    path = model_dir / "regional_eval_summary.json"
    if not path.is_file():
        return 0.5
    return float(json.loads(path.read_text(encoding="utf-8")).get("threshold", 0.5))


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def has_term(value: Any, terms: tuple[str, ...]) -> bool:
    text = normalize_text(value).lower().replace("_", " ")
    return any(term in text for term in terms)


def first_variation_id(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    for sep in [";", ",", "|"]:
        if sep in text:
            text = text.split(sep)[0].strip()
            break
    return text


def clinvar_variation_url(variation_ids: Any) -> str:
    variation_id = first_variation_id(variation_ids)
    if not variation_id:
        return ""
    return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{quote_plus(variation_id)}/"


def clinvar_search_url(row: pd.Series) -> str:
    parts = [
        normalize_text(row.get("variant_key")),
        normalize_text(row.get("GeneSymbol")),
        normalize_text(row.get("regional_clinical_significance_values")),
    ]
    term = " ".join(part for part in parts if part)
    return f"https://www.ncbi.nlm.nih.gov/clinvar/?term={quote_plus(term)}"


def classify_public_evidence(row: pd.Series) -> tuple[str, str, str]:
    variation_id = first_variation_id(row.get("clinvar_variation_ids"))
    significance = row.get("regional_clinical_significance_values")
    label = int(row.get("label", -1))
    has_benign = has_term(significance, ("benign", "likely benign"))
    has_pathogenic = has_term(significance, ("pathogenic", "likely pathogenic"))

    if not variation_id and not normalize_text(significance):
        return "needs_public_lookup", "no_local_variation_id_or_significance", "none"
    if has_pathogenic and has_benign:
        return "conflict_review", "mixed_pathogenic_and_benign_public_assertions", "moderate"
    if label == 1 and has_pathogenic:
        return "local_public_supports_label", "supports_plp_sentinel", "moderate"
    if label == 0 and has_benign:
        return "local_public_supports_label", "supports_benign_sentinel", "moderate"
    if label == 1 and has_benign:
        return "public_label_conflict", "public_benign_conflicts_with_plp_label", "high"
    if label == 0 and has_pathogenic:
        return "public_label_conflict", "public_pathogenic_conflicts_with_benign_label", "high"
    return "needs_public_lookup", "local_variation_id_without_decisive_significance", "low"


def classify_curated_evidence(row: pd.Series) -> tuple[str, str, str]:
    """Classify local public evidence, optionally overridden by completed manual curation."""
    status = normalize_text(row.get("manual_curation_status")).lower().replace(" ", "_")
    classification = normalize_text(row.get("manual_public_classification"))
    source = normalize_text(row.get("manual_public_source_url_or_pmid"))
    note = normalize_text(row.get("manual_evidence_note"))
    has_manual_payload = bool(status and status != "pending") or bool(classification or source or note)
    if not has_manual_payload:
        return classify_public_evidence(row)

    if status in MANUAL_EXCLUDED_STATUSES:
        return "manual_excluded", "manual_excluded_from_sentinel", "manual"
    if not classification:
        return "manual_review_incomplete", "manual_missing_classification", "low"
    if not source:
        return "manual_review_incomplete", "manual_missing_public_source", "low"

    label = int(row.get("label", -1))
    manual_has_benign = has_term(classification, ("benign", "likely benign", "b/lb", "blb"))
    manual_has_pathogenic = has_term(classification, ("pathogenic", "likely pathogenic", "p/lp", "plp"))
    manual_has_uncertain = has_term(classification, ("conflict", "conflicting", "uncertain", "vus"))

    if manual_has_uncertain or (manual_has_pathogenic and manual_has_benign):
        return "manual_conflict_or_uncertain", "manual_uncertain_or_conflicting_public_evidence", "manual"
    if label == 1 and manual_has_pathogenic:
        return "manual_public_supports_label", "manual_supports_plp_sentinel", "manual"
    if label == 0 and manual_has_benign:
        return "manual_public_supports_label", "manual_supports_benign_sentinel", "manual"
    if label == 1 and manual_has_benign:
        return "manual_label_conflict", "manual_benign_conflicts_with_plp_label", "manual"
    if label == 0 and manual_has_pathogenic:
        return "manual_label_conflict", "manual_pathogenic_conflicts_with_benign_label", "manual"
    return "manual_review_incomplete", "manual_classification_not_decisive", "low"


def review_question(row: pd.Series) -> str:
    if row["audit_type"] == "false_benign_plp":
        return "Is this ABRAOM-present ClinVar P/LP label valid, founder/recessive, or contradicted by public evidence?"
    return "Is this ABRAOM-common benign label valid, or is the molecular pathogenic signal publicly supported?"


def read_review_queue_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_review_queue(path: Path, priority_tiers: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = read_review_queue_table(path)
    frame = frame.loc[frame["priority_tier"].isin(priority_tiers)].copy()
    for column, default in MANUAL_FIELD_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)
    frame["clinvar_variation_url"] = frame["clinvar_variation_ids"].apply(clinvar_variation_url)
    frame["clinvar_search_url"] = frame.apply(clinvar_search_url, axis=1)
    evidence = frame.apply(classify_curated_evidence, axis=1, result_type="expand")
    frame["public_review_status"] = evidence[0]
    frame["public_evidence_decision"] = evidence[1]
    frame["public_evidence_strength"] = evidence[2]
    frame["review_question"] = frame.apply(review_question, axis=1)
    return frame.sort_values(["priority_tier", "priority_score"], ascending=[True, False]).reset_index(drop=True)


def load_model_predictions(model: str, dataset: str, model_root: Path, m5_v2_dir: Path, m5_v3_dir: Path) -> pd.DataFrame:
    specs = dict(MODEL_SPECS)
    specs["M0"] = {"root": model_root / "m0_nonbr_beatv10_v1_sagemaker", "kind": "raw"}
    specs["M7_dynamic_scrambled"] = {
        "root": model_root / "m7_dynamic_scrambled_nonbr_beatv10_v1_sagemaker",
        "kind": "raw",
    }
    specs["M5_v2_calibrated"] = {"root": m5_v2_dir, "kind": "calibrated"}
    specs["M5_v3_safety"] = {"root": m5_v3_dir, "kind": "calibrated"}
    spec = specs[model]
    path = spec["root"] / f"{dataset}.test.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).copy()
    if spec["kind"] == "raw":
        threshold = read_threshold(spec["root"])
        frame["score"] = frame["probability"].astype(float)
        frame["prediction"] = (frame["score"] >= threshold).astype(int)
        frame["threshold"] = threshold
    else:
        frame["score"] = frame["score_used"].astype(float)
        frame["prediction"] = frame["prediction_used"].astype(int)
        frame["threshold"] = frame["threshold_used"].astype(float)
    return frame[["dataset", "original_index", "label", "score", "threshold", "prediction"]]


def confusion_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return classification_metrics(labels.astype(int), predictions.astype(float), 0.5)


def panel_metrics(
    queue: pd.DataFrame,
    *,
    model_root: Path,
    m5_v2_dir: Path,
    m5_v3_dir: Path,
) -> pd.DataFrame:
    panels = {
        "high_priority_review_queue": queue.index == queue.index,
        "local_public_evidence_available": queue["public_review_status"].ne("needs_public_lookup"),
        "local_public_supports_label": queue["public_review_status"].eq("local_public_supports_label"),
        "curated_or_public_supports_label": queue["public_review_status"].isin(READY_REVIEW_STATUSES),
        "public_label_conflict": queue["public_review_status"].eq("public_label_conflict"),
        "manual_label_conflict": queue["public_review_status"].eq("manual_label_conflict"),
        "manual_review_incomplete": queue["public_review_status"].eq("manual_review_incomplete"),
        "pending_public_lookup": queue["public_review_status"].eq("needs_public_lookup"),
        "plp_high_priority": queue["audit_type"].eq("false_benign_plp"),
        "common_benign_high_priority": queue["audit_type"].eq("false_pathogenic_common_benign"),
    }
    rows: list[dict[str, Any]] = []
    models = ["M0", "M7_dynamic_scrambled", "M5_v2_calibrated", "M5_v3_safety"]
    for panel_name, mask in panels.items():
        part = queue.loc[mask].copy()
        if part.empty:
            continue
        for model in models:
            merged_parts = []
            for dataset, dataset_part in part.groupby("dataset", dropna=False):
                preds = load_model_predictions(model, str(dataset), model_root, m5_v2_dir, m5_v3_dir)
                merged_parts.append(
                    dataset_part[["dataset", "original_index", "label"]].merge(
                        preds[["dataset", "original_index", "score", "threshold", "prediction"]],
                        on=["dataset", "original_index"],
                        how="left",
                        validate="one_to_one",
                    )
                )
            merged = pd.concat(merged_parts, ignore_index=True)
            labels = merged["label"].to_numpy(dtype=int)
            predictions = merged["prediction"].fillna(0).to_numpy(dtype=int)
            metrics = classification_metrics(labels, predictions.astype(float), 0.5)
            rows.append(
                {
                    "panel": panel_name,
                    "model": model,
                    "n": int(len(merged)),
                    "n_positive": int((labels == 1).sum()),
                    "n_negative": int((labels == 0).sum()),
                    "mcc": metrics["mcc"],
                    "recall": metrics["recall"],
                    "specificity": metrics["specificity"],
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
            )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, queue: pd.DataFrame, metrics: pd.DataFrame) -> None:
    counts = queue["public_review_status"].value_counts().to_dict()
    decisions = queue["public_evidence_decision"].value_counts().to_dict()
    p1_counts = queue.groupby(["audit_type", "priority_tier", "public_review_status"], dropna=False).size().reset_index(name="n")
    local_supported = int(queue["public_review_status"].eq("local_public_supports_label").sum())
    manual_supported = int(queue["public_review_status"].eq("manual_public_supports_label").sum())
    conflicts = int(queue["public_review_status"].isin(["public_label_conflict", "manual_label_conflict"]).sum())
    pending = int(queue["public_review_status"].eq("needs_public_lookup").sum())
    lines = [
        "# Public ABRAOM Validation Package",
        "",
        f"Generated at UTC: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Bottom Line",
        "",
        "This package is a reproducible public-evidence review scaffold, not a completed clinical curation.",
        f"Of `{len(queue)}` high-priority variants, `{local_supported}` have local public ClinVar evidence supporting the current label, `{manual_supported}` have completed manual public curation supporting the current label, `{conflicts}` have public/manual evidence conflicting with the current label, and `{pending}` need public lookup.",
        "",
        "## Evidence Status",
        "",
        "| Status | n |",
        "|---|---:|",
    ]
    for status, count in counts.items():
        lines.append(f"| `{status}` | {int(count)} |")
    lines.extend(["", "## Evidence Decisions", "", "| Decision | n |", "|---|---:|"])
    for decision, count in decisions.items():
        lines.append(f"| `{decision}` | {int(count)} |")
    lines.extend(["", "## High-Priority Breakdown", "", "| Audit | Tier | Status | n |", "|---|---|---|---:|"])
    for row in p1_counts.itertuples(index=False):
        lines.append(f"| `{row.audit_type}` | `{row.priority_tier}` | `{row.public_review_status}` | {int(row.n)} |")
    lines.extend(
        [
            "",
            "## Panel Metrics",
            "",
            "| Panel | Model | n | Recall | Specificity | MCC | FP | FN |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| `{row.panel}` | `{row.model}` | {int(row.n)} | {row.recall:.3f} | "
            f"{row.specificity:.3f} | {row.mcc:.3f} | {int(row.fp)} | {int(row.fn)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current blocker is evidence resolution, not architecture.",
            "- Rows marked `public_label_conflict` should be adjudicated before being used as a sentinel truth set.",
            "- Rows marked `needs_public_lookup` require ClinVar/PMID/source entry before any new training or claim of regional specificity.",
            "- This is a stress/error panel, not a prevalence-balanced benchmark; poor metrics here identify failure modes rather than global model quality.",
            "- The ready subset is currently too small to justify a new model decision by itself.",
            "",
            "## Key Artifacts",
            "",
            "- `public_evidence_review_queue.tsv`",
            "- `public_curated_sentinel_panel.csv`",
            "- `public_validation_metrics_by_panel.csv`",
            "- `PUBLIC_ABRAOM_VALIDATION_DECISION_REPORT.md`",
        ]
    )
    (output_dir / "PUBLIC_ABRAOM_VALIDATION_DECISION_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue = load_review_queue(args.review_queue, args.priority_tiers)
    output_columns = [column for column in PUBLIC_QUEUE_COLUMNS if column in queue.columns]
    queue[output_columns].to_csv(args.output_dir / "public_evidence_review_queue.tsv", sep="\t", index=False)
    queue[output_columns].to_csv(args.output_dir / "public_curated_sentinel_panel.csv", index=False)
    queue[output_columns].to_parquet(args.output_dir / "public_curated_sentinel_panel.parquet", index=False)

    ready = queue.loc[queue["public_review_status"].isin(READY_REVIEW_STATUSES), output_columns].copy()
    ready.to_csv(args.output_dir / "public_curated_ready_subset.csv", index=False)
    ready.to_parquet(args.output_dir / "public_curated_ready_subset.parquet", index=False)

    metrics = panel_metrics(
        queue,
        model_root=args.model_root,
        m5_v2_dir=args.m5_v2_dir,
        m5_v3_dir=args.m5_v3_dir,
    )
    metrics.to_csv(args.output_dir / "public_validation_metrics_by_panel.csv", index=False)
    metrics.to_parquet(args.output_dir / "public_validation_metrics_by_panel.parquet", index=False)

    write_report(args.output_dir, queue, metrics)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "high_priority_rows": int(len(queue)),
        "public_review_status": {key: int(value) for key, value in queue["public_review_status"].value_counts().items()},
        "public_evidence_decision": {
            key: int(value) for key, value in queue["public_evidence_decision"].value_counts().items()
        },
        "ready_subset_rows": int(len(ready)),
        "decision": "do_not_train_next_public_evidence_unresolved",
    }
    (args.output_dir / "public_abraom_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
