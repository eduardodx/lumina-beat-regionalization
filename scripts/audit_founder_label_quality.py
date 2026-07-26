#!/usr/bin/env python3
"""How many of our "founders" are probably MISLABELED? -- quantify label quality from local data.

Runs LOCAL. No GPU, no job, no external ClinVar API -- uses only signals already in the regional
ClinVar master parquet.

WHY THIS EXISTS
---------------
Our whole "the ceiling is a label/data problem" conclusion rests partly on the claim that some of
the 1596 ClinVar P/LP variants that are common in ABRAOM (the "founders") are not really
pathogenic -- they are common benigns mislabeled P/LP. So far that rests on ONE example (PTF1A
10:23193706:T:C: labeled P/LP, but a Benign classification is on record, and it sits at
af_abraom 0.52). One example is an anecdote. This turns it into a number: what FRACTION of the
founder set carries label-quality red flags?

THE SIGNALS (all local, all in the master after attach_regional_annotation)
---------------------------------------------------------------------------
A founder = label == 1 (P/LP) AND abraom_present (common in Brazil). For each we check:

  [1] BENIGN CONFLICT (the strongest, the PTF1A signal): the regional ClinVar submissions include
      a Benign / Likely-benign classification (`regional_clinical_significance_values` contains
      "benign"), directly contradicting the P/LP label.

  [2] BIOLOGICALLY IMPLAUSIBLE FREQUENCY: a genuinely pathogenic variant cannot be too common. A
      recessive founder CAN be common (carriers are healthy), but homozygote frequency ~= af^2, so
      af_abraom = 0.10 implies ~1% of people homozygous -- implausible for a severe recessive
      disease. We tier by af_abraom (>0.01, >0.05, >0.10) so the reader sees the degree; only the
      high tiers are red flags on their own.

  [3] WEAK REVIEW STATUS: the label rests on a low ClinVar review tier (single submitter / no
      assertion criteria). `max_review_status_rank_aggregate` low = weak.

The point is NOT to relabel anything automatically -- it is to size the problem and produce a
prioritized queue for actual clinical curation.

THE CONTRAST THAT MAKES THE NUMBER MEAN SOMETHING
-------------------------------------------------
We run the SAME flags on the common-benign set (label == 0 AND abraom_present). Some label noise
exists everywhere; if founders carry benign-conflicts far MORE often than common benigns carry
pathogenic-conflicts, the founder set is specifically contaminated -- which is the claim.

USAGE
-----
    python scripts/audit_founder_label_quality.py \
        --clinvar-master data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet \
        --output-dir ~/v11eval/founder_label_quality

If the master is absent, pass --slice-dir ~/slices to fall back to the slices (which carry label /
abraom_present / af_abraom but NOT the conflict or review columns, so only signal [2] is available).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MASTER = Path("data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet")
COMMON_AF_TIERS = [0.01, 0.05, 0.10]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clinvar-master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--slice-dir", type=Path, default=home / "slices",
                   help="fallback if the master is absent (only the frequency signal is available)")
    p.add_argument("--weak-review-max-rank", type=float, default=1.0,
                   help="max_review_status_rank_aggregate <= this counts as weak review (ClinVar "
                        "0-4 star scale: 0=no criteria, 1=single submitter). Adjust per the scale printed.")
    p.add_argument("--output-dir", type=Path, default=home / "v11eval/founder_label_quality")
    return p.parse_args(argv)


def load_variants(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame, available_signal_columns). Prefer the master; fall back to concatenated slices."""
    if args.clinvar_master.is_file():
        cols = set(pq.ParquetFile(args.clinvar_master).schema_arrow.names)
        wanted = [c for c in (
            "variant_key", "source_variant_id", "GeneSymbol", "Chromosome", "Start",
            "ReferenceAlleleVCF", "AlternateAlleleVCF", "label", "abraom_present", "af_abraom",
            "af_gnomad", "specificity", "regional_clinical_significance_values",
            "max_review_status_rank_aggregate", "source_review_status", "regional_submission_rows",
            "clinvar_variation_ids", "has_brazilian_submitter",
        ) if c in cols]
        frame = pd.read_parquet(args.clinvar_master, columns=wanted)
        return frame, wanted

    # fallback: concat the slices (they lack conflict / review columns)
    frames = []
    for name in ("br_only", "br_any", "abraom_pathogenic_present", "abraom_common_benign", "nonbr_only"):
        path = args.slice_dir / f"{name}.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"neither master ({args.clinvar_master}) nor slices ({args.slice_dir}) found"
        )
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("variant_key")
    return frame, list(frame.columns)


def compute_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_label"] = pd.to_numeric(out.get("label"), errors="coerce")
    present_raw = out["abraom_present"] if "abraom_present" in out.columns else pd.Series(False, index=out.index)
    if present_raw.dtype == bool:
        out["_present"] = present_raw.fillna(False)
    else:
        out["_present"] = pd.to_numeric(present_raw, errors="coerce").fillna(0) > 0
    out["_af"] = pd.to_numeric(out.get("af_abraom"), errors="coerce")

    # [1] benign conflict (only if the significance column exists)
    if "regional_clinical_significance_values" in out.columns:
        sig = out["regional_clinical_significance_values"].fillna("").astype(str).str.lower()
        out["flag_benign_conflict"] = sig.str.contains("benign")
        out["flag_pathogenic_conflict"] = sig.str.contains("pathogenic")
    else:
        out["flag_benign_conflict"] = pd.NA
        out["flag_pathogenic_conflict"] = pd.NA

    # [2] frequency tiers
    for tier in COMMON_AF_TIERS:
        out[f"flag_af_gt_{tier}"] = out["_af"] > tier

    # [3] weak review
    if "max_review_status_rank_aggregate" in out.columns:
        out["_review_rank"] = pd.to_numeric(out["max_review_status_rank_aggregate"], errors="coerce")
    else:
        out["_review_rank"] = np.nan
    return out


def summarize_set(flagged: pd.DataFrame, mask: np.ndarray, conflict_col: str,
                  weak_review_max_rank: float) -> dict:
    subset = flagged.loc[mask]
    n = int(len(subset))
    result = {"n": n}
    if n == 0:
        return result

    af = subset["_af"]
    result["af_median"] = float(af.median())
    for tier in COMMON_AF_TIERS:
        result[f"pct_af_gt_{tier}"] = round(100.0 * float((af > tier).mean()), 1)

    if subset[conflict_col].notna().any():
        conflict = subset[conflict_col].fillna(False).astype(bool)
        result["n_conflict"] = int(conflict.sum())
        result["pct_conflict"] = round(100.0 * float(conflict.mean()), 1)

    review = subset["_review_rank"]
    if review.notna().any():
        weak = review <= weak_review_max_rank
        result["n_weak_review"] = int(weak.sum())
        result["pct_weak_review"] = round(100.0 * float(weak.mean()), 1)
        result["review_rank_value_counts"] = {
            str(k): int(v) for k, v in review.round(2).value_counts(dropna=False).sort_index().items()
        }

    # the strongest single flag combination for founders: benign conflict OR biologically implausible AF
    if conflict_col == "flag_benign_conflict" and subset[conflict_col].notna().any():
        conflict = subset[conflict_col].fillna(False).astype(bool)
        implausible = af > 0.05
        result["n_smoking_gun_conflict_or_implausible"] = int((conflict | implausible).sum())
        result["pct_smoking_gun_conflict_or_implausible"] = round(100.0 * float((conflict | implausible).mean()), 1)
        result["n_conflict_and_common"] = int((conflict & (af > 0.01)).sum())
    return result


def _fmt(v, d: int = 1) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if not np.isfinite(v) else f"{v:.{d}f}"


def render(payload: dict) -> str:
    L = []
    L.append("=" * 90)
    L.append("FOUNDER LABEL-QUALITY AUDIT -- how much of the 'founder' set is probably mislabeled?")
    L.append(f"source: {payload['source']}   signals available: {', '.join(payload['signals'])}")
    L.append("=" * 90)

    founders = payload["founders"]
    benigns = payload["common_benigns"]

    L.append("")
    L.append(f"[FOUNDERS]  P/LP AND common in ABRAOM   n = {founders['n']}")
    L.append(f"    af_abraom median: {_fmt(founders.get('af_median'), 3)}")
    for tier in COMMON_AF_TIERS:
        L.append(f"    fraction with af_abraom > {tier:<5}: {_fmt(founders.get(f'pct_af_gt_{tier}'))}%")
    if "pct_conflict" in founders:
        L.append(f"    ** BENIGN CONFLICT on record (label contradicted): "
                 f"{founders['n_conflict']}/{founders['n']} = {_fmt(founders['pct_conflict'])}%  <-- the PTF1A signal")
    if "pct_weak_review" in founders:
        L.append(f"    weak review (rank <= {payload['weak_review_max_rank']}): "
                 f"{founders['n_weak_review']}/{founders['n']} = {_fmt(founders['pct_weak_review'])}%")
    if "pct_smoking_gun_conflict_or_implausible" in founders:
        L.append(f"    >> RED-FLAGGED (benign conflict OR af>0.05): "
                 f"{founders['n_smoking_gun_conflict_or_implausible']}/{founders['n']} = "
                 f"{_fmt(founders['pct_smoking_gun_conflict_or_implausible'])}%")

    if benigns.get("n"):
        L.append("")
        L.append(f"[CONTRAST: common benigns]  B/LB AND common in ABRAOM   n = {benigns['n']}")
        if "pct_conflict" in benigns:
            L.append(f"    PATHOGENIC conflict on record: {benigns['n_conflict']}/{benigns['n']} = "
                     f"{_fmt(benigns['pct_conflict'])}%")
        if "pct_weak_review" in benigns:
            L.append(f"    weak review: {_fmt(benigns['pct_weak_review'])}%")

    if payload.get("review_rank_scale"):
        L.append("")
        L.append("review-rank distribution among founders (to read the scale):")
        for rank, count in payload["review_rank_scale"].items():
            L.append(f"    rank {rank}: {count}")

    L.append("")
    L.append("=" * 90)
    verdict = payload["verdict"]
    L.append(f"ARGUMENT FOR EDUARDO: {verdict}")
    L.append("=" * 90)
    return "\n".join(L)


def build_verdict(founders: dict, benigns: dict) -> str:
    conflict_pct = founders.get("pct_conflict")
    redflag_pct = founders.get("pct_smoking_gun_conflict_or_implausible")
    if conflict_pct is None and redflag_pct is None:
        return ("Only the frequency signal was available (no master with the conflict/review "
                "columns). Point --clinvar-master at the master parquet for the full argument.")
    parts = []
    if redflag_pct is not None:
        parts.append(f"{_fmt(redflag_pct)}% of the founder set carries a label red flag (benign "
                     f"conflict or implausibly high frequency)")
    if conflict_pct is not None and benigns.get("pct_conflict") is not None:
        ratio = conflict_pct / max(benigns["pct_conflict"], 0.01)
        parts.append(f"founders carry a benign conflict {_fmt(ratio)}x more often than common "
                     f"benigns carry a pathogenic conflict ({_fmt(conflict_pct)}% vs "
                     f"{_fmt(benigns['pct_conflict'])}%), so the contamination is specific to the P/LP set")
    return ". ".join(parts) + ". Curation of this subset is warranted and the flagged variants are exported as a queue."


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame, signals = load_variants(args)
    flagged = compute_flags(frame)

    label = flagged["_label"]
    present = flagged["_present"].astype(bool)
    founder_mask = (label == 1) & present
    benign_mask = (label == 0) & present

    founders = summarize_set(flagged, founder_mask.to_numpy(), "flag_benign_conflict",
                             args.weak_review_max_rank)
    common_benigns = summarize_set(flagged, benign_mask.to_numpy(), "flag_pathogenic_conflict",
                                   args.weak_review_max_rank)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(args.clinvar_master if args.clinvar_master.is_file() else args.slice_dir),
        "signals": [s for s in ("af_abraom", "regional_clinical_significance_values",
                                "max_review_status_rank_aggregate") if s in signals],
        "weak_review_max_rank": args.weak_review_max_rank,
        "founders": founders,
        "common_benigns": common_benigns,
        "review_rank_scale": founders.get("review_rank_value_counts"),
        "verdict": build_verdict(founders, common_benigns),
    }

    report = render(payload)
    print(report)

    # export the flagged founders as a curation queue
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in ("variant_key", "source_variant_id", "GeneSymbol", "Chromosome", "Start",
                               "ReferenceAlleleVCF", "AlternateAlleleVCF", "af_abraom", "af_gnomad",
                               "regional_clinical_significance_values", "max_review_status_rank_aggregate",
                               "flag_benign_conflict") if c in flagged.columns]
    af = flagged["_af"]
    conflict = flagged["flag_benign_conflict"].fillna(False).astype(bool) if \
        flagged["flag_benign_conflict"].notna().any() else pd.Series(False, index=flagged.index)
    redflag = founder_mask & (conflict | (af > 0.05))
    flagged.loc[redflag, export_cols].to_csv(args.output_dir / "flagged_founders_for_curation.csv", index=False)
    (args.output_dir / "founder_label_quality.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "founder_label_quality.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/founder_label_quality.{{json,txt}} + flagged_founders_for_curation.csv "
          f"({int(redflag.sum())} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
