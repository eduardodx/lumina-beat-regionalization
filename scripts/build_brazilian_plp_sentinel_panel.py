#!/usr/bin/env python3
"""Build the Brazilian P/LP sensitivity sentinel panel for regional evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MASTER = Path("data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet")
DEFAULT_CURATED_PUBLIC = Path("data/curation/brazilian_founder_plp_public.tsv")
DEFAULT_OUTPUT_DIR = Path("artifacts/clinvar_regional_eval/brazilian_plp_sentinel_panel")

MASTER_COLUMNS = [
    "variant_key",
    "source_variant_id",
    "GeneSymbol",
    "Chromosome",
    "Start",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "label",
    "split_within_gene",
    "variant_type",
    "is_snv",
    "clinvar_regional_cohort",
    "abraom_present",
    "af_abraom",
    "af_gnomad",
    "specificity",
]

CURATED_COLUMNS = [
    "variant_key",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene_symbol",
    "condition",
    "classification",
    "source_name",
    "source_url_or_pmid",
    "evidence_note",
    "curation_status",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinvar-master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--curated-public-tsv", type=Path, default=DEFAULT_CURATED_PUBLIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def read_master_subset(path: Path) -> pd.DataFrame:
    available = set(parquet_columns(path))
    columns = [column for column in MASTER_COLUMNS if column in available]
    required = {"variant_key", "label", "abraom_present"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return pd.read_parquet(path, columns=columns)


def normalize_variant_key(frame: pd.DataFrame) -> pd.Series:
    if "variant_key" in frame:
        raw = frame["variant_key"].fillna("").astype(str).str.strip()
    else:
        raw = pd.Series([""] * len(frame), index=frame.index)
    missing = raw == ""
    coordinate_columns = {"chrom", "pos", "ref", "alt"}
    if missing.any() and coordinate_columns.issubset(frame.columns):
        constructed = (
            frame["chrom"].astype(str).str.replace("^chr", "", regex=True)
            + ":"
            + frame["pos"].astype(str)
            + ":"
            + frame["ref"].astype(str).str.upper()
            + ":"
            + frame["alt"].astype(str).str.upper()
        )
        raw = raw.where(~missing, constructed)
    return raw


def write_curated_template(path: Path) -> None:
    pd.DataFrame(columns=CURATED_COLUMNS).to_csv(path, sep="\t", index=False)


def read_curated_public(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    missing = sorted(set(CURATED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required curated columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["variant_key"] = normalize_variant_key(frame)
    frame["curation_status"] = frame["curation_status"].str.strip().str.lower()
    accepted = frame["curation_status"].isin({"accepted", "include", "included"})
    return frame.loc[accepted].reset_index(drop=True)


def build_panel(*, master_path: Path, curated_public_path: Path, output_dir: Path, overwrite: bool) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    master = read_master_subset(master_path)
    label = pd.to_numeric(master["label"], errors="coerce")
    abraom_present = master["abraom_present"].fillna(False).astype(bool)
    clinvar_panel = master.loc[(label == 1) & abraom_present].copy()
    clinvar_panel["panel_source"] = "clinvar_plp_abraom_present"
    clinvar_panel["curation_source"] = "local_clinvar_abraom_master"
    clinvar_panel["curation_status"] = "accepted"

    curated_available = curated_public_path.is_file()
    curated_rows = 0
    curated_matched_rows = 0
    curated_unmatched_rows = 0
    curated_output = output_dir / "curated_public_founder_plp_panel.parquet"
    unmatched_output = output_dir / "curated_public_unmatched.tsv"
    if curated_available:
        curated = read_curated_public(curated_public_path)
        curated_rows = len(curated)
        master_by_variant = master.drop_duplicates("variant_key")
        curated_matched = curated.merge(
            master_by_variant,
            on="variant_key",
            how="left",
            suffixes=("_curated", ""),
            indicator=True,
        )
        curated_matched_rows = int((curated_matched["_merge"] == "both").sum())
        curated_unmatched = curated_matched.loc[curated_matched["_merge"] != "both", CURATED_COLUMNS].copy()
        curated_unmatched_rows = len(curated_unmatched)
        curated_matched["panel_source"] = "curated_public_brazilian_founder_plp"
        curated_matched["curation_source"] = curated_matched["source_url_or_pmid"]
        curated_matched.drop(columns=["_merge"]).to_parquet(curated_output, index=False)
        curated_unmatched.to_csv(unmatched_output, sep="\t", index=False)
    else:
        empty_columns = list(dict.fromkeys([*CURATED_COLUMNS, *master.columns, "panel_source", "curation_source"]))
        curated_matched = pd.DataFrame(columns=empty_columns)
        write_curated_template(output_dir / "curated_public_founder_plp_template.tsv")

    sentinel = pd.concat(
        [
            clinvar_panel,
            curated_matched.loc[
                curated_matched.get("variant_key", pd.Series(dtype=str)).astype(str).isin(set(master["variant_key"])),
                clinvar_panel.columns.intersection(curated_matched.columns),
            ].reindex(columns=clinvar_panel.columns),
        ],
        ignore_index=True,
    )
    sentinel = sentinel.drop_duplicates(["variant_key", "panel_source"]).reset_index(drop=True)

    clinvar_output = output_dir / "clinvar_plp_abraom_present.parquet"
    sentinel_output = output_dir / "brazilian_plp_sentinel_panel.parquet"
    clinvar_panel.to_parquet(clinvar_output, index=False)
    sentinel.to_parquet(sentinel_output, index=False)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "clinvar_master": str(master_path),
            "clinvar_master_sha256": sha256_file(master_path),
            "curated_public_tsv": str(curated_public_path),
            "curated_public_available": curated_available,
            "curated_public_sha256": sha256_file(curated_public_path) if curated_available else None,
        },
        "outputs": {
            "clinvar_plp_abraom_present": str(clinvar_output),
            "brazilian_plp_sentinel_panel": str(sentinel_output),
            "curated_public_founder_plp_panel": str(curated_output) if curated_available else None,
            "curated_public_unmatched": str(unmatched_output) if curated_available else None,
        },
        "rows": {
            "clinvar_plp_abraom_present": len(clinvar_panel),
            "curated_public_accepted": curated_rows,
            "curated_public_matched_local_master": curated_matched_rows,
            "curated_public_unmatched_local_master": curated_unmatched_rows,
            "sentinel_total": len(sentinel),
        },
        "usage": "Evaluation-only sensitivity panel. Do not train on this panel.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(output_dir / "README.md", manifest)
    return manifest


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    rows = manifest["rows"]
    lines = [
        "# Brazilian P/LP Sentinel Panel",
        "",
        "Evaluation-only sensitivity panel for the ABRAOM regionalization blueprint.",
        "",
        "## Rows",
        "",
        f"- ClinVar P/LP ABRAOM-present: `{rows['clinvar_plp_abraom_present']}`",
        f"- Curated public accepted: `{rows['curated_public_accepted']}`",
        f"- Curated public matched to local master: `{rows['curated_public_matched_local_master']}`",
        f"- Curated public unmatched: `{rows['curated_public_unmatched_local_master']}`",
        f"- Sentinel total: `{rows['sentinel_total']}`",
        "",
        "## Notes",
        "",
        "- This panel is for sensitivity testing only.",
        "- If the curated public TSV is absent, a template is emitted in the output directory.",
        "- Accepted curated rows must include a traceable source URL or PMID.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_panel(
        master_path=args.clinvar_master,
        curated_public_path=args.curated_public_tsv,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
