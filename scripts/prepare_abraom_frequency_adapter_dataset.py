#!/usr/bin/env python3
"""Prepare the ABRAOM allele-frequency dataset for a regional adapter.

This is the next local artifact after sequence generation. It does not create
synthetic genomes and does not write to S3. It converts the ABRAOM v2 index into
train/val/test variant-level tables with explicit frequency targets and control
targets for the future ``A_BR`` adapter described in the regionalization plan.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_WORKSPACE = Path("/home/sagemaker-user")
DEFAULT_ABRAOM_INDEX = DEFAULT_WORKSPACE / "gen-abraom-seqs/data/production_v2/abraom_index.v2.parquet"
DEFAULT_SPLIT_MANIFEST = DEFAULT_WORKSPACE / "gen-abraom-seqs/data/production_v2/split_manifest.parquet"
DEFAULT_OUTPUT_DIR = Path("data/datasets/abraom_frequency_adapter")

ABRAOM_COLUMNS = ["variant_id", "chrom", "pos", "ref", "alt", "af_abraom", "af_gnomad", "specificity"]
SPLIT_COLUMNS = [
    "block_id",
    "chrom",
    "block_index",
    "block_start",
    "block_end",
    "split",
    "usable_start",
    "usable_end",
    "usable",
]

AF_BINS = [-0.0000001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
AF_LABELS = ["[0,0.005]", "(0.005,0.01]", "(0.01,0.05]", "(0.05,0.1]", "(0.1,0.2]", "(0.2,0.5]", "(0.5,1]"]

SPECIFICITY_BINS = [-0.0000001, 0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
SPECIFICITY_LABELS = [
    "0",
    "(0,0.001]",
    "(0.001,0.005]",
    "(0.005,0.01]",
    "(0.01,0.05]",
    "(0.05,0.1]",
    "(0.1,0.5]",
    "(0.5,1]",
]

OUTPUT_COLUMNS = [
    "variant_id",
    "variant_key",
    "chrom",
    "pos_0_based",
    "Start",
    "ref",
    "alt",
    "allele_pair",
    "af_abraom",
    "af_gnomad",
    "specificity",
    "delta_af",
    "logit_af_abraom",
    "logit_af_gnomad",
    "delta_logit",
    "scrambled_af_abraom",
    "scrambled_logit_af_abraom",
    "scrambled_delta_logit",
    "af_abraom_bin",
    "af_gnomad_bin",
    "specificity_bin",
    "gnomad_zero",
    "sampling_weight_raw",
    "block_id",
    "block_index",
    "block_start",
    "block_end",
    "usable_start",
    "usable_end",
    "inside_usable_region",
    "split",
    "recommended_window_start",
    "recommended_window_end",
    "variant_context_offset",
]


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    text = chrom.removeprefix("chr")
    if text.isdigit():
        return (int(text), "")
    if text == "X":
        return (23, "")
    if text == "Y":
        return (24, "")
    if text in {"M", "MT"}:
        return (25, "")
    return (99, text)


def _require_columns(path: Path, required: Iterable[str]) -> None:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _load_split_manifest(path: Path) -> pd.DataFrame:
    _require_columns(path, SPLIT_COLUMNS)
    manifest = pd.read_parquet(path, columns=SPLIT_COLUMNS)
    manifest = manifest.loc[manifest["split"].isin(["train", "val", "test"])].copy()
    manifest["block_start"] = pd.to_numeric(manifest["block_start"], errors="coerce").astype("int64")
    manifest["block_end"] = pd.to_numeric(manifest["block_end"], errors="coerce").astype("int64")
    manifest["usable_start"] = pd.to_numeric(manifest["usable_start"], errors="coerce").astype("int64")
    manifest["usable_end"] = pd.to_numeric(manifest["usable_end"], errors="coerce").astype("int64")
    manifest["block_index"] = pd.to_numeric(manifest["block_index"], errors="coerce").astype("int64")
    manifest["usable"] = manifest["usable"].astype(bool)
    return manifest.sort_values(["chrom", "block_start"]).reset_index(drop=True)


def _available_chromosomes(manifest: pd.DataFrame, requested: Iterable[str] | None) -> list[str]:
    chromosomes = sorted({str(chrom) for chrom in manifest["chrom"].dropna().unique()}, key=_chrom_sort_key)
    if requested is None:
        return chromosomes
    requested_set = {chrom if chrom.startswith("chr") else f"chr{chrom}" for chrom in requested}
    return [chrom for chrom in chromosomes if chrom in requested_set]


def _logit(values: pd.Series | np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _cut(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(
        "string"
    )


def _assign_blocks(variants: pd.DataFrame, manifest: pd.DataFrame, chrom: str) -> pd.DataFrame:
    blocks = manifest.loc[manifest["chrom"] == chrom].sort_values("block_start")
    if blocks.empty or variants.empty:
        return pd.DataFrame(columns=[*variants.columns, *[column for column in SPLIT_COLUMNS if column != "chrom"]])

    variants = variants.sort_values("pos").reset_index(drop=True)
    assigned = pd.merge_asof(
        variants,
        blocks.drop(columns=["chrom"]).sort_values("block_start"),
        left_on="pos",
        right_on="block_start",
        direction="backward",
    )
    in_block = assigned["block_start"].notna() & (assigned["pos"] >= assigned["block_start"]) & (
        assigned["pos"] < assigned["block_end"]
    )
    return assigned.loc[in_block].reset_index(drop=True)


def _add_derived_columns(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    seq_len: int,
    frequency_epsilon: float,
    specificity_weight_floor: float,
) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"pos": "pos_0_based"})
    out["Start"] = out["pos_0_based"].astype("int64") + 1
    out["ref"] = out["ref"].astype("string").str.upper()
    out["alt"] = out["alt"].astype("string").str.upper()
    out["allele_pair"] = out["ref"] + ">" + out["alt"]
    out["variant_key"] = (
        out["chrom"].astype("string")
        + ":"
        + out["Start"].astype("int64").astype("string")
        + ":"
        + out["ref"].astype("string")
        + ":"
        + out["alt"].astype("string")
    )

    out["af_abraom"] = pd.to_numeric(out["af_abraom"], errors="coerce").astype("float32")
    out["af_gnomad"] = pd.to_numeric(out["af_gnomad"], errors="coerce").fillna(0.0).astype("float32")
    out["specificity"] = pd.to_numeric(out["specificity"], errors="coerce").fillna(0.0).astype("float32")
    out["delta_af"] = (out["af_abraom"] - out["af_gnomad"]).astype("float32")
    out["logit_af_abraom"] = _logit(out["af_abraom"], frequency_epsilon)
    out["logit_af_gnomad"] = _logit(out["af_gnomad"], frequency_epsilon)
    out["delta_logit"] = (out["logit_af_abraom"] - out["logit_af_gnomad"]).astype("float32")

    out["scrambled_af_abraom"] = out["af_abraom"].astype("float32")
    for split in sorted(out["split"].dropna().unique()):
        mask = out["split"] == split
        values = out.loc[mask, "af_abraom"].to_numpy(dtype=np.float32)
        if len(values) > 1:
            out.loc[mask, "scrambled_af_abraom"] = values[rng.permutation(len(values))]
    out["scrambled_logit_af_abraom"] = _logit(out["scrambled_af_abraom"], frequency_epsilon)
    out["scrambled_delta_logit"] = (out["scrambled_logit_af_abraom"] - out["logit_af_gnomad"]).astype("float32")

    out["af_abraom_bin"] = _cut(out["af_abraom"], AF_BINS, AF_LABELS)
    out["af_gnomad_bin"] = _cut(out["af_gnomad"], AF_BINS, AF_LABELS).fillna("[0,0.005]")
    out["specificity_bin"] = _cut(out["specificity"], SPECIFICITY_BINS, SPECIFICITY_LABELS)
    out["gnomad_zero"] = out["af_gnomad"].fillna(0.0).eq(0.0)
    out["sampling_weight_raw"] = np.maximum(out["specificity"].to_numpy(dtype=np.float32), specificity_weight_floor)

    out["inside_usable_region"] = (
        out["usable"].astype(bool)
        & (out["pos_0_based"] >= out["usable_start"])
        & (out["pos_0_based"] < out["usable_end"])
    )
    offset = seq_len // 2
    out["variant_context_offset"] = offset
    out["recommended_window_start"] = np.maximum(out["pos_0_based"].to_numpy(dtype=np.int64) - offset, 0)
    out["recommended_window_end"] = out["recommended_window_start"] + seq_len

    for column in ["block_index", "block_start", "block_end", "usable_start", "usable_end"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("int64")

    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def _empty_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "af_abraom_sum": 0.0,
        "af_gnomad_sum": 0.0,
        "specificity_sum": 0.0,
        "delta_af_sum": 0.0,
        "sampling_weight_raw_sum": 0.0,
        "gnomad_zero": 0,
        "af_abraom_bin_counts": {},
        "specificity_bin_counts": {},
        "chrom_counts": {},
    }


def _increment_count_map(target: dict[str, int], counts: pd.Series) -> None:
    for key, value in counts.items():
        target[str(key)] = int(target.get(str(key), 0) + int(value))


def _update_stats(stats: dict[str, Any], df: pd.DataFrame) -> None:
    if df.empty:
        return
    stats["rows"] += len(df)
    stats["af_abraom_sum"] += float(df["af_abraom"].sum())
    stats["af_gnomad_sum"] += float(df["af_gnomad"].sum())
    stats["specificity_sum"] += float(df["specificity"].sum())
    stats["delta_af_sum"] += float(df["delta_af"].sum())
    stats["sampling_weight_raw_sum"] += float(df["sampling_weight_raw"].sum())
    stats["gnomad_zero"] += int(df["gnomad_zero"].sum())
    _increment_count_map(stats["af_abraom_bin_counts"], df["af_abraom_bin"].value_counts(dropna=False))
    _increment_count_map(stats["specificity_bin_counts"], df["specificity_bin"].value_counts(dropna=False))
    _increment_count_map(stats["chrom_counts"], df["chrom"].value_counts(dropna=False))


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    rows = int(stats["rows"])
    result = {
        "rows": rows,
        "gnomad_zero": int(stats["gnomad_zero"]),
        "af_abraom_bin_counts": dict(sorted(stats["af_abraom_bin_counts"].items())),
        "specificity_bin_counts": dict(sorted(stats["specificity_bin_counts"].items())),
        "chrom_counts": dict(sorted(stats["chrom_counts"].items(), key=lambda item: _chrom_sort_key(item[0]))),
        "sampling_weight_raw_sum": float(stats["sampling_weight_raw_sum"]),
    }
    for column in ["af_abraom", "af_gnomad", "specificity", "delta_af"]:
        result[f"mean_{column}"] = float(stats[f"{column}_sum"] / rows) if rows else None
    return result


def _write_split(
    writers: dict[str, pq.ParquetWriter],
    paths: dict[str, Path],
    split: str,
    df: pd.DataFrame,
) -> None:
    if df.empty:
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    if split not in writers:
        writers[split] = pq.ParquetWriter(paths[split], table.schema, compression="zstd")
    writers[split].write_table(table)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ABRAOM Frequency Adapter Dataset",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Purpose",
        "",
        "Variant-level ABRAOM frequency targets for the future regional adapter (`A_BR`).",
        "The split comes from the existing 1 Mb `split_manifest.parquet`; no S3 writes are performed.",
        "",
        "## Outputs",
        "",
    ]
    for name, output_path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{output_path}`")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Input ABRAOM variants: `{summary['input_rows']}`",
            f"- Written variants: `{summary['written_rows']}`",
            f"- Dropped outside split blocks: `{summary['dropped_no_block']}`",
            f"- Dropped outside usable split region: `{summary['dropped_unusable']}`",
            "",
            "## By Split",
            "",
            "| Split | Rows | Mean AF ABRAOM | Mean AF gnomAD | Mean specificity | gnomAD zero |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, payload in summary["by_split"].items():
        lines.append(
            "| {split} | {rows} | {mean_af_abraom:.6f} | {mean_af_gnomad:.6f} | "
            "{mean_specificity:.6f} | {gnomad_zero} |".format(split=split, **payload)
        )
    path.write_text("\n".join(lines) + "\n")


def prepare_abraom_frequency_adapter_dataset(
    *,
    abraom_index: Path,
    split_manifest: Path,
    output_dir: Path,
    seq_len: int = 4096,
    frequency_epsilon: float = 1e-6,
    specificity_weight_floor: float = 1e-4,
    seed: int = 42,
    chromosomes: tuple[str, ...] | None = None,
    keep_boundary_variants: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    _require_columns(abraom_index, ABRAOM_COLUMNS)
    manifest = _load_split_manifest(split_manifest)
    chromosome_list = _available_chromosomes(manifest, chromosomes)
    rng = np.random.default_rng(seed)

    split_paths = {
        "train": output_dir / "abraom_frequency_train.parquet",
        "val": output_dir / "abraom_frequency_val.parquet",
        "test": output_dir / "abraom_frequency_test.parquet",
    }
    writers: dict[str, pq.ParquetWriter] = {}
    stats_by_split = {split: _empty_stats() for split in split_paths}
    input_rows = pq.ParquetFile(abraom_index).metadata.num_rows
    dropped_no_block = 0
    dropped_unusable = 0

    try:
        for chrom in chromosome_list:
            table = pq.read_table(abraom_index, columns=ABRAOM_COLUMNS, filters=[("chrom", "=", chrom)])
            if table.num_rows == 0:
                continue
            variants = table.to_pandas()
            assigned = _assign_blocks(variants, manifest, chrom)
            dropped_no_block += int(len(variants) - len(assigned))
            if assigned.empty:
                continue

            derived = _add_derived_columns(
                assigned,
                rng=rng,
                seq_len=seq_len,
                frequency_epsilon=frequency_epsilon,
                specificity_weight_floor=specificity_weight_floor,
            )
            if keep_boundary_variants:
                kept = derived
            else:
                kept = derived.loc[derived["inside_usable_region"]].reset_index(drop=True)
                dropped_unusable += int(len(derived) - len(kept))
            if kept.empty:
                continue

            for split, split_df in kept.groupby("split", sort=True):
                if split not in split_paths:
                    continue
                split_df = split_df.reset_index(drop=True)
                _update_stats(stats_by_split[str(split)], split_df)
                _write_split(writers, split_paths, str(split), split_df)
    finally:
        for writer in writers.values():
            writer.close()

    missing_outputs = sorted(set(split_paths) - set(writers))
    if missing_outputs:
        raise RuntimeError(f"No variants were written for split(s): {', '.join(missing_outputs)}")

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "README.md"
    by_split = {split: _finalize_stats(stats) for split, stats in stats_by_split.items()}
    written_rows = sum(payload["rows"] for payload in by_split.values())
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "abraom_index": str(abraom_index),
            "split_manifest": str(split_manifest),
        },
        "outputs": {
            "train": str(split_paths["train"]),
            "val": str(split_paths["val"]),
            "test": str(split_paths["test"]),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "parameters": {
            "seq_len": seq_len,
            "frequency_epsilon": frequency_epsilon,
            "specificity_weight_floor": specificity_weight_floor,
            "seed": seed,
            "keep_boundary_variants": keep_boundary_variants,
            "chromosomes": chromosome_list,
        },
        "input_rows": int(input_rows),
        "written_rows": int(written_rows),
        "dropped_no_block": int(dropped_no_block),
        "dropped_unusable": int(dropped_unusable),
        "by_split": by_split,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_report(report_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abraom-index", type=Path, default=DEFAULT_ABRAOM_INDEX)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--frequency-epsilon", type=float, default=1e-6)
    parser.add_argument("--specificity-weight-floor", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chromosomes", nargs="+")
    parser.add_argument("--keep-boundary-variants", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = prepare_abraom_frequency_adapter_dataset(
        abraom_index=args.abraom_index,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        frequency_epsilon=args.frequency_epsilon,
        specificity_weight_floor=args.specificity_weight_floor,
        seed=args.seed,
        chromosomes=tuple(args.chromosomes) if args.chromosomes else None,
        keep_boundary_variants=args.keep_boundary_variants,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
