from __future__ import annotations

import json

import pandas as pd

from scripts.build_brazilian_plp_sentinel_panel import build_panel
from scripts.compile_adapter_fusion_blueprint_completion_report import build_report


def test_build_brazilian_plp_sentinel_panel_without_curated_file(tmp_path) -> None:
    master = tmp_path / "master.parquet"
    pd.DataFrame(
        {
            "variant_key": ["1:10:A:G", "1:20:C:T", "1:30:G:A"],
            "GeneSymbol": ["GENE1", "GENE2", "GENE3"],
            "label": [1, 1, 0],
            "abraom_present": [True, False, True],
            "af_abraom": [0.01, 0.0, 0.02],
        }
    ).to_parquet(master, index=False)

    manifest = build_panel(
        master_path=master,
        curated_public_path=tmp_path / "missing.tsv",
        output_dir=tmp_path / "out",
        overwrite=True,
    )

    assert manifest["rows"]["clinvar_plp_abraom_present"] == 1
    assert manifest["rows"]["sentinel_total"] == 1
    assert (tmp_path / "out" / "curated_public_founder_plp_template.tsv").is_file()


def test_compile_blueprint_completion_report_marks_pending_dynamic(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    rows = []
    for model in ["M0", "M4", "M5", "M6", "M7_scrambled", "M5_v2_calibrated"]:
        for dataset in [
            "br_only",
            "br_any",
            "regional_benchmark_any",
            "abraom_common_benign",
            "abraom_pathogenic_present",
            "abraom_pathogenic_common",
            "global_nonbr_no_abraom",
        ]:
            rows.append({"model": model, "dataset": dataset, "mcc": 0.1, "specificity": 0.9, "recall": 0.8})
    pd.DataFrame(rows).to_csv(baseline, index=False)
    alignment = tmp_path / "ABRAOM_FREQUENCY_ALIGNMENT_REPORT.md"
    alignment.write_text("# ok\n", encoding="utf-8")
    final = tmp_path / "M5_V2.md"
    final.write_text("# ok\n", encoding="utf-8")
    sentinel = tmp_path / "manifest.json"
    sentinel.write_text(json.dumps({"rows": {"sentinel_total": 1}}), encoding="utf-8")

    class Args:
        baseline_csv = baseline
        abraom_alignment_report = alignment
        m5_v2_final_report = final
        sentinel_manifest = sentinel
        dynamic_summary_csv = None

    report = build_report(Args())

    assert report["checklist"]["baseline_ablation_table"]["status"] == "complete"
    assert report["checklist"]["dynamic_gate_experiments"]["status"] == "pending"
    assert report["decision"] == "needs_more_validation"
