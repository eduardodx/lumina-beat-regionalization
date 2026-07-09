#!/usr/bin/env python3
"""Compile a researcher handoff report for the ABRAOM regionalization study."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd


OUTPUT_DIR = Path("artifacts/clinvar_regional_eval/researcher_transfer_report")
REPORT_BASENAME = "ABRAOM_RESEARCHER_TRANSFER_REPORT"
COMPACT_REPORT_BASENAME = "ABRAOM_RESEARCHER_TRANSFER_REPORT_COMPACT"

ABRAOM_FREQ_SUMMARY = Path("data/datasets/abraom_frequency_adapter/summary.json")
CLINVAR_REGIONAL_SUMMARY = Path("data/datasets/clinvar/regional_abraom/summary.json")
SLICE_SUMMARY = Path("data/datasets/clinvar/regional_abraom/slices/summary.json")
FINAL_MODEL_SUMMARY = Path(
    "artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/"
    "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.csv"
)
DYNAMIC_SUMMARY = Path("artifacts/adapter_fusion_blueprint_completion/dynamic_fusion_regional_summary.csv")
M5_V3_SUMMARY = Path("artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m5_v3_safety_summary.json")
STRONG_CONTROLS = Path(
    "artifacts/clinvar_regional_eval/regional_signal_validation_next_step/strong_negative_control_comparison.csv"
)
REGIONAL_SIGNAL_SUMMARY = Path(
    "artifacts/clinvar_regional_eval/regional_signal_validation_next_step/regional_signal_validation_summary.json"
)
CRITICAL_ERROR_CATEGORIES = Path(
    "artifacts/clinvar_regional_eval/regional_signal_validation_next_step/critical_error_category_summary.csv"
)
CRITICAL_ERROR_GENES = Path(
    "artifacts/clinvar_regional_eval/regional_signal_validation_next_step/critical_error_gene_summary.csv"
)
PUBLIC_VALIDATION_SUMMARY = Path(
    "artifacts/clinvar_regional_eval/public_abraom_validation/public_abraom_validation_summary.json"
)
PUBLIC_VALIDATION_METRICS = Path(
    "artifacts/clinvar_regional_eval/public_abraom_validation/public_validation_metrics_by_panel.csv"
)

S3_REUSE_ARTIFACTS: list[dict[str, str]] = [
    {
        "category": "Dados ABRAOM",
        "artifact": "ABRAOM v2 processado",
        "s3_uri": "s3://ai4bio-lumina/benchmarks/mosaic/data/processed/gen-abraom-seqs/v2/",
        "reuse": "Entrada primária para reconstruir o índice ABRAOM usado no estudo.",
        "source": "artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md",
        "status": "documentado",
    },
    {
        "category": "Dados SageMaker",
        "artifact": "Raiz de dados e saídas do estudo",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/",
        "reuse": "Prefixo guarda datasets, treinos, avaliações e artefatos SageMaker do estudo.",
        "source": "ABRAOM comparative report e scripts SageMaker",
        "status": "documentado",
    },
    {
        "category": "Dados SageMaker",
        "artifact": "Dataset do adapter de frequência ABRAOM",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/abraom_frequency_adapter/",
        "reuse": "Canal `frequency` para treinar/reusar A_BR, A_gnomAD e controle scrambled.",
        "source": "scripts/sagemaker_abraom_frequency_adapter.py",
        "status": "prefixo default do launcher",
    },
    {
        "category": "Dados SageMaker",
        "artifact": "Dataset ClinVar x ABRAOM regional",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/",
        "reuse": "Raiz esperada para master/slices; usar antes de regenerar parquets grandes.",
        "source": "scripts/sagemaker_clinvar_m0.py e scripts/sagemaker_clinvar_fusion.py",
        "status": "prefixo default do launcher",
    },
    {
        "category": "Dados SageMaker",
        "artifact": "Slices regionais ClinVar x ABRAOM",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/slices/",
        "reuse": "Canal `dataset` usado por M0, fusion e avaliação regional.",
        "source": "artifacts/clinvar_regional_m0/M0_RUN_STATUS.md",
        "status": "documentado",
    },
    {
        "category": "Referência",
        "artifact": "Genoma hg38",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/hg38/",
        "reuse": "Canal `reference` dos jobs SageMaker.",
        "source": "artifacts/clinvar_regional_m0/M0_RUN_STATUS.md",
        "status": "documentado",
    },
    {
        "category": "Checkpoint base",
        "artifact": "Lumina BEAT-v10",
        "s3_uri": "s3://ai4bio-lumina/releases/lumina-beat-v10-20260527182934/",
        "reuse": "F0/base checkpoint para M0, adapters e fusion.",
        "source": "artifacts/clinvar_regional_m0/M0_RUN_STATUS.md",
        "status": "documentado",
    },
    {
        "category": "Modelo ClinVar",
        "artifact": "M0 baseline non-BR completo",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/"
            "clinvar-m0-nonbr-beatv10-v1/sagemaker-artifacts/"
            "clinvar-m0-nonbr-beatv10-v1-2e6520-20260621191336/output/model.tar.gz"
        ),
        "reuse": "Modelo molecular geral usado como baseline e init model da fusion.",
        "source": "artifacts/clinvar_regional_m0/M0_RUN_STATUS.md",
        "status": "documentado",
    },
    {
        "category": "Adapter de frequência",
        "artifact": "A_BR ABRAOM balanced",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/"
            "abraom-freq-adapter-abraom-balanced-v1-rerun/sagemaker-artifacts/"
            "abraom-freq-abraom-balanced-v1-reru-dbb2ad-20260617023522/output/"
        ),
        "reuse": "Adapter populacional brasileiro usado em M4/M5 fusion.",
        "source": "scripts/sagemaker_clinvar_fusion.py",
        "status": "prefixo default do launcher",
    },
    {
        "category": "Adapter de frequência",
        "artifact": "A_gnomAD balanced",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/"
            "abraom-freq-adapter-gnomad-balanced-v1/sagemaker-artifacts/"
            "abraom-freq-gnomad-balanced-v1-787c1c-20260621124601/output/"
        ),
        "reuse": "Adapter comparador global usado em M4/M5/M2.",
        "source": "scripts/sagemaker_clinvar_fusion.py",
        "status": "prefixo default do launcher",
    },
    {
        "category": "Adapter de frequência",
        "artifact": "A_scrambled balanced",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/abraom-frequency-adapter/"
            "abraom-freq-adapter-scrambled-balanced-v1/sagemaker-artifacts/"
            "abraom-freq-scrambled-balanced-v1-1b573c-20260616222939/output/"
        ),
        "reuse": "Controle negativo de adapter embaralhado usado em M7.",
        "source": "scripts/sagemaker_clinvar_fusion.py",
        "status": "prefixo default do launcher",
    },
    {
        "category": "Treino ClinVar",
        "artifact": "M6 explicit frequency",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-m0/"
            "clinvar-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/"
        ),
        "reuse": "Controle com features explícitas de frequência, sem adapter populacional.",
        "source": "artifacts/clinvar_regional_m6_m4_parallel_status.md",
        "status": "documentado",
    },
    {
        "category": "Treino ClinVar",
        "artifact": "M4 static fusion",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/"
            "clinvar-m4-staticfusion-nonbr-beatv10-v1-rerun-g52x/sagemaker-artifacts/"
        ),
        "reuse": "Fusion estática sem frequência explícita forte.",
        "source": "artifacts/clinvar_regional_fusion_status.md",
        "status": "documentado",
    },
    {
        "category": "Treino ClinVar",
        "artifact": "M5 static fusion explicit frequency",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/"
            "clinvar-m5-staticfusion-explicitfreq-nonbr-beatv10-v1-g5-4x/sagemaker-artifacts/"
        ),
        "reuse": "Fusion estática com adapters populacionais e frequência explícita.",
        "source": "artifacts/clinvar_regional_fusion_status.md",
        "status": "documentado",
    },
    {
        "category": "Treino ClinVar",
        "artifact": "M4 dynamic gated",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/"
            "clinvar-m4-dynamic-gated-nonbr-beatv10-v1/sagemaker-artifacts/"
        ),
        "reuse": "Fusion dinâmica M4 para completar o blueprint.",
        "source": "artifacts/adapter_fusion_blueprint_completion/DYNAMIC_FUSION_JOB_RUNBOOK.md",
        "status": "documentado",
    },
    {
        "category": "Treino ClinVar",
        "artifact": "M5 dynamic gated bounded",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-fusion/"
            "clinvar-m5-dynamic-gated-bounded-nonbr-beatv10-v1/sagemaker-artifacts/"
        ),
        "reuse": "Fusion dinâmica M5; forte em especificidade, insegura para P/LP sem calibração.",
        "source": "artifacts/adapter_fusion_blueprint_completion/DYNAMIC_FUSION_JOB_RUNBOOK.md",
        "status": "documentado",
    },
    {
        "category": "Avaliação regional",
        "artifact": "M4 regional eval",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/"
            "clinvar-regional-eval-m4-staticfusion-nonbr-beatv10-v1/sagemaker-artifacts/"
        ),
        "reuse": "Saídas SageMaker da avaliação regional M4.",
        "source": "artifacts/clinvar_regional_fusion_status.md",
        "status": "documentado",
    },
    {
        "category": "Avaliação regional",
        "artifact": "M5 regional eval",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/"
            "clinvar-regional-eval-m5-staticfusion-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/"
        ),
        "reuse": "Saídas SageMaker da avaliação regional M5.",
        "source": "artifacts/clinvar_regional_fusion_status.md",
        "status": "documentado",
    },
    {
        "category": "Avaliação regional",
        "artifact": "M6 regional eval",
        "s3_uri": (
            "s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/"
            "clinvar-regional-eval-m6-explicitfreq-nonbr-beatv10-v1/sagemaker-artifacts/"
        ),
        "reuse": "Saídas SageMaker da avaliação regional M6.",
        "source": "artifacts/clinvar_regional_fusion_status.md",
        "status": "documentado",
    },
    {
        "category": "Fine-tuning ABRAOM inicial",
        "artifact": "ABRAOM weighted full",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-weighted-full-8gpu-ff-v1/",
        "reuse": "Experimento inicial de fine-tuning ABRAOM ponderado.",
        "source": "artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md",
        "status": "documentado",
    },
    {
        "category": "Fine-tuning ABRAOM inicial",
        "artifact": "ABRAOM uniform full",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-uniform-full-8gpu-ff-v1/",
        "reuse": "Experimento inicial de fine-tuning ABRAOM uniforme.",
        "source": "artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md",
        "status": "documentado",
    },
    {
        "category": "Fine-tuning ABRAOM inicial",
        "artifact": "ABRAOM wild-only full",
        "s3_uri": "s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beatv10-abraom-wild-only-full-8gpu-ff-v1/",
        "reuse": "Controle inicial wild-only.",
        "source": "artifacts/abraom_regional_eval/ABRAOM_FINE_TUNING_COMPARATIVE_REPORT.md",
        "status": "documentado",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-pandoc", action="store_true")
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{float(value) * 100:.{digits}f}%"


def md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_Sem linhas disponíveis._\n"
    out = [
        "| " + " | ".join(md_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def metric_lookup(frame: pd.DataFrame, model: str, dataset: str, metric: str) -> float:
    rows = frame.loc[(frame["model"] == model) & (frame["dataset"] == dataset)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][metric])


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def github_repo_url() -> str:
    remote_url = git_value(["config", "--get", "remote.origin.url"])
    if remote_url.startswith("git@github.com:"):
        repo_path = remote_url.removeprefix("git@github.com:")
        return f"https://github.com/{repo_path.removesuffix('.git')}"
    if remote_url.startswith("https://github.com/"):
        return remote_url.removesuffix(".git")
    return "unknown"


def github_branch_url(branch: str) -> str:
    repo_url = github_repo_url()
    if repo_url == "unknown" or branch == "unknown":
        return "unknown"
    return f"{repo_url}/tree/{quote(branch, safe='')}"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def human_size(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def model_comparison_table(final_summary: pd.DataFrame) -> str:
    model_notes = {
        "M0": "ClinVar non-BR; baseline molecular sem regionalização",
        "M4": "Static fusion regional sem frequência explícita forte",
        "M5": "Fusion + frequência explícita; bruto",
        "M6": "Frequência explícita alternativa; bruto",
        "M5_calibrated": "M5 com frequência limitada/calibrada v2 inicial",
        "M6_calibrated": "M6 com frequência limitada/calibrada v2 inicial",
        "M7_scrambled": "Controle negativo com ABRAOM embaralhado",
        "M5_v2_calibrated": "Lead calibrado em holdout",
        "M5_v3_safety": "Lead com guarda molecular",
    }
    rows = []
    for model, note in model_notes.items():
        rows.append(
            [
                f"`{model}`",
                note,
                fmt(metric_lookup(final_summary, model, "br_only", "mcc")),
                fmt(metric_lookup(final_summary, model, "abraom_common_benign", "specificity")),
                fmt(metric_lookup(final_summary, model, "abraom_pathogenic_present", "recall")),
                fmt(metric_lookup(final_summary, model, "global_nonbr_no_abraom", "mcc")),
                fmt(metric_lookup(final_summary, model, "global_nonbr_no_abraom", "specificity")),
            ]
        )
    return md_table(
        [
            "Modelo",
            "Papel",
            "br_only MCC",
            "ABRAOM common benign specificity",
            "ABRAOM P/LP present recall",
            "global nonBR MCC",
            "global nonBR specificity",
        ],
        rows,
    )


def dynamic_model_table(dynamic: pd.DataFrame) -> str:
    rows = []
    for model in ["M0", "M2_gnomad_only", "M4_dynamic_gated", "M5_dynamic_gated", "M7_dynamic_scrambled", "M5_v2_calibrated"]:
        rows.append(
            [
                f"`{model}`",
                fmt(metric_lookup(dynamic, model, "br_only", "mcc")),
                fmt(metric_lookup(dynamic, model, "abraom_common_benign", "specificity")),
                fmt(metric_lookup(dynamic, model, "abraom_pathogenic_present", "recall")),
                fmt(metric_lookup(dynamic, model, "global_nonbr_no_abraom", "mcc")),
            ]
        )
    return md_table(
        ["Modelo", "br_only MCC", "ABRAOM benign specificity", "ABRAOM P/LP recall", "global nonBR MCC"],
        rows,
    )


def split_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for split, payload in summary["by_split"].items():
        rows.append(
            [
                split,
                fmt(payload.get("rows")),
                fmt(payload.get("positives", "")),
                fmt(payload.get("negatives", "")),
                fmt(payload.get("snv", "")),
                fmt(payload.get("abraom_present", "")),
                fmt(payload.get("brazilian_submitter", "")),
                fmt(payload.get("mean_af_abraom", ""), 4),
                fmt(payload.get("mean_specificity", ""), 4),
            ]
        )
    return rows


def slice_rows(slice_summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for payload in slice_summary["slices"]:
        rows.append(
            [
                f"`{payload['slice']}`",
                fmt(payload["rows"]),
                fmt(payload["positives"]),
                fmt(payload["negatives"]),
                fmt(payload["abraom_present"]),
                fmt(payload.get("mean_af_abraom"), 4),
                fmt(payload.get("mean_specificity"), 4),
                payload["description"],
            ]
        )
    return rows


def control_rows(controls: pd.DataFrame) -> list[list[Any]]:
    rows = []
    for row in controls.itertuples(index=False):
        rows.append(
            [
                f"`{row.dataset}`",
                f"`{row.metric}`",
                f"`{row.control_mode}`",
                fmt(row.real_value),
                fmt(row.control_mean),
                fmt(row.control_p95),
                fmt(row.empirical_p_control_ge_real, 4),
                pct(row.mean_changed_discount_fraction),
            ]
        )
    return rows


def critical_category_rows(categories: pd.DataFrame) -> list[list[Any]]:
    rows = []
    for row in categories.itertuples(index=False):
        rows.append([f"`{row.audit_type}`", f"`{row.failure_category}`", f"`{row.priority_tier}`", fmt(int(row.n))])
    return rows


def critical_gene_rows(genes: pd.DataFrame, limit: int = 20) -> list[list[Any]]:
    rows = []
    for row in genes.head(limit).itertuples(index=False):
        rows.append(
            [
                f"`{row.audit_type}`",
                f"`{row.GeneSymbol}`",
                fmt(int(row.n)),
                fmt(row.mean_priority),
                fmt(row.max_priority),
                fmt(row.median_af_abraom, 4),
                fmt(row.median_molecular_probability, 4),
            ]
        )
    return rows


def s3_reuse_rows() -> list[list[Any]]:
    rows = []
    for item in S3_REUSE_ARTIFACTS:
        rows.append(
            [
                item["category"],
                item["artifact"],
                f"`{item['s3_uri']}`",
                item["reuse"],
                item["status"],
                f"`{item['source']}`",
            ]
        )
    return rows


def s3_reuse_inventory() -> str:
    return md_table(
        ["Categoria", "Artefato", "S3 URI", "Como reutilizar", "Status", "Fonte local"],
        s3_reuse_rows(),
    )


def compact_s3_reuse_inventory() -> str:
    rows = []
    for item in S3_REUSE_ARTIFACTS:
        rows.append([item["category"], item["artifact"], f"`{item['s3_uri']}`", item["reuse"], item["status"]])
    return md_table(["Categoria", "Artefato", "S3 URI", "Uso", "Status"], rows)


def compact_model_table(final_summary: pd.DataFrame) -> str:
    rows = []
    model_notes = {
        "M0": "baseline molecular geral, sem ABRAOM",
        "M4": "fusion regional conservadora",
        "M5": "fusion + frequência; reduziu FP, mas suprimiu P/LP",
        "M6": "frequência explícita; forte, mas inseguro para P/LP",
        "M7_scrambled": "controle negativo com ABRAOM embaralhado",
        "M5_v2_calibrated": "M5 com desconto regional limitado",
        "M5_v3_safety": "candidato final com guarda molecular",
    }
    for model, note in model_notes.items():
        rows.append(
            [
                f"`{model}`",
                note,
                fmt(metric_lookup(final_summary, model, "br_only", "mcc")),
                fmt(metric_lookup(final_summary, model, "abraom_common_benign", "specificity")),
                fmt(metric_lookup(final_summary, model, "abraom_pathogenic_present", "recall")),
                fmt(metric_lookup(final_summary, model, "global_nonbr_no_abraom", "mcc")),
            ]
        )
    return md_table(
        [
            "Modelo",
            "Leitura curta",
            "br_only MCC",
            "Benignas comuns ABRAOM specificity",
            "P/LP ABRAOM recall",
            "Global nonBR MCC",
        ],
        rows,
    )


def artifact_inventory() -> str:
    paths = [
        ("ABRAOM source index", Path("/home/sagemaker-user/gen-abraom-seqs/data/production_v2")),
        ("ABRAOM frequency adapter dataset", Path("data/datasets/abraom_frequency_adapter")),
        ("ClinVar x ABRAOM regional dataset", Path("data/datasets/clinvar/regional_abraom")),
        ("ABRAOM frequency adapter runs", Path("artifacts/abraom_frequency_adapter")),
        ("ClinVar regional eval/model artifacts", Path("artifacts/clinvar_regional_eval")),
        ("Adapter fusion blueprint completion", Path("artifacts/adapter_fusion_blueprint_completion")),
        ("Deep error analysis", Path("artifacts/adapter_fusion_error_analysis_deep")),
        ("M7 scrambled control analysis", Path("artifacts/adapter_fusion_m7_control_analysis")),
        ("Final comparison tables", Path("artifacts/clinvar_regional_comparison")),
    ]
    rows = []
    for label, path in paths:
        rows.append([label, f"`{path}`", human_size(dir_size_bytes(path)), "exists" if path.exists() else "missing"])
    return md_table(["Item", "Caminho", "Tamanho local aproximado", "Status"], rows)


def generate_markdown() -> str:
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    branch = git_value(["branch", "--show-current"])
    commit = git_value(["rev-parse", "--short", "HEAD"])
    commit_long = git_value(["rev-parse", "HEAD"])
    branch_url = github_branch_url(branch)

    abraom_freq = read_json(ABRAOM_FREQ_SUMMARY)
    clinvar_regional = read_json(CLINVAR_REGIONAL_SUMMARY)
    slices = read_json(SLICE_SUMMARY)
    final_summary = read_csv(FINAL_MODEL_SUMMARY)
    dynamic_summary = read_csv(DYNAMIC_SUMMARY)
    m5_v3 = read_json(M5_V3_SUMMARY)
    regional_signal = read_json(REGIONAL_SIGNAL_SUMMARY)
    strong_controls = read_csv(STRONG_CONTROLS)
    categories = read_csv(CRITICAL_ERROR_CATEGORIES)
    genes = read_csv(CRITICAL_ERROR_GENES)
    public_summary = read_json(PUBLIC_VALIDATION_SUMMARY)
    public_metrics = read_csv(PUBLIC_VALIDATION_METRICS)

    br_m0 = metric_lookup(final_summary, "M0", "br_only", "mcc")
    br_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "br_only", "mcc")
    benign_m0 = metric_lookup(final_summary, "M0", "abraom_common_benign", "specificity")
    benign_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "abraom_common_benign", "specificity")
    plp_m0 = metric_lookup(final_summary, "M0", "abraom_pathogenic_present", "recall")
    plp_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "abraom_pathogenic_present", "recall")
    global_m0 = metric_lookup(final_summary, "M0", "global_nonbr_no_abraom", "mcc")
    global_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "global_nonbr_no_abraom", "mcc")

    public_status_rows = [
        [f"`{key}`", fmt(value)] for key, value in public_summary.get("public_review_status", {}).items()
    ]
    public_decision_rows = [
        [f"`{key}`", fmt(value)] for key, value in public_summary.get("public_evidence_decision", {}).items()
    ]
    stress_rows = []
    for row in public_metrics.loc[public_metrics["panel"].eq("high_priority_review_queue")].itertuples(index=False):
        stress_rows.append(
            [f"`{row.model}`", fmt(int(row.n)), fmt(row.recall), fmt(row.specificity), fmt(row.mcc), fmt(int(row.fp)), fmt(int(row.fn))]
        )

    markdown = f"""---
title: "ABRAOM Regionalization Researcher Transfer Report"
subtitle: "Datasets, artifacts, models, evaluation logic, and handoff instructions"
author: "Lumina ABRAOM/ClinVar regionalization study"
date: "{generated_at}"
github_branch_url: "{branch_url}"
lang: pt-BR
---

# ABRAOM Regionalization Researcher Transfer Report

**Gerado em UTC:** `{generated_at}`

**Repositório:** `/home/sagemaker-user/lumina-ssm`

**Branch:** `{branch}`

**GitHub branch URL:** `{branch_url}`

**Commit curto na geração:** `{commit}`

**Commit completo na geração:** `{commit_long}`

**Escopo:** este relatório transfere o conhecimento técnico e científico acumulado no estudo de regionalização ABRAOM/ClinVar. Ele foi escrito para um pesquisador que precisa continuar, auditar ou reproduzir o trabalho.

**Uso permitido deste pacote:** validação científica, desenvolvimento de método, análise de erro, preparação de curadoria e desenho de próxima rodada experimental.

**Uso não suportado:** decisão clínica, interpretação clínica final de variante, liberação de modelo para uso diagnóstico, ou alegação definitiva de superioridade regional sem curadoria externa.

## 1. Resumo Executivo

O estudo investigou se informação populacional brasileira do ABRAOM pode melhorar a interpretação de variantes ClinVar em contexto brasileiro sem destruir a sensibilidade para variantes P/LP presentes no ABRAOM.

O resultado mais importante é que **há sinal útil de regionalização**, mas ele ainda não está completamente falsificado contra controles negativos estratificados. O melhor candidato operacional até agora é `M5_v3_safety`: uma calibração de `M5_v2` com separação entre score molecular e score regional, mais uma guarda molecular que impede que frequência ABRAOM apague completamente evidência molecular forte.

Principais números no test set:

{md_table(
        ["Métrica", "M0 baseline", "M5_v3_safety", "Interpretação"],
        [
            ["`br_only` MCC", fmt(br_m0), fmt(br_m5v3), "Ganho brasileiro claro em relação ao baseline geral."],
            [
                "`abraom_common_benign` specificity",
                fmt(benign_m0),
                fmt(benign_m5v3),
                "Redução forte de falso positivo em benignas comuns ABRAOM.",
            ],
            [
                "`abraom_pathogenic_present` recall",
                fmt(plp_m0),
                fmt(plp_m5v3),
                "Sensibilidade P/LP recuperada para nível próximo/acima de M0.",
            ],
            [
                "`global_nonbr_no_abraom` MCC",
                fmt(global_m0),
                fmt(global_m5v3),
                "Não degradação global em MCC no candidato safety.",
            ],
        ],
    )}

Decisão científica atual registrada nos artefatos:

- `M5_v3_safety`: `{m5_v3.get("decision", "NA")}`
- Próximo passo de validação: `{regional_signal.get("decision", "NA")}`
- Pacote público ABRAOM: `{public_summary.get("decision", "NA")}`

Conclusão prática: **não é hora de treinar um novo modelo como próximo passo imediato**. O gargalo agora é curadoria/evidência pública das variantes críticas.

## 2. Perguntas Científicas que o Estudo Respondeu

1. **ABRAOM melhora algo em relação ao modelo geral?**
   Sim. O ganho aparece principalmente em `br_only` MCC e na especificidade de variantes benignas comuns no ABRAOM.

2. **O ganho vem de abaixar falsos positivos ou falsos negativos?**
   O ganho inicial veio sobretudo de **reduzir falsos positivos** em variantes benignas comuns no ABRAOM. As versões brutas `M5`/`M6` também reduziram falsos positivos, mas criaram problema grave de falsos benignos em P/LP ABRAOM-presentes.

3. **A frequência ABRAOM estava forte demais?**
   Sim. `M5`/`M6` brutos derrubaram demais o recall de P/LP presentes no ABRAOM. Isso motivou `M5_v2_calibrated` e depois `M5_v3_safety`.

4. **O modelo final resolve tudo?**
   Não. `M5_v3_safety` é o melhor candidato científico atual, mas os controles negativos estratificados mostram que a especificidade biológica do sinal ABRAOM ainda não está completamente demonstrada.

5. **Qual é o gargalo agora?**
   Evidência, não arquitetura. Existem 75 variantes críticas em fila pública de alta prioridade; 70 ainda precisam de lookup público, 4 têm conflito de evidência/rótulo, e apenas 1 está pronta como sentinela pública.

## 3. Mapa de Artefatos e Dados

{artifact_inventory()}

### O que está versionado no Git

O branch versiona código, configs, documentação, relatórios Markdown/HTML, JSONs, CSVs e TSVs leves de curadoria. O commit atual também inclui tabelas de resultado necessárias para auditoria.

### O que não está versionado no Git

Os seguintes itens são intencionalmente excluídos por `.gitignore` e devem ser transferidos por canal de dados, S3, artefact store ou storage compartilhado:

- `data/datasets/**`: datasets parquet grandes.
- `artifacts/**/*.parquet`: predições, painéis e caches binários.
- `artifacts/**/*.pt`, `*.tar.gz`, `*.safetensors`: checkpoints/modelos.
- `artifacts/**/*.pdf`, `*.png`, `*.svg`: figuras e PDFs derivados.
- caches de janela de sequência e diretórios `_extracted`.

Não reverter isso sem uma decisão explícita de gestão de artefatos.

### Artefatos já apontados no S3 para reuso

Antes de regenerar datasets, treinos ou avaliações, verificar os objetos abaixo. Eles são os prefixes/arquivos S3 já documentados nos relatórios, runbooks ou launchers desta rodada.

{s3_reuse_inventory()}

Uso prático:

```bash
aws s3 ls <S3_URI>
aws s3 sync <S3_PREFIX> <local_dir>
aws s3 cp <S3_FILE> <local_file>
```

Notas:

- Para prefixes terminando em `/sagemaker-artifacts/`, procurar subdiretórios de job com `output/model.tar.gz`, `output.tar.gz`, `metrics.json`, `summary.json` ou predições.
- Alguns caminhos são `prefixo default do launcher`: eles vêm do código que lançou/consome os jobs e devem ser validados com `aws s3 ls` antes de assumir disponibilidade.
- Os artefatos leves de decisão continuam no Git; os parquets, checkpoints e outputs SageMaker grandes devem ser recuperados do S3 ou storage compartilhado.
- Se um prefixo S3 existir, **não regenerar** o dataset/job correspondente salvo se a intenção for mudar receita, seed, split ou código.
- Lacuna conhecida: alguns artefatos derivados/calibrados, como M5_v2/M5_v3 e partes das avaliações dinâmicas M2/M7, estão preservados nos artefatos locais/Git, mas o prefixo S3 exato não apareceu nos runbooks locais. Para esses casos, primeiro procurar pelo job no SageMaker/S3; só regenerar se o objeto realmente não existir.

## 4. Origem dos Dados

### 4.1 ABRAOM original

O ABRAOM bruto/indexado usado neste estudo veio de:

`/home/sagemaker-user/gen-abraom-seqs/data/production_v2/`

Arquivos principais:

- `abraom_index.v2.parquet`
- `split_manifest.parquet`
- `release_manifest.json`
- `abraom_index.v2.manifest.json`

O índice ABRAOM v2 contém `17.833.190` variantes de entrada, com colunas como `chrom`, `pos`, `ref`, `alt`, `af_abraom`, `af_gnomad` e `specificity`.

### 4.2 Dataset ABRAOM para adapter de frequência

Script de geração:

`scripts/prepare_abraom_frequency_adapter_dataset.py`

Comando reprodutível padrão:

```bash
uv run python scripts/prepare_abraom_frequency_adapter_dataset.py --overwrite
```

Entradas:

- `{abraom_freq["inputs"]["abraom_index"]}`
- `{abraom_freq["inputs"]["split_manifest"]}`

Saídas:

- `data/datasets/abraom_frequency_adapter/abraom_frequency_train.parquet`
- `data/datasets/abraom_frequency_adapter/abraom_frequency_val.parquet`
- `data/datasets/abraom_frequency_adapter/abraom_frequency_test.parquet`
- `data/datasets/abraom_frequency_adapter/summary.json`
- `data/datasets/abraom_frequency_adapter/README.md`

Resumo:

{md_table(
        ["Campo", "Valor"],
        [
            ["Variantes ABRAOM de entrada", fmt(abraom_freq["input_rows"])],
            ["Variantes escritas", fmt(abraom_freq["written_rows"])],
            ["Removidas por região não usável", fmt(abraom_freq.get("dropped_unusable"))],
            ["Comprimento de contexto recomendado", fmt(abraom_freq["parameters"]["seq_len"])],
            ["Seed", fmt(abraom_freq["parameters"]["seed"])],
            ["Tamanho local aproximado", human_size(dir_size_bytes(Path("data/datasets/abraom_frequency_adapter")))],
        ],
    )}

Por split:

{md_table(
        ["Split", "Rows", "Mean AF ABRAOM", "Mean AF gnomAD", "Mean specificity", "gnomAD zero"],
        [
            [
                split,
                fmt(payload["rows"]),
                fmt(payload["mean_af_abraom"], 4),
                fmt(payload["mean_af_gnomad"], 4),
                fmt(payload["mean_specificity"], 4),
                fmt(payload["gnomad_zero"]),
            ]
            for split, payload in abraom_freq["by_split"].items()
        ],
    )}

Colunas conceituais importantes:

- `af_abraom`: frequência alélica ABRAOM.
- `af_gnomad`: frequência global comparadora.
- `specificity`: sinal de enriquecimento/diferença regional.
- `logit_af_abraom`, `logit_af_gnomad`, `delta_logit`: alvos transformados para treino/diagnóstico.
- `scrambled_af_abraom`: controle negativo para detectar efeito de arquitetura ou vazamento.
- `af_abraom_bin`, `specificity_bin`: bins usados para balanceamento e controle.
- `block_id`, `split`: herdam a separação espacial do `split_manifest`.

### 4.3 Dataset ClinVar x ABRAOM

Script de geração:

`scripts/prepare_regional_clinvar_dataset.py`

Comando reprodutível padrão:

```bash
uv run python scripts/prepare_regional_clinvar_dataset.py --overwrite
```

Entradas:

- ClinVar base: `{clinvar_regional["inputs"]["lumina_clinvar_dir"]}`
- ClinVar regional enriquecido: `{clinvar_regional["inputs"]["regional_clinvar_path"]}`
- ABRAOM v2: `{clinvar_regional["inputs"]["abraom_index"]}`

Saídas:

- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet`
- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_train_test.parquet`
- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_holdout.parquet`
- `data/datasets/clinvar/regional_abraom/regional_annotation_by_variant.parquet`
- `data/datasets/clinvar/regional_abraom/abraom_matches.parquet`
- `data/datasets/clinvar/regional_abraom/summary.json`

Resumo:

{md_table(
        ["Campo", "Valor"],
        [
            ["Linhas totais", fmt(clinvar_regional["rows"])],
            ["Variant keys únicos", fmt(clinvar_regional["unique_variant_keys"])],
            ["Labels B/LB", fmt(clinvar_regional["label_counts"]["0"])],
            ["Labels P/LP", fmt(clinvar_regional["label_counts"]["1"])],
            ["Linhas ClinVar presentes no ABRAOM", fmt(clinvar_regional["abraom"]["matched_rows"])],
            ["Matches no benchmark regional", fmt(clinvar_regional["matched_regional_clinvar_benchmark"])],
            ["Tamanho local aproximado", human_size(dir_size_bytes(Path("data/datasets/clinvar/regional_abraom")))],
        ],
    )}

Por split:

{md_table(
        ["Split", "Rows", "Positives", "Negatives", "SNV", "ABRAOM present", "Brazilian submitter", "Mean AF ABRAOM", "Mean specificity"],
        split_rows(clinvar_regional),
    )}

Notas críticas:

- Submitter brasileiro é usado para **estratificação e avaliação**, não como feature de inferência.
- Presença no ABRAOM é evidência populacional, não rótulo clínico.
- Uma variante P/LP presente no ABRAOM pode ser real, founder, recessiva, de penetrância variável, mal classificada ou dependente de contexto de doença. Não use a presença no ABRAOM como regra automática de benignidade.

### 4.4 Slices de avaliação regional

Script:

`scripts/build_regional_clinvar_eval_slices.py`

Comando:

```bash
uv run python scripts/build_regional_clinvar_eval_slices.py --overwrite
```

Parâmetros:

- `common_af_threshold = {slices["parameters"]["common_af_threshold"]}`
- `high_specificity_threshold = {slices["parameters"]["high_specificity_threshold"]}`

Slices:

{md_table(
        ["Slice", "Rows", "P/LP", "B/LB", "ABRAOM present", "Mean AF ABRAOM", "Mean specificity", "Uso"],
        slice_rows(slices),
    )}

Os slices mais importantes para decisão foram:

- `br_only`: principal métrica brasileira.
- `abraom_common_benign`: mede redução de falso positivo em variantes benignas comuns no ABRAOM.
- `abraom_pathogenic_present`: mede risco de falso benigno em P/LP presentes no ABRAOM.
- `global_nonbr_no_abraom`: mede não degradação global fora da regionalização.

## 5. Arquitetura e Famílias de Modelos

O blueprint original recomendava modularidade:

```text
F0       = modelo base congelado ou parcialmente adaptado
A_BR     = adapter populacional brasileiro treinado com frequência ABRAOM
A_gnomAD = adapter populacional comparador global
A_path   = adapter/head de patogenicidade ClinVar
G        = fusion controller / gate
H_mol    = molecular_effect_score
H_reg    = region_calibrated_pathogenicity_score
```

As versões treinadas/avaliadas neste estudo podem ser lidas assim:

- `M0`: baseline ClinVar non-BR, sem ABRAOM. Serve como modelo molecular geral.
- `M4`: adapter/fusion regional sem frequência explícita forte.
- `M5`: fusion com frequência explícita ABRAOM/gnomAD; bruto.
- `M6`: frequência explícita alternativa; bruto.
- `M5_calibrated`/`M6_calibrated`: versões com desconto de frequência limitado.
- `M7_scrambled`: controle negativo com ABRAOM embaralhado.
- `M2_gnomad_only`: controle populacional usando gnomAD.
- `M5_dynamic_gated`: fusion dinâmica; forte em especificidade, insegura em P/LP.
- `M5_v2_calibrated`: candidato calibrado em holdout.
- `M5_v3_safety`: candidato safety com guarda molecular.

## 6. Resultados Principais

### 6.1 Comparação final M0/M4/M5/M6/M7/M5_v2/M5_v3

{model_comparison_table(final_summary)}

Interpretação:

- `M5` e `M6` brutos mostraram que frequência ABRAOM reduz falsos positivos, mas suprimiram demais P/LP.
- `M5_v2_calibrated` recuperou recall P/LP e manteve especificidade ABRAOM-common >= 0.95.
- `M5_v3_safety` preservou o comportamento de `M5_v2` e restaurou `global_nonbr_no_abraom` MCC para o nível de `M0`.

### 6.2 Fusion dinâmica

{dynamic_model_table(dynamic_summary)}

Ponto essencial: `M5_dynamic_gated` foi excelente para reduzir falso positivo em benignas comuns ABRAOM, mas colapsou o recall em P/LP ABRAOM-presentes. Isso mostrou que um gate dinâmico bruto não é suficiente; é necessário limitar o impacto da frequência e proteger evidência molecular forte.

### 6.3 Configuração M5_v3 safety

Configuração selecionada:

```json
{json.dumps(m5_v3.get("selected_config", {}), indent=2)}
```

Interpretação da guarda molecular:

- O modelo mantém dois conceitos separados: `molecular_score` e `regional_score`.
- A frequência regional pode reduzir o score, mas não deve apagar completamente evidência molecular forte.
- A guarda molecular é uma regra de segurança contra falso benigno em variantes P/LP presentes no ABRAOM.

## 7. Controles Negativos e Falsificação

O estudo não deve ser apresentado como "ABRAOM provado biologicamente" sem nuance. Os controles negativos foram desenhados para preservar estruturas como gene, AF bin, cromossomo, tipo de variante e specificity bin enquanto quebram a associação real variante-frequência.

Tabela dos controles estratificados fortes:

{md_table(
        ["Dataset", "Metric", "Controle", "Real", "Média controle", "P95 controle", "P(controle >= real)", "Discount alterado"],
        control_rows(strong_controls),
    )}

Leitura científica:

- Em `br_only`, o real fica acima de muitos controles, mas alguns controles estratificados chegam muito perto.
- Em `abraom_common_benign`, controles dentro de AF/gene frequentemente igualam ou superam o real, indicando que parte do ganho pode vir de estrutura de frequência/benchmark.
- Em `abraom_pathogenic_present`, o recall real não supera de forma robusta vários controles.
- Portanto, o ganho é útil para calibração regional, mas a especificidade biológica do ABRAOM ainda precisa de curadoria externa e validação independente.

## 8. Análise de Erro e Painel de Curadoria

Resumo registrado:

- P/LP ABRAOM-presentes falso benignos remanescentes: `92`.
- P/LP falso benignos de alta prioridade: `60`.
- ABRAOM-common benignas falso patogênicas remanescentes: `502`.
- Painel derivado review/sentinel: `757` linhas.
- Fila pública de alta prioridade: `{public_summary.get("high_priority_rows")}` variantes.

Categorias críticas:

{md_table(["Audit type", "Failure category", "Priority tier", "n"], critical_category_rows(categories))}

Top genes por prioridade:

{md_table(
        ["Audit type", "Gene", "n", "Mean priority", "Max priority", "Median AF ABRAOM", "Median molecular probability"],
        critical_gene_rows(genes),
    )}

Status da curadoria pública:

{md_table(["Status", "n"], public_status_rows)}

Decisões públicas:

{md_table(["Decision", "n"], public_decision_rows)}

Painel de estresse de alta prioridade:

{md_table(["Modelo", "n", "Recall", "Specificity", "MCC", "FP", "FN"], stress_rows)}

Importante: esse painel é construído a partir de erros de alta prioridade. Ele é um **painel de estresse**, não um benchmark populacional balanceado. Métricas ruins nele identificam falhas; não representam desempenho global.

## 9. Como Reproduzir ou Continuar o Trabalho

### 9.1 Ambiente básico

```bash
cd /home/sagemaker-user/lumina-ssm
git switch {branch}
uv sync
```

Para rodar testes relevantes:

```bash
uv run pytest \\
  tests/test_prepare_abraom_frequency_adapter_dataset.py \\
  tests/test_prepare_regional_clinvar_dataset.py \\
  tests/test_build_regional_clinvar_eval_slices.py \\
  tests/test_train_abraom_frequency_adapter.py \\
  tests/test_eval_clinvar_fusion_lora.py \\
  tests/test_calibrate_m5_v3_safety.py \\
  tests/test_validate_regional_signal_next_step.py \\
  tests/test_build_public_abraom_validation_package.py
```

### 9.2 Regenerar datasets principais

ABRAOM frequency adapter:

```bash
uv run python scripts/prepare_abraom_frequency_adapter_dataset.py --overwrite
```

ClinVar x ABRAOM:

```bash
uv run python scripts/prepare_regional_clinvar_dataset.py --overwrite
```

Slices regionais:

```bash
uv run python scripts/build_regional_clinvar_eval_slices.py --overwrite
```

### 9.3 Treinar ou avaliar adapter de frequência

Treino local/smoke:

```bash
uv run python scripts/train_abraom_frequency_adapter.py \\
  --data-dir data/datasets/abraom_frequency_adapter \\
  --output-dir artifacts/abraom_frequency_adapter/smoke-local \\
  --max-train-rows 1000 \\
  --max-val-rows 500 \\
  --max-test-rows 500 \\
  --max-steps 10 \\
  --overwrite
```

Para produção, usar os scripts SageMaker:

- `scripts/sagemaker_abraom_frequency_adapter.py`
- `scripts/abraom_frequency_adapter_job.py`

### 9.4 Avaliação regional ClinVar

Scripts relevantes:

- `scripts/clinvar_m0_job.py`
- `scripts/clinvar_fusion_job.py`
- `scripts/clinvar_regional_eval_job.py`
- `scripts/sagemaker_clinvar_m0.py`
- `scripts/sagemaker_clinvar_fusion.py`
- `scripts/sagemaker_clinvar_regional_eval.py`

Calibração M5_v2:

```bash
uv run python scripts/calibrate_m5_v2_regional_scores.py
```

Calibração M5_v3 safety:

```bash
uv run python scripts/calibrate_m5_v3_safety.py
```

Validação de sinal e análise de erro:

```bash
uv run python scripts/validate_regional_signal_next_step.py
```

Pacote público de curadoria:

```bash
uv run python scripts/build_public_abraom_validation_package.py
```

### 9.5 Gerar relatórios

Relatório HTML executivo/técnico:

```bash
uv run python scripts/compile_abraom_study_html_report.py
```

Este relatório de transferência:

```bash
uv run python scripts/compile_abraom_researcher_transfer_report.py
```

## 10. Como Usar os Artefatos na Prática

### Para revisar resultados rapidamente

Abrir:

- `artifacts/clinvar_regional_eval/abraom_study_report/abraom_regionalization_study.html`
- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/M5_V3_SAFETY_REPORT.md`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/REGIONAL_SIGNAL_VALIDATION_NEXT_STEP_REPORT.md`
- `artifacts/clinvar_regional_eval/public_abraom_validation/PUBLIC_ABRAOM_VALIDATION_DECISION_REPORT.md`

### Para usar datasets em análise

Usar:

- `data/datasets/abraom_frequency_adapter/*.parquet` para tarefas de frequência/adapters.
- `data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet` para análise geral ClinVar x ABRAOM.
- `data/datasets/clinvar/regional_abraom/slices/*.parquet` para avaliação regional padronizada.

### Para continuar curadoria

Abrir e preencher:

`artifacts/clinvar_regional_eval/public_abraom_validation/public_evidence_review_queue.tsv`

Campos a preencher:

- `manual_curation_status`
- `manual_public_classification`
- `manual_public_source_url_or_pmid`
- `manual_evidence_note`

Critério de inclusão no painel sentinela:

- Deve haver classificação manual clara.
- Deve haver fonte pública ou PMID.
- Conflitos devem ser marcados como conflito, não forçados para P/LP ou benigno.
- Variantes sem fonte permanecem como `manual_review_incomplete` ou `needs_public_lookup`.

## 11. Recomendações para o Próximo Pesquisador

Prioridade 1: curadoria pública das 75 variantes críticas.

1. Resolver primeiro as 4 variantes `public_label_conflict`.
2. Revisar as 2 P0 `needs_public_lookup`.
3. Revisar as 53 P1 P/LP `needs_public_lookup`.
4. Revisar as 15 benignas comuns P1 `needs_public_lookup`.
5. Rerodar `build_public_abraom_validation_package.py`.

Prioridade 2: transformar a fila em painel sentinela real.

- O painel atual é scaffold/review queue.
- O painel pronto atual tem apenas `{public_summary.get("ready_subset_rows")}` linha(s), insuficiente para decisão de modelo.
- A próxima decisão de treino deve esperar um painel público/manual suficientemente resolvido.

Prioridade 3: validação externa.

- Ideal: coorte clínica brasileira com variante causal/diagnóstico.
- Alternativa mínima: painel curado de founder/P/LP brasileiros e benignas comuns validadas.
- Sem isso, não usar métricas ABRAOM como prova clínica final.

## 12. Principais Riscos de Interpretação

1. **ABRAOM não é truth set clínico.**
   ABRAOM é frequência populacional, não rótulo de benignidade.

2. **Submitter brasileiro não prova origem populacional brasileira.**
   Submitter country é metadado de submissão; deve ser usado para split/estrato, não como feature clínica.

3. **P/LP presente em ABRAOM não é automaticamente erro.**
   Pode ser founder, recessivo, penetrância reduzida, classificação antiga, ou problema de contexto de doença.

4. **Controle negativo forte importa.**
   Se controle preservando gene/AF/type iguala o real, a alegação ABRAOM-specific fica fraca.

5. **Não otimizar só especificidade.**
   As versões brutas provaram que é fácil reduzir falso positivo destruindo recall P/LP.

## 13. Checklist de Entrega para Outro Pesquisador

Entregar:

- Branch/commit de geração: `{branch}` / `{commit_long}`.
- Este relatório em Markdown/HTML/DOCX.
- Manifesto S3: `artifacts/clinvar_regional_eval/researcher_transfer_report/s3_reuse_manifest.json`.
- `artifacts/clinvar_regional_eval/abraom_study_report/abraom_regionalization_study.html`.
- Diretório `data/datasets/abraom_frequency_adapter/`.
- Diretório `data/datasets/clinvar/regional_abraom/`.
- Diretórios de artefatos leves já versionados.
- Modelos/checkpoints/parquets grandes por storage externo, se forem necessários para reprodução exata.

Não esquecer:

- Confirmar disponibilidade de `/home/sagemaker-user/gen-abraom-seqs/data/production_v2/` ou transferir esse diretório.
- Confirmar disponibilidade dos prefixes listados no manifesto S3 antes de regenerar qualquer dataset ou job SageMaker.
- Confirmar disponibilidade dos dados ClinVar originais de `/home/sagemaker-user/lumina` e `/home/sagemaker-user/lumina-benchmarks` se o pesquisador for regenerar datasets do zero.
- Documentar credenciais/S3/SageMaker separadamente se ele for relançar jobs.

## 14. Apêndice: Arquivos-Chave

Scripts principais:

- `scripts/prepare_abraom_frequency_adapter_dataset.py`
- `scripts/train_abraom_frequency_adapter.py`
- `scripts/prepare_regional_clinvar_dataset.py`
- `scripts/build_regional_clinvar_eval_slices.py`
- `scripts/calibrate_m5_v2_regional_scores.py`
- `scripts/calibrate_m5_v3_safety.py`
- `scripts/validate_regional_signal_next_step.py`
- `scripts/build_public_abraom_validation_package.py`
- `scripts/compile_abraom_study_html_report.py`
- `scripts/compile_abraom_researcher_transfer_report.py`

Documentos e relatórios:

- `artifacts/adapter_fusion_regionalization_blueprint.txt`
- `artifacts/adapter_fusion_blueprint_completion/ABRAOM_ADAPTER_FUSION_BLUEPRINT_COMPLETION_REPORT.md`
- `artifacts/adapter_fusion_error_analysis_deep/DEEP_ERROR_ANALYSIS_REPORT.md`
- `artifacts/adapter_fusion_m7_control_analysis/M7_SCRAMBLED_CONTROL_REPORT.md`
- `artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/M5_V2_CALIBRATED_REPORT.md`
- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/M5_V3_SAFETY_REPORT.md`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/REGIONAL_SIGNAL_VALIDATION_NEXT_STEP_REPORT.md`
- `artifacts/clinvar_regional_eval/public_abraom_validation/PUBLIC_ABRAOM_VALIDATION_DECISION_REPORT.md`

Tabelas de decisão:

- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.csv`
- `artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m5_v3_negative_control_comparison.csv`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/strong_negative_control_comparison.csv`
- `artifacts/clinvar_regional_eval/regional_signal_validation_next_step/critical_error_audit/combined_manual_review_queue.csv`
- `artifacts/clinvar_regional_eval/public_abraom_validation/public_evidence_review_queue.tsv`

## 15. Estado Final do Estudo

O blueprint foi explorado de forma ampla: datasets, ABRAOM adapter, gnomAD/scrambled controls, M0 baseline, M4/M5/M6, M7 scrambled, M5_v2 calibrado, M5_v3 safety, controles negativos estratificados e análise profunda de erro.

A estrutura mais promissora é `M5_v3_safety`, porque combina:

- ganho brasileiro (`br_only` MCC),
- redução de falso positivo em benignas comuns ABRAOM,
- proteção parcial/recuperação de P/LP ABRAOM-presentes,
- não degradação global em MCC,
- separação audível entre evidência molecular e regional.

Mas a conclusão final é deliberadamente conservadora: **o próximo avanço depende de curadoria e validação externa, não de mais uma rodada imediata de treino.**
"""
    return markdown


def generate_compact_markdown() -> str:
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    branch = git_value(["branch", "--show-current"])
    commit_long = git_value(["rev-parse", "HEAD"])
    branch_url = github_branch_url(branch)

    abraom_freq = read_json(ABRAOM_FREQ_SUMMARY)
    clinvar_regional = read_json(CLINVAR_REGIONAL_SUMMARY)
    slices = read_json(SLICE_SUMMARY)
    final_summary = read_csv(FINAL_MODEL_SUMMARY)
    dynamic_summary = read_csv(DYNAMIC_SUMMARY)
    m5_v3 = read_json(M5_V3_SUMMARY)
    regional_signal = read_json(REGIONAL_SIGNAL_SUMMARY)
    public_summary = read_json(PUBLIC_VALIDATION_SUMMARY)
    public_metrics = read_csv(PUBLIC_VALIDATION_METRICS)

    br_m0 = metric_lookup(final_summary, "M0", "br_only", "mcc")
    br_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "br_only", "mcc")
    benign_m0 = metric_lookup(final_summary, "M0", "abraom_common_benign", "specificity")
    benign_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "abraom_common_benign", "specificity")
    plp_m0 = metric_lookup(final_summary, "M0", "abraom_pathogenic_present", "recall")
    plp_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "abraom_pathogenic_present", "recall")
    global_m0 = metric_lookup(final_summary, "M0", "global_nonbr_no_abraom", "mcc")
    global_m5v3 = metric_lookup(final_summary, "M5_v3_safety", "global_nonbr_no_abraom", "mcc")

    dynamic_rows = [
        [
            "`M5_dynamic_gated`",
            fmt(metric_lookup(dynamic_summary, "M5_dynamic_gated", "br_only", "mcc")),
            fmt(metric_lookup(dynamic_summary, "M5_dynamic_gated", "abraom_common_benign", "specificity")),
            fmt(metric_lookup(dynamic_summary, "M5_dynamic_gated", "abraom_pathogenic_present", "recall")),
            "não selecionado: colapso de P/LP",
        ],
        [
            "`M7_dynamic_scrambled`",
            fmt(metric_lookup(dynamic_summary, "M7_dynamic_scrambled", "br_only", "mcc")),
            fmt(metric_lookup(dynamic_summary, "M7_dynamic_scrambled", "abraom_common_benign", "specificity")),
            fmt(metric_lookup(dynamic_summary, "M7_dynamic_scrambled", "abraom_pathogenic_present", "recall")),
            "controle negativo; não superou ABRAOM real",
        ],
    ]

    slice_by_name = {payload["slice"]: payload for payload in slices["slices"]}
    compact_slice_rows = []
    for name, use in [
        ("br_only", "ganho no subconjunto brasileiro"),
        ("abraom_common_benign", "redução de falso positivo"),
        ("abraom_pathogenic_present", "proteção contra falso benigno P/LP"),
        ("global_nonbr_no_abraom", "não degradação global"),
    ]:
        payload = slice_by_name.get(name)
        if payload is None:
            continue
        compact_slice_rows.append(
            [
                f"`{name}`",
                fmt(payload["rows"]),
                fmt(payload["positives"]),
                fmt(payload["negatives"]),
                fmt(payload["abraom_present"]),
                use,
            ]
        )

    stress_rows = []
    for panel, model, note in [
        ("high_priority_review_queue", "M0", "baseline no painel de stress"),
        ("high_priority_review_queue", "M5_v3_safety", "alerta: colapso no painel crítico"),
        ("plp_high_priority", "M0", "P/LP críticos no baseline"),
        ("plp_high_priority", "M5_v3_safety", "alerta: 60 falsos benignos"),
        ("common_benign_high_priority", "M0", "benignas comuns críticas no baseline"),
        ("common_benign_high_priority", "M5_v3_safety", "alerta: 15 falsos positivos"),
    ]:
        rows = public_metrics.loc[(public_metrics["panel"] == panel) & (public_metrics["model"] == model)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        stress_rows.append(
            [
                f"`{panel}`",
                f"`{model}`",
                fmt(int(row["n"])),
                fmt(row["recall"]),
                fmt(row["specificity"]),
                fmt(row["mcc"]),
                fmt(int(row["fp"])),
                fmt(int(row["fn"])),
                note,
            ]
        )

    markdown = f"""---
title: "ABRAOM Regionalization Compact Handoff"
subtitle: "Resumo técnico, artefatos S3 e próximos passos"
author: "Lumina ABRAOM/ClinVar regionalization study"
date: "{generated_at}"
github_branch_url: "{branch_url}"
lang: pt-BR
---

# ABRAOM Regionalization Compact Handoff

**Gerado em UTC:** `{generated_at}`

**Repositório:** `/home/sagemaker-user/lumina-ssm`

**Branch:** `{branch}`

**GitHub branch URL:** `{branch_url}`

**Commit na geração:** `{commit_long}`

**Versão extensa para auditoria:** `artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md`

## 1. Conclusão Curta

O estudo encontrou **sinal operacional útil nos benchmarks regionais**, principalmente por reduzir falso positivo em variantes benignas comuns no ABRAOM. Isso ainda não prova um sinal biológico específico do ABRAOM. O melhor candidato atual é `M5_v3_safety`, porque combina o sinal regional com uma guarda molecular: a frequência ABRAOM pode reduzir o score de patogenicidade, mas não pode apagar completamente evidência molecular forte.

O resultado ainda **não é prova clínica final**. Os controles negativos estratificados mostram que parte do ganho pode vir de estrutura de frequência, gene ou benchmark. A próxima etapa científica deve ser curadoria pública/externa das variantes críticas, não outro treino imediato.

## 2. Principais Resultados

{md_table(
        ["Métrica", "M0", "M5_v3_safety", "Leitura"],
        [
            ["`br_only` MCC", fmt(br_m0), fmt(br_m5v3), "ganho brasileiro claro"],
            ["`abraom_common_benign` specificity", fmt(benign_m0), fmt(benign_m5v3), "menos falso positivo benigno"],
            ["`abraom_pathogenic_present` recall", fmt(plp_m0), fmt(plp_m5v3), "sensibilidade P/LP recuperada"],
            ["`global_nonbr_no_abraom` MCC", fmt(global_m0), fmt(global_m5v3), "não degradação global em MCC"],
        ],
    )}

Leitura prática:

- `M5` e `M6` brutos reduziram falsos positivos, mas criaram risco forte de falso benigno em P/LP presentes no ABRAOM.
- `M5_v2_calibrated` limitou o impacto da frequência.
- `M5_v3_safety` manteve o ganho regional e adicionou proteção molecular.
- `M7_scrambled` foi usado como controle negativo; ele não deve ser lido como modelo candidato principal.

Alertas no painel crítico de curadoria:

{md_table(
        ["Panel", "Modelo", "n", "Recall", "Specificity", "MCC", "FP", "FN", "Leitura"],
        stress_rows,
    )}

Esse painel é um **stress test derivado de erros**, não um benchmark populacional balanceado. Ele mostra onde a hipótese ainda falha e por que a próxima etapa é curadoria, não treino imediato.

## 3. Dados Essenciais

{md_table(
        ["Item", "Valor"],
        [
            ["ABRAOM v2 de entrada", "17.833.190 variantes"],
            ["Dataset adapter de frequência", f"{fmt(abraom_freq['written_rows'])} linhas escritas"],
            ["ClinVar x ABRAOM", f"{fmt(clinvar_regional['rows'])} linhas; {fmt(clinvar_regional['unique_variant_keys'])} variant keys únicos"],
            ["ClinVar presentes no ABRAOM", fmt(clinvar_regional["abraom"]["matched_rows"])],
            ["Fila crítica pública", f"{fmt(public_summary.get('high_priority_rows'))} variantes"],
            ["Painel público pronto", f"{fmt(public_summary.get('ready_subset_rows'))} variante(s)"],
        ],
    )}

Slices decisivos:

{md_table(["Slice", "Rows", "P/LP", "B/LB", "ABRAOM present", "Uso"], compact_slice_rows)}

## 4. O que foi Treinado e Comparado

{compact_model_table(final_summary)}

Fusion dinâmica:

{md_table(["Modelo", "br_only MCC", "ABRAOM benign specificity", "ABRAOM P/LP recall", "Decisão"], dynamic_rows)}

Receita científica:

1. Treinar/usar `M0` como baseline molecular ClinVar non-BR.
2. Treinar adapters de frequência `A_BR`, `A_gnomAD` e `A_scrambled`.
3. Comparar M4/M5/M6 contra M0 nos mesmos slices regionais.
4. Usar M7/scrambled e controles estratificados para falsificação.
5. Calibrar M5 para limitar desconto por frequência.
6. Selecionar `M5_v3_safety` como melhor candidato experimental, com guarda molecular.

## 5. Artefatos S3 para Não Regenerar

Esta seção é crítica. Antes de rodar qualquer script pesado, verificar estes caminhos. Se o objeto existir e a receita não mudou, reutilizar o artefato.

{compact_s3_reuse_inventory()}

Comandos mínimos:

```bash
aws s3 ls <S3_URI>
aws s3 sync <S3_PREFIX> <local_dir>
aws s3 cp <S3_FILE> <local_file>
```

Manifesto legível por máquina:

`artifacts/clinvar_regional_eval/researcher_transfer_report/s3_reuse_manifest.json`

Lacuna conhecida: alguns derivados/calibrados, especialmente `M5_v2`, `M5_v3` e partes das avaliações dinâmicas `M2/M7`, estão preservados nos artefatos locais/Git, mas não tinham prefixo S3 exato nos runbooks locais. Para esses casos, procurar primeiro no SageMaker/S3; só regenerar se o objeto realmente não existir.

## 6. Onde Estão os Arquivos Locais Importantes

{md_table(
        ["Tipo", "Caminho"],
        [
            ["Relatório compacto", "`artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT_COMPACT.md`"],
            ["Relatório extenso", "`artifacts/clinvar_regional_eval/researcher_transfer_report/ABRAOM_RESEARCHER_TRANSFER_REPORT.md`"],
            ["Dataset ABRAOM adapter", "`data/datasets/abraom_frequency_adapter/`"],
            ["Dataset ClinVar x ABRAOM", "`data/datasets/clinvar/regional_abraom/`"],
            ["Tabela final", "`artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.csv`"],
            ["Relatório M5_v3", "`artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/M5_V3_SAFETY_REPORT.md`"],
            ["Validação de sinal", "`artifacts/clinvar_regional_eval/regional_signal_validation_next_step/REGIONAL_SIGNAL_VALIDATION_NEXT_STEP_REPORT.md`"],
            ["Fila de curadoria", "`artifacts/clinvar_regional_eval/public_abraom_validation/public_evidence_review_queue.tsv`"],
        ],
    )}

Cobertura GitHub:

- Estão no GitHub: scripts, relatórios, manifesto S3, CSS/HTML/DOCX/Markdown, CSV/TSV/JSON leves de decisão.
- Não estão no GitHub: `data/datasets/**`, parquets grandes, checkpoints/modelos e artefatos binários grandes; recuperar pelo S3 ou storage compartilhado.
- Para qualquer caminho S3 com status `prefixo default do launcher`, validar existência com `aws s3 ls` antes de assumir disponibilidade.

## 7. Limitações e Próximo Passo

Limitações principais:

- ABRAOM é frequência populacional, não verdade clínica.
- Submitter brasileiro ajuda a estratificar, mas não prova ancestralidade.
- P/LP presente no ABRAOM pode ser founder, recessivo, penetrância variável ou erro de classificação.
- Controles negativos fortes ainda chegam perto do real em alguns cenários.
- O painel público pronto ainda é pequeno demais para decisão clínica.
- No painel de stress high-priority, `M5_v3_safety` ainda falha fortemente; isso deve guiar curadoria antes de qualquer nova alegação de modelo.

Próximo passo imediato:

1. Curar as `{fmt(public_summary.get('high_priority_rows'))}` variantes críticas públicas.
2. Resolver os conflitos e preencher fontes públicas/PMIDs.
3. Rerodar `scripts/build_public_abraom_validation_package.py`.
4. Reavaliar `M5_v3_safety` contra um painel sentinela público maior.
5. Só depois decidir se vale novo treino ou nova arquitetura.

## 8. Comandos de Continuidade

```bash
cd /home/sagemaker-user/lumina-ssm
git switch {branch}
uv sync
uv run python scripts/compile_abraom_researcher_transfer_report.py
```

Decisão atual:

- `M5_v3_safety`: `{m5_v3.get("decision", "NA")}`
- Validação regional: `{regional_signal.get("decision", "NA")}`
- Pacote público: `{public_summary.get("decision", "NA")}`
"""
    return markdown


def write_html(md_path: Path, html_path: Path, *, title: str = "ABRAOM Regionalization Researcher Transfer Report") -> bool:
    if shutil.which("pandoc") is None:
        return False
    md_path = md_path.resolve()
    html_path = html_path.resolve()
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.55; color: #1f2933; max-width: 1080px; margin: 36px auto; padding: 0 28px; }
h1, h2, h3 { color: #17324d; line-height: 1.2; }
h1 { border-bottom: 3px solid #356b57; padding-bottom: 12px; }
h2 { border-bottom: 1px solid #d8dee9; padding-bottom: 6px; margin-top: 34px; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 0.92em; }
th, td { border: 1px solid #d8dee9; padding: 7px 9px; vertical-align: top; }
th { background: #eef3f8; }
code { background: #f3f6f9; padding: 1px 4px; border-radius: 4px; }
pre code { background: transparent; padding: 0; }
pre { background: #f6f8fa; padding: 12px; overflow-x: auto; border: 1px solid #d8dee9; border-radius: 6px; }
blockquote { border-left: 4px solid #356b57; padding-left: 14px; color: #4b5563; }
@media print { body { max-width: none; margin: 18mm; } table { font-size: 0.78em; } }
"""
    css_path = html_path.with_suffix(".css")
    css_path.write_text(css, encoding="utf-8")
    result = subprocess.run(
        [
            "pandoc",
            str(md_path),
            "--standalone",
            "--toc",
            "--toc-depth=2",
            "--metadata",
            f"title={title}",
            "--css",
            css_path.name,
            "-o",
            str(html_path),
        ],
        cwd=html_path.parent,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        (html_path.with_suffix(".html.error.log")).write_text(result.stderr, encoding="utf-8")
        return False
    return True


def write_docx(md_path: Path, docx_path: Path) -> bool:
    if shutil.which("pandoc") is None:
        return False
    md_path = md_path.resolve()
    docx_path = docx_path.resolve()
    result = subprocess.run(
        ["pandoc", str(md_path), "--toc", "--toc-depth=2", "-o", str(docx_path)],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        docx_path.with_suffix(".docx.error.log").write_text(result.stderr, encoding="utf-8")
        return False
    return True


def write_pdf(md_path: Path, pdf_path: Path) -> tuple[bool, str]:
    if shutil.which("pandoc") is None:
        return False, "pandoc not available"
    md_path = md_path.resolve()
    pdf_path = pdf_path.resolve()
    for engine in ["xelatex", "lualatex", "pdflatex", "wkhtmltopdf", "tectonic", "typst"]:
        if shutil.which(engine):
            result = subprocess.run(
                ["pandoc", str(md_path), "--toc", "--toc-depth=2", f"--pdf-engine={engine}", "-o", str(pdf_path)],
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                return True, engine
            pdf_path.with_suffix(f".{engine}.error.log").write_text(result.stderr, encoding="utf-8")
    return False, "no PDF engine found; use HTML or DOCX export"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    md_path = args.output_dir / f"{REPORT_BASENAME}.md"
    html_path = args.output_dir / f"{REPORT_BASENAME}.html"
    docx_path = args.output_dir / f"{REPORT_BASENAME}.docx"
    pdf_path = args.output_dir / f"{REPORT_BASENAME}.pdf"
    compact_md_path = args.output_dir / f"{COMPACT_REPORT_BASENAME}.md"
    compact_html_path = args.output_dir / f"{COMPACT_REPORT_BASENAME}.html"
    compact_docx_path = args.output_dir / f"{COMPACT_REPORT_BASENAME}.docx"
    compact_pdf_path = args.output_dir / f"{COMPACT_REPORT_BASENAME}.pdf"
    s3_manifest_path = args.output_dir / "s3_reuse_manifest.json"

    md_path.write_text(generate_markdown(), encoding="utf-8")
    compact_md_path.write_text(generate_compact_markdown(), encoding="utf-8")
    s3_manifest_path.write_text(json.dumps(S3_REUSE_ARTIFACTS, indent=2), encoding="utf-8")
    html_ok = False
    docx_ok = False
    pdf_ok = False
    compact_html_ok = False
    compact_docx_ok = False
    compact_pdf_ok = False
    pdf_note = "skipped"
    compact_pdf_note = "skipped"
    if not args.skip_pandoc:
        html_ok = write_html(md_path, html_path)
        docx_ok = write_docx(md_path, docx_path)
        pdf_ok, pdf_note = write_pdf(md_path, pdf_path)
        compact_html_ok = write_html(
            compact_md_path,
            compact_html_path,
            title="ABRAOM Regionalization Compact Handoff",
        )
        compact_docx_ok = write_docx(compact_md_path, compact_docx_path)
        compact_pdf_ok, compact_pdf_note = write_pdf(compact_md_path, compact_pdf_path)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "markdown": str(md_path),
        "html": str(html_path) if html_ok else None,
        "docx": str(docx_path) if docx_ok else None,
        "pdf": str(pdf_path) if pdf_ok else None,
        "compact_markdown": str(compact_md_path),
        "compact_html": str(compact_html_path) if compact_html_ok else None,
        "compact_docx": str(compact_docx_path) if compact_docx_ok else None,
        "compact_pdf": str(compact_pdf_path) if compact_pdf_ok else None,
        "s3_reuse_manifest": str(s3_manifest_path),
        "pdf_note": pdf_note,
        "compact_pdf_note": compact_pdf_note,
    }
    summary_path = args.output_dir / "report_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
