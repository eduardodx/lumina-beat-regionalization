#!/usr/bin/env python3
"""Compile a static HTML report for the ABRAOM regionalization study."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_OUTPUT = Path("artifacts/clinvar_regional_eval/abraom_study_report/abraom_regionalization_study.html")
FINAL_SUMMARY = Path(
    "artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/"
    "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.csv"
)
DYNAMIC_SUMMARY = Path("artifacts/adapter_fusion_blueprint_completion/dynamic_fusion_regional_summary.csv")
M5_V3_SUMMARY = Path("artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m5_v3_safety_summary.json")
M5_V3_NEGATIVE_CONTROLS = Path(
    "artifacts/clinvar_regional_eval/m5_v3_safety_calibrated/m5_v3_negative_control_comparison.csv"
)
STRONG_NEGATIVE_CONTROLS = Path(
    "artifacts/clinvar_regional_eval/regional_signal_validation_next_step/strong_negative_control_comparison.csv"
)
ERROR_COUNTS = Path("artifacts/adapter_fusion_error_analysis_deep/error_counts_by_model_dataset.csv")
CRITICAL_ERROR_CATEGORIES = Path(
    "artifacts/clinvar_regional_eval/regional_signal_validation_next_step/critical_error_category_summary.csv"
)
CRITICAL_ERROR_GENES = Path("artifacts/clinvar_regional_eval/regional_signal_validation_next_step/critical_error_gene_summary.csv")
PUBLIC_VALIDATION_SUMMARY = Path(
    "artifacts/clinvar_regional_eval/public_abraom_validation/public_abraom_validation_summary.json"
)
PUBLIC_VALIDATION_METRICS = Path(
    "artifacts/clinvar_regional_eval/public_abraom_validation/public_validation_metrics_by_panel.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "NA"
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
    return f"{100 * float(value):.{digits}f}%"


def metric_lookup(frame: pd.DataFrame, model: str, dataset: str, metric: str) -> float:
    rows = frame.loc[(frame["model"].eq(model)) & (frame["dataset"].eq(dataset))]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][metric])


def render_table(headers: list[str], rows: Iterable[Iterable[Any]], *, classes: str = "") -> str:
    class_attr = f' class="{esc(classes)}"' if classes else ""
    thead = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<div class=\"table-wrap\"><table{class_attr}><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def artifact_link(path: Path) -> str:
    return f"<code>{esc(str(path))}</code>"


def delta_badge(delta: float, *, good_when_positive: bool = True) -> str:
    if math.isnan(delta):
        return '<span class="badge neutral">NA</span>'
    good = delta >= 0 if good_when_positive else delta <= 0
    klass = "good" if good else "warn"
    sign = "+" if delta >= 0 else ""
    return f'<span class="badge {klass}">{sign}{fmt(delta)}</span>'


def model_comparison_rows(summary: pd.DataFrame) -> list[list[str]]:
    models = [
        ("M0", "Geral ClinVar non-BR"),
        ("M4", "Fusion regional sem frequência explícita forte"),
        ("M5", "Fusion + frequência explícita, bruto"),
        ("M6", "Frequência explícita alternativa, bruto"),
        ("M5_calibrated", "M5 com desconto limitado v2 inicial"),
        ("M6_calibrated", "M6 com desconto limitado v2 inicial"),
        ("M7_scrambled", "Controle negativo com ABRAOM embaralhado"),
        ("M5_v2_calibrated", "Lead calibrado holdout"),
        ("M5_v3_safety", "Lead com guarda molecular"),
    ]
    rows: list[list[str]] = []
    for model, description in models:
        br_mcc = metric_lookup(summary, model, "br_only", "mcc")
        benign_spec = metric_lookup(summary, model, "abraom_common_benign", "specificity")
        plp_recall = metric_lookup(summary, model, "abraom_pathogenic_present", "recall")
        global_mcc = metric_lookup(summary, model, "global_nonbr_no_abraom", "mcc")
        global_spec = metric_lookup(summary, model, "global_nonbr_no_abraom", "specificity")
        rows.append(
            [
                f"<code>{esc(model)}</code>",
                esc(description),
                fmt(br_mcc),
                fmt(benign_spec),
                fmt(plp_recall),
                fmt(global_mcc),
                fmt(global_spec),
            ]
        )
    return rows


def dynamic_rows(dynamic: pd.DataFrame) -> list[list[str]]:
    models = ["M0", "M2_gnomad_only", "M4_dynamic_gated", "M5_dynamic_gated", "M7_dynamic_scrambled", "M5_v2_calibrated"]
    rows: list[list[str]] = []
    for model in models:
        rows.append(
            [
                f"<code>{esc(model)}</code>",
                fmt(metric_lookup(dynamic, model, "br_only", "mcc")),
                fmt(metric_lookup(dynamic, model, "abraom_common_benign", "specificity")),
                fmt(metric_lookup(dynamic, model, "abraom_pathogenic_present", "recall")),
                fmt(metric_lookup(dynamic, model, "global_nonbr_no_abraom", "mcc")),
            ]
        )
    return rows


def slice_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    m5 = summary.loc[summary["model"].eq("M5_v3_safety")].copy()
    labels = {
        "br_only": "ClinVar com submissões brasileiras apenas",
        "br_any": "ClinVar com qualquer submissão brasileira",
        "regional_benchmark_any": "Benchmark regional agregado",
        "abraom_common_benign": "B/LB comuns no ABRAOM",
        "abraom_pathogenic_present": "P/LP presentes no ABRAOM",
        "abraom_pathogenic_common": "P/LP comuns no ABRAOM",
        "global_nonbr_no_abraom": "Controle global non-BR sem ABRAOM",
        "nonbr_only": "ClinVar non-BR apenas",
    }
    for dataset, label in labels.items():
        row = m5.loc[m5["dataset"].eq(dataset)]
        if row.empty:
            continue
        rows.append([f"<code>{esc(dataset)}</code>", esc(label), str(int(row.iloc[0]["n"]))])
    return rows


def negative_control_rows(controls: pd.DataFrame, datasets: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    part = controls.loc[controls["dataset"].isin(datasets)].copy()
    for row in part.itertuples(index=False):
        rows.append(
            [
                f"<code>{esc(row.dataset)}</code>",
                f"<code>{esc(row.metric)}</code>",
                f"<code>{esc(row.control_mode)}</code>",
                fmt(row.real_value),
                fmt(row.control_mean),
                fmt(row.control_p95),
                fmt(row.empirical_p_control_ge_real, 4),
                pct(row.mean_changed_discount_fraction),
            ]
        )
    return rows


def public_status_rows(summary: dict[str, Any]) -> list[list[str]]:
    status = summary.get("public_review_status", {})
    decisions = summary.get("public_evidence_decision", {})
    rows = []
    for key, value in status.items():
        rows.append([f"<code>{esc(key)}</code>", str(int(value)), "status"])
    for key, value in decisions.items():
        rows.append([f"<code>{esc(key)}</code>", str(int(value)), "decision"])
    return rows


def stress_panel_rows(metrics: pd.DataFrame) -> list[list[str]]:
    part = metrics.loc[metrics["panel"].eq("high_priority_review_queue")].copy()
    rows = []
    for row in part.itertuples(index=False):
        rows.append(
            [
                f"<code>{esc(row.model)}</code>",
                str(int(row.n)),
                fmt(row.recall),
                fmt(row.specificity),
                fmt(row.mcc),
                str(int(row.fp)),
                str(int(row.fn)),
            ]
        )
    return rows


def critical_error_rows(categories: pd.DataFrame, limit: int = 12) -> list[list[str]]:
    part = categories.head(limit)
    rows = []
    for row in part.itertuples(index=False):
        category = getattr(row, "failure_category", getattr(row, "category", ""))
        rows.append(
            [
                f"<code>{esc(row.audit_type)}</code>",
                f"<code>{esc(category)}</code>",
                f"<code>{esc(row.priority_tier)}</code>",
                str(int(row.n)),
            ]
        )
    return rows


def top_gene_rows(genes: pd.DataFrame, limit: int = 12) -> list[list[str]]:
    part = genes.head(limit)
    rows = []
    for row in part.itertuples(index=False):
        mean_priority = getattr(row, "mean_priority", getattr(row, "mean_priority_score", float("nan")))
        rows.append(
            [
                f"<code>{esc(row.audit_type)}</code>",
                f"<strong>{esc(row.GeneSymbol)}</strong>",
                str(int(row.n)),
                fmt(mean_priority),
                fmt(row.median_af_abraom, 4),
            ]
        )
    return rows


def generate_html() -> str:
    final_summary = read_csv(FINAL_SUMMARY)
    dynamic_summary = read_csv(DYNAMIC_SUMMARY)
    m5_v3_summary = read_json(M5_V3_SUMMARY)
    negative_controls = read_csv(M5_V3_NEGATIVE_CONTROLS)
    strong_controls = read_csv(STRONG_NEGATIVE_CONTROLS)
    error_counts = read_csv(ERROR_COUNTS)
    critical_categories = read_csv(CRITICAL_ERROR_CATEGORIES)
    critical_genes = read_csv(CRITICAL_ERROR_GENES)
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

    config = m5_v3_summary.get("selected_config", {})
    decision = m5_v3_summary.get("decision", "conditional_candidate")
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    pending_public = public_summary.get("public_review_status", {}).get("needs_public_lookup", 0)
    conflicts_public = public_summary.get("public_review_status", {}).get("public_label_conflict", 0)
    ready_public = public_summary.get("ready_subset_rows", 0)

    model_table = render_table(
        [
            "Modelo",
            "Receita",
            "br_only MCC",
            "ABRAOM common benign specificity",
            "ABRAOM P/LP present recall",
            "global nonBR MCC",
            "global nonBR specificity",
        ],
        model_comparison_rows(final_summary),
    )

    html_doc = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ABRAOM Adapter-Fusion Regionalization Study</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #5b6675;
      --line: #d7dde5;
      --panel: #f7f9fc;
      --panel-2: #eef6f1;
      --blue: #2458a6;
      --green: #1f7a4d;
      --amber: #a35f00;
      --red: #a43d3d;
      --white: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #fbfcfe;
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }}
    header {{
      background: linear-gradient(115deg, #0f243d 0%, #254e6f 58%, #3f6b55 100%);
      color: white;
      padding: 42px 28px 30px;
    }}
    header .inner, main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 44px); line-height: 1.05; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 12px; font-size: 24px; letter-spacing: 0; }}
    h3 {{ margin: 24px 0 8px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 8px 0 14px; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.93em; }}
    .subtitle {{ max-width: 880px; color: #dbe7f3; font-size: 17px; }}
    .meta {{ color: #d2dee9; font-size: 13px; margin-top: 18px; }}
    main {{ padding: 24px 28px 60px; }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 28px;
      padding: 12px 0;
      border-bottom: 1px solid var(--line);
    }}
    nav a {{
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0 24px;
    }}
    .metric {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 138px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .metric .label {{ color: var(--muted); font-size: 13px; min-height: 40px; }}
    .metric .value {{ font-size: 29px; font-weight: 730; margin-top: 10px; }}
    .metric .sub {{ color: var(--muted); font-size: 13px; }}
    .bar {{
      margin-top: 10px;
      height: 8px;
      background: #e7edf4;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar span {{ display: block; height: 100%; background: var(--green); }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: var(--panel);
      padding: 14px 16px;
      margin: 18px 0;
    }}
    .callout.warning {{ border-color: var(--amber); background: #fff7e8; }}
    .callout.danger {{ border-color: var(--red); background: #fff1f1; }}
    .callout.good {{ border-color: var(--green); background: var(--panel-2); }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      margin: 12px 0 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #eef2f7;
      color: #273241;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 650;
      font-size: 12px;
      white-space: nowrap;
    }}
    .badge.good {{ color: #0c5130; background: #dff3e7; }}
    .badge.warn {{ color: #794100; background: #ffe6bd; }}
    .badge.neutral {{ color: #394554; background: #e9eef5; }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(5, minmax(160px, 1fr));
      gap: 10px;
      margin: 14px 0 22px;
    }}
    .step {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 12px;
      min-height: 120px;
    }}
    .step strong {{ display: block; margin-bottom: 6px; }}
    .sources {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 18px;
      font-size: 13px;
    }}
    footer {{
      color: var(--muted);
      border-top: 1px solid var(--line);
      margin-top: 34px;
      padding-top: 16px;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .sources {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      header {{ padding: 30px 18px 24px; }}
      main {{ padding: 18px; }}
      .grid, .steps {{ grid-template-columns: 1fr; }}
      table {{ min-width: 680px; }}
    }}
    @media print {{
      nav {{ display: none; }}
      body {{ background: white; }}
      header {{ background: #24364a; }}
      .table-wrap {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="inner">
      <h1>ABRAOM Adapter-Fusion Regionalization Study</h1>
      <p class="subtitle">Relatório técnico dos treinamentos e avaliações para regionalização brasileira de interpretação ClinVar com adapters, ABRAOM, gnomAD, controles negativos e análise de segurança P/LP.</p>
      <div class="meta">Gerado em UTC: {esc(generated)} · Uso: validação científica, não validação clínica final.</div>
    </div>
  </header>
  <main>
    <nav aria-label="Seções">
      <a href="#conclusao">Conclusão</a>
      <a href="#desenho">Desenho</a>
      <a href="#dados">Dados</a>
      <a href="#modelos">Modelos</a>
      <a href="#dinamico">Fusion dinâmica</a>
      <a href="#safety">M5_v3 safety</a>
      <a href="#controles">Controles negativos</a>
      <a href="#erros">Erro e curadoria</a>
      <a href="#fontes">Fontes</a>
    </nav>

    <section id="conclusao">
      <h2>Conclusão Executiva Técnica</h2>
      <div class="callout good">
        <strong>Resultado principal:</strong> a regionalização com ABRAOM gerou ganho mensurável em desempenho brasileiro e reduziu falsos positivos em variantes benignas comuns no ABRAOM. A melhor estrutura operacional até agora é <code>M5_v3_safety</code>, uma calibração de <code>M5_v2</code> com separação entre evidência molecular e score regional, mais guarda molecular contra supressão excessiva.
      </div>
      <div class="grid">
        <div class="metric">
          <div class="label">Ganho brasileiro: <code>br_only</code> MCC</div>
          <div class="value">{fmt(br_m5v3)}</div>
          <div class="sub">M0 {fmt(br_m0)} · Δ {delta_badge(br_m5v3 - br_m0)}</div>
          <div class="bar"><span style="width:{min(100, max(0, br_m5v3 * 100)):.1f}%"></span></div>
        </div>
        <div class="metric">
          <div class="label">Redução de falso positivo: benignas comuns ABRAOM</div>
          <div class="value">{pct(benign_m5v3)}</div>
          <div class="sub">M0 {pct(benign_m0)} · Δ {delta_badge(benign_m5v3 - benign_m0)}</div>
          <div class="bar"><span style="width:{min(100, max(0, benign_m5v3 * 100)):.1f}%"></span></div>
        </div>
        <div class="metric">
          <div class="label">Proteção de sensibilidade: P/LP presentes no ABRAOM</div>
          <div class="value">{pct(plp_m5v3)}</div>
          <div class="sub">M0 {pct(plp_m0)} · Δ {delta_badge(plp_m5v3 - plp_m0)}</div>
          <div class="bar"><span style="width:{min(100, max(0, plp_m5v3 * 100)):.1f}%"></span></div>
        </div>
        <div class="metric">
          <div class="label">Não degradação global: <code>global_nonbr_no_abraom</code> MCC</div>
          <div class="value">{fmt(global_m5v3)}</div>
          <div class="sub">M0 {fmt(global_m0)} · Δ {delta_badge(global_m5v3 - global_m0)}</div>
          <div class="bar"><span style="width:{min(100, max(0, global_m5v3 * 100)):.1f}%"></span></div>
        </div>
      </div>
      <div class="callout warning">
        <strong>Limitação central:</strong> os controles negativos estratificados ainda enfraquecem a afirmação de que o ganho é especificamente biológico do ABRAOM. Eles sugerem que parte do sinal pode ser explicada por frequência, gene, tipo de variante ou estrutura do benchmark. Por isso, a decisão atual é não treinar outro modelo antes de resolver evidência pública e curadoria manual.
      </div>
    </section>

    <section id="desenho">
      <h2>Desenho Científico</h2>
      <p>O blueprint recomendava separar três fontes de evidência: gramática molecular do modelo base, evidência populacional/frequencial ABRAOM ou gnomAD, e interpretação ClinVar P/LP versus B/LB. O estudo implementou essa lógica por ablações, calibração e controles negativos.</p>
      <div class="steps">
        <div class="step"><strong>1. M0</strong>Modelo ClinVar non-BR geral, sem regionalização ABRAOM. Serve como baseline molecular.</div>
        <div class="step"><strong>2. M4/M5/M6</strong>Adapters/fusion regionais e frequências explícitas. Mostraram ganho brasileiro, mas risco de over-suppression.</div>
        <div class="step"><strong>3. M5_v2</strong>Calibração holdout com desconto regional limitado e score molecular separado.</div>
        <div class="step"><strong>4. M7</strong>Controle negativo com sinal ABRAOM embaralhado para medir efeito de arquitetura/parâmetros.</div>
        <div class="step"><strong>5. M5_v3</strong>Guarda molecular: frequência pode reduzir score, mas não apagar evidência molecular forte.</div>
      </div>
      <div class="callout">
        <strong>Receita M5_v3 selecionada:</strong> <code>discount_scale={fmt(config.get("discount_scale"))}</code>, <code>max_discount={fmt(config.get("max_discount"))}</code>, <code>molecular_guard_threshold={fmt(config.get("molecular_guard_threshold"))}</code>, <code>guarded_max_discount={fmt(config.get("guarded_max_discount"))}</code>, <code>regional_threshold={fmt(config.get("regional_threshold"))}</code>, <code>global_threshold={fmt(config.get("global_threshold"))}</code>.
      </div>
    </section>

    <section id="dados">
      <h2>Dados e Slices de Avaliação</h2>
      <p>Os resultados abaixo usam os mesmos slices regionais definidos ao longo do estudo. O ABRAOM é usado como evidência populacional/frequencial, não como truth set clínico.</p>
      {render_table(["Slice", "Interpretação", "n"], slice_rows(final_summary))}
    </section>

    <section id="modelos">
      <h2>Comparação Principal dos Modelos</h2>
      <p>A tabela mostra as métricas centrais de test. O alvo científico era: aumentar <code>br_only</code> MCC, elevar especificidade em benignas comuns ABRAOM, preservar recall P/LP ABRAOM-presentes, e não degradar o controle global non-BR.</p>
      {model_table}
      <div class="callout">
        <strong>Leitura:</strong> <code>M5</code>/<code>M6</code> brutos reduziram falsos positivos, mas derrubaram recall P/LP. <code>M5_v2_calibrated</code> corrigiu grande parte desse problema. <code>M5_v3_safety</code> manteve os ganhos de M5_v2 e restaurou o MCC global para o nível de M0, mas ainda não prova especificidade biológica ABRAOM.
      </div>
    </section>

    <section id="dinamico">
      <h2>Fusion Dinâmica e Controle M7</h2>
      <p>A rodada dinâmica testou se o gate aprendia a combinar adapters populacionais de modo útil. O achado foi misto: <code>M5_dynamic_gated</code> teve forte ganho em <code>br_only</code> e quase eliminou falso positivo comum ABRAOM, mas colapsou o recall P/LP ABRAOM-presentes.</p>
      {render_table(["Modelo", "br_only MCC", "ABRAOM common benign specificity", "ABRAOM P/LP present recall", "global nonBR MCC"], dynamic_rows(dynamic_summary))}
      <div class="callout danger">
        <strong>Falha de segurança do dinâmico bruto:</strong> <code>M5_dynamic_gated</code> chegou a specificity {fmt(metric_lookup(dynamic_summary, "M5_dynamic_gated", "abraom_common_benign", "specificity"))} em benignas comuns ABRAOM, mas recall P/LP caiu para {fmt(metric_lookup(dynamic_summary, "M5_dynamic_gated", "abraom_pathogenic_present", "recall"))}. Isso é o padrão de over-suppression que motivou M5_v2/M5_v3.
      </div>
    </section>

    <section id="safety">
      <h2>M5_v3 Safety</h2>
      <p>Decisão registrada: <code>{esc(decision)}</code>. A mudança conceitual foi impor uma guarda molecular: quando o score molecular é forte, o desconto regional é bloqueado ou limitado para evitar transformar P/LP ABRAOM-presentes em falso benigno.</p>
      {render_table(
        ["Dataset", "M0", "M7", "M5_v2", "M5_v3", "Métrica"],
        [
            [
                "<code>br_only</code>",
                fmt(metric_lookup(final_summary, "M0", "br_only", "mcc")),
                fmt(metric_lookup(final_summary, "M7_scrambled", "br_only", "mcc")),
                fmt(metric_lookup(final_summary, "M5_v2_calibrated", "br_only", "mcc")),
                fmt(metric_lookup(final_summary, "M5_v3_safety", "br_only", "mcc")),
                "<code>mcc</code>",
            ],
            [
                "<code>abraom_common_benign</code>",
                fmt(metric_lookup(final_summary, "M0", "abraom_common_benign", "specificity")),
                fmt(metric_lookup(final_summary, "M7_scrambled", "abraom_common_benign", "specificity")),
                fmt(metric_lookup(final_summary, "M5_v2_calibrated", "abraom_common_benign", "specificity")),
                fmt(metric_lookup(final_summary, "M5_v3_safety", "abraom_common_benign", "specificity")),
                "<code>specificity</code>",
            ],
            [
                "<code>abraom_pathogenic_present</code>",
                fmt(metric_lookup(final_summary, "M0", "abraom_pathogenic_present", "recall")),
                fmt(metric_lookup(final_summary, "M7_scrambled", "abraom_pathogenic_present", "recall")),
                fmt(metric_lookup(final_summary, "M5_v2_calibrated", "abraom_pathogenic_present", "recall")),
                fmt(metric_lookup(final_summary, "M5_v3_safety", "abraom_pathogenic_present", "recall")),
                "<code>recall</code>",
            ],
            [
                "<code>global_nonbr_no_abraom</code>",
                fmt(metric_lookup(final_summary, "M0", "global_nonbr_no_abraom", "mcc")),
                fmt(metric_lookup(final_summary, "M7_scrambled", "global_nonbr_no_abraom", "mcc")),
                fmt(metric_lookup(final_summary, "M5_v2_calibrated", "global_nonbr_no_abraom", "mcc")),
                fmt(metric_lookup(final_summary, "M5_v3_safety", "global_nonbr_no_abraom", "mcc")),
                "<code>mcc</code>",
            ],
        ],
      )}
    </section>

    <section id="controles">
      <h2>Controles Negativos e Falsificação</h2>
      <p>Os controles embaralham ou permutam o sinal regional preservando estruturas importantes. Se o controle iguala ou supera o ABRAOM real, a evidência de especificidade regional não está provada.</p>
      <h3>Controles M5_v3</h3>
      {render_table(["Dataset", "Métrica", "Controle", "Real", "Média controle", "P95", "P(controle >= real)", "Desconto alterado"], negative_control_rows(negative_controls, ["br_only", "abraom_common_benign", "abraom_pathogenic_present", "global_nonbr_no_abraom"]))}
      <h3>Controles Estratificados Fortes</h3>
      {render_table(["Dataset", "Métrica", "Controle", "Real", "Média controle", "P95", "P(controle >= real)", "Desconto alterado"], negative_control_rows(strong_controls, ["br_only", "abraom_common_benign", "abraom_pathogenic_present", "global_nonbr_no_abraom"]))}
      <div class="callout warning">
        <strong>Interpretação rigorosa:</strong> <code>br_only</code> ainda mostra ganho absoluto, mas controles dentro de gene/AF bin e outros estratos chegam muito perto ou superam o real em vários alvos. Portanto, a conclusão correta é “sinal útil para calibração regional”, não “prova fechada de aprendizado biológico específico do ABRAOM”.
      </div>
    </section>

    <section id="erros">
      <h2>Análise de Erro e Curadoria</h2>
      <p>Após M5_v3, a principal barreira deixou de ser arquitetura e passou a ser evidência: ainda há P/LP ABRAOM-presentes chamados como benignos e benignas comuns ABRAOM chamadas como patogênicas.</p>
      <div class="grid">
        <div class="metric">
          <div class="label">P/LP ABRAOM-presentes falso benignos</div>
          <div class="value">92</div>
          <div class="sub">60 em alta prioridade de revisão</div>
        </div>
        <div class="metric">
          <div class="label">Benignas comuns ABRAOM falso patogênicas</div>
          <div class="value">502</div>
          <div class="sub">15 em alta prioridade de revisão</div>
        </div>
        <div class="metric">
          <div class="label">Fila pública de alta prioridade</div>
          <div class="value">{int(public_summary.get("high_priority_rows", 0))}</div>
          <div class="sub">{int(pending_public)} precisam lookup público</div>
        </div>
        <div class="metric">
          <div class="label">Prontas como sentinela pública</div>
          <div class="value">{int(ready_public)}</div>
          <div class="sub">{int(conflicts_public)} conflitos públicos a resolver</div>
        </div>
      </div>
      <h3>Categorias Críticas</h3>
      {render_table(["Tipo de auditoria", "Categoria", "Tier", "n"], critical_error_rows(critical_categories))}
      <h3>Genes Prioritários</h3>
      {render_table(["Tipo de auditoria", "Gene", "n", "Prioridade média", "AF ABRAOM mediana"], top_gene_rows(critical_genes))}
      <h3>Painel de Estresse de Alta Prioridade</h3>
      <p>Este painel foi construído a partir de erros; ele serve para caracterizar falhas, não para estimar desempenho populacional.</p>
      {render_table(["Modelo", "n", "Recall", "Specificity", "MCC", "FP", "FN"], stress_panel_rows(public_metrics))}
      <h3>Status de Evidência Pública</h3>
      {render_table(["Categoria", "n", "Tipo"], public_status_rows(public_summary))}
    </section>

    <section id="fontes">
      <h2>Artefatos de Origem</h2>
      <div class="sources">
        <div>{artifact_link(FINAL_SUMMARY)}</div>
        <div>{artifact_link(DYNAMIC_SUMMARY)}</div>
        <div>{artifact_link(M5_V3_SUMMARY)}</div>
        <div>{artifact_link(M5_V3_NEGATIVE_CONTROLS)}</div>
        <div>{artifact_link(STRONG_NEGATIVE_CONTROLS)}</div>
        <div>{artifact_link(ERROR_COUNTS)}</div>
        <div>{artifact_link(CRITICAL_ERROR_CATEGORIES)}</div>
        <div>{artifact_link(CRITICAL_ERROR_GENES)}</div>
        <div>{artifact_link(PUBLIC_VALIDATION_SUMMARY)}</div>
        <div>{artifact_link(PUBLIC_VALIDATION_METRICS)}</div>
      </div>
      <footer>
        Relatório gerado por <code>scripts/compile_abraom_study_html_report.py</code>. A conclusão é científica e retrospectiva sobre os artefatos presentes; não substitui validação clínica, curadoria independente, nem avaliação prospectiva em coorte brasileira.
      </footer>
    </section>
  </main>
</body>
</html>
"""
    return html_doc


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_html(), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "generated_at_utc": datetime.now(UTC).isoformat()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
