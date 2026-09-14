#!/usr/bin/env python3
"""Diagnostico da divergencia de submissor brasileiro: Mosaic (lista revisada) x pipeline regional v1.

Roda no notebook (pandas + pyarrow + pyyaml; sem GPU). Le o S3 so para o master regional; so escreve em
--out-dir. Contrato v2, secao 6.

POR QUE EXISTE
--------------
A auditoria [F] mostrou que as duas definicoes de "variante com submissor brasileiro" discordam (1.250 casos do
br_clinical_evidence no treino v1, 46 no T_nonBR v1, 20 controles do Mosaic no T_BR v1). Antes de reconstruir os
dados e preciso saber POR QUE: cobertura da tabela regional, mapeamento de instituicoes, classe de SCV, release
ou definicao (any x only).

COMO
----
1. Reproduz o `is_br` do Mosaic por SCV com as funcoes e a configuracao do PROPRIO Mosaic (commit conferido), a
   partir do submission_summary do release (sha256 conferido contra config/sources.yaml) e das
   clinvar_variation_ids publicadas no pb_examples (o mesmo indice do build). Valida contra os
   br_lab_any/only/shared publicados em TODOS os exemplos. Se um unico exemplo divergir, PARA com codigo 2: sem
   reproducao fiel, nenhuma conclusao sobre a divergencia vale (pedir a pasta interim/ ao Eduardo).
2. Cruza cada exemplo do Mosaic com o master regional v1 e com os conjuntos v1 (T_BR, pares, splits).
3. Classifica pelas MARCACOES dos dois lados. Marcar como brasileira nao prova nacionalidade, e nao marcar nao
   quer dizer "nao brasileira":
     A1  Mosaic marca BR; a v1 nao tem nenhuma linha da variante (cobertura)
     A2  Mosaic marca BR; a v1 tem linhas da variante, nenhuma brazilian
     B   a v1 marca BR; o Mosaic nao marca
     C1  os dois marcam; v1 sem linha non_brazilian, Mosaic compartilhada (br_lab_shared)
     C2  os dois marcam; v1 com linha non_brazilian, Mosaic so marcadas (br_lab_only)
     ambos_marcam_br, nenhum_marca_br_v1_coberta, nenhum_marca_br_v1_sem_cobertura, fora_do_master_v1
   Para as divergentes, grava a evidencia por SCV com o status da instituicao: na lista BR do Mosaic, pais Brazil
   no NCBI fora da lista, nao brasileira pelo registro do NCBI, pais nao resolvido ou ambiguo.
4. Escopo da lista: varre as SCVs do filtro P/B de TODOS os exemplos (inclusive onde os dois concordam) atras de
   instituicoes com pais Brazil no NCBI fora da lista do Mosaic.

O QUE NAO PROVA
---------------
- Reproducao fiel confirma compatibilidade com as flags publicadas, nao a correcao geografica das instituicoes.
- A v1 nao guarda o cohort por submissor e o eval_unified nao foi encontrado nos caminhos pesquisados: a presenca
  de um submissor vem de `regional_submitters` (nomes unicos unidos por ';', truncada em 20). So o nome inteiro
  entre separadores conta como presenca; substring ou nome com ';' sem espaco e so indicio; lista ausente, ou
  truncada sem o nome, e indeterminado.
- O pais vem do organization_summary ATUAL do NCBI (nao do congelado pelo Mosaic em 2026-08-22), casado pelo nome
  canonico; submissor e organizacao podem ter nomes diferentes. Pais nao resolvido nao prova nada.
- A varredura do passo 4 so enxerga SCVs do filtro P/B nos exemplos e instituicoes com registro Brazil no NCBI:
  nao valida a lista brasileira de forma completa.
- O release da v1 e desconhecido: ausencia de submissao nao separa release de cobertura.
- Nenhuma correcao automatica: o relatorio alimenta a definicao e a reconstrucao dos dados.

USO (notebook)
--------------
    set -o pipefail
    PYTHONPATH="$WORK" python3 scripts/diagnose_brazilian_submitter_divergence.py \\
        --mosaic-root ~/testeArq/lumina-mosaic \\
        --submission-summary ~/clinvar/2026-06/submission_summary_2026-06.txt.gz \\
        --organization-summary ~/clinvar/2026-06/organization_summary_2026-09-14.txt \\
        --pb-examples ~/mosaic-v1/pb_examples.parquet \\
        --membership ~/mosaic-v1/studies/brazil/membership.parquet \\
        --master-uri s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet \\
        --br ~/slices_enriched/br_only.enriched.parquet \\
        --pairs ~/artifacts/fase0/t_nonbr_matched_soft.parquet \\
        --splits ~/artifacts/fase0/clinvar_splits/clinvar_splits_combined.parquet \\
        --out-dir ~/artifacts/redesenho/diagnostico_submissor_br 2>&1 | tee ~/artifacts/redesenho/diagnostico.log
    echo "codigo de saida: $?"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_regionalization_data import as_bool, build_key, load_slice  # noqa: E402

EXPECTED_MOSAIC_COMMIT = "814e7f0"
DEFAULT_RELEASE = "2026-06"
TRUNCATION_SEP = ";...(+"
SET_PRIORITY = ("t_br_pareado", "t_br_slice", "t_nonbr_pareado", "split_train", "split_validation",
                "split_calibration")
TEST_SETS = ("t_br_slice", "t_br_pareado", "t_nonbr_pareado")
MASTER_COLUMNS = ["variant_key", "has_brazilian_submitter", "has_non_brazilian_submitter",
                  "brazilian_submission_rows", "non_brazilian_submission_rows", "regional_submission_rows",
                  "clinvar_regional_cohort", "regional_submitters"]
EXAMPLE_COLUMNS = ["variant_id", "clinvar_variation_ids", "chrom", "pos_1based", "ref", "alt", "label_tier",
                   "binary_label", "br_lab_any", "br_lab_only", "br_lab_shared"]
EVIDENCE_COLUMNS = ["variant_id", "key", "categoria", "conjunto_v1", "papel_mosaic", "binary_label", "variation_id",
                    "scv", "submitter", "clinical_significance", "review_status", "origin_counts", "contributes_raw",
                    "date_last_evaluated", "passa_filtro_pb", "org_id_mosaic", "is_br_mosaic", "pais_ncbi",
                    "status_instituicao", "submissor_na_tabela_v1"]

# categorias: descrevem MARCACOES, nao nacionalidade
A1 = "A1_mosaic_marca_br_v1_sem_cobertura"
A2 = "A2_mosaic_marca_br_v1_coberta_sem_linha_br"
B = "B_v1_marca_br_mosaic_nao_marca"
C1 = "C1_ambos_marcam_v1_sem_linha_nao_br_mosaic_compartilhada"
C2 = "C2_ambos_marcam_v1_com_linha_nao_br_mosaic_so_marcadas"
BOTH_MARK = "ambos_marcam_br"
NONE_MARK_COVERED = "nenhum_marca_br_v1_coberta"
NONE_MARK_UNCOVERED = "nenhum_marca_br_v1_sem_cobertura"
OUTSIDE_MASTER = "fora_do_master_v1"
NOT_DIVERGENT = (BOTH_MARK, NONE_MARK_COVERED, NONE_MARK_UNCOVERED, OUTSIDE_MASTER)
DEFINITIONS = {
    A1: "Mosaic marca BR (alguma SCV do filtro P/B de instituicao da lista); a v1 nao tem linha da variante.",
    A2: "Mosaic marca BR; a v1 tem linhas da variante e nenhuma com cohort brazilian.",
    B: "A v1 tem linha com cohort brazilian; o Mosaic nao marca (nenhuma SCV do filtro de instituicao da lista).",
    C1: "Os dois marcam; a v1 nao tem linha non_brazilian e o Mosaic tem SCV do filtro nao marcada (br_lab_shared).",
    C2: "Os dois marcam; a v1 tem linha non_brazilian e o Mosaic so tem SCVs do filtro marcadas (br_lab_only).",
    BOTH_MARK: "Os dois marcam BR com a mesma divisao so/compartilhada.",
    NONE_MARK_COVERED: "Nenhum dos dois marca BR; a v1 tem linhas da variante. NAO e nacionalidade confirmada.",
    NONE_MARK_UNCOVERED: "Nenhum dos dois marca BR; a v1 nao tem linha da variante (sem dado). NAO e nacionalidade.",
    OUTSIDE_MASTER: "Exemplo do Mosaic ausente do master regional v1.",
}

# status da instituicao de uma SCV
LISTED = "na_lista_br_do_mosaic"
REGISTRY_BRAZIL_UNLISTED = "pais_brazil_no_ncbi_fora_da_lista"
REGISTRY_NOT_BRAZIL = "nao_brasileira_pelo_registro_ncbi"
COUNTRY_UNRESOLVED = "pais_nao_resolvido"
COUNTRY_AMBIGUOUS = "pais_ambiguo_no_ncbi"
UNRESOLVED = (COUNTRY_UNRESOLVED, COUNTRY_AMBIGUOUS)
STATUS_ORDER = (REGISTRY_BRAZIL_UNLISTED, COUNTRY_UNRESOLVED, COUNTRY_AMBIGUOUS, REGISTRY_NOT_BRAZIL, LISTED)

# presenca do submissor na lista regional_submitters da v1
EXACT = "presente_exato"
SEMICOLON_AMBIGUOUS = "indicio_nome_com_ponto_e_virgula"
PARTIAL = "indicio_correspondencia_parcial"
ABSENT = "ausente"
TRUNCATED = "indeterminado_lista_truncada"
NO_LIST = "indeterminado_sem_lista"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mosaic-root", type=Path, required=True, help="clone do lumina-mosaic no commit do release")
    p.add_argument("--expected-mosaic-commit", default=EXPECTED_MOSAIC_COMMIT,
                   help="prefixo do commit exigido (vazio desliga a checagem e fica registrado no relatorio)")
    p.add_argument("--clinvar-release", default=DEFAULT_RELEASE)
    p.add_argument("--submission-summary", type=Path, required=True)
    p.add_argument("--organization-summary", type=Path, default=None, help="organization_summary do NCBI (pais)")
    p.add_argument("--pb-examples", type=Path, required=True)
    p.add_argument("--membership", type=Path, required=True, help="studies/brazil/membership.parquet do Mosaic")
    p.add_argument("--master-uri", required=True, help="master regional v1 (s3:// ou caminho local)")
    p.add_argument("--br", type=Path, required=True, help="slice br_only (T_BR v1)")
    p.add_argument("--pairs", type=Path, required=True, help="t_nonbr_matched_soft (pares v1)")
    p.add_argument("--splits", type=Path, required=True, help="clinvar_splits_combined.parquet (v1)")
    p.add_argument("--exclude-chrom", default="8")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# Mosaic: funcoes e configuracao do proprio release
# ------------------------------------------------------------------------------------------------


def load_mosaic(root: Path, *, release: str, expected_commit: str | None) -> SimpleNamespace:
    """Importa as funcoes do Mosaic a partir do clone e confere o commit (a reproducao tem de usar o release)."""
    root = Path(root).expanduser().resolve()
    commit = None
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    if expected_commit and not (commit or "").startswith(expected_commit):
        raise SystemExit(f"{root}: commit {commit!r}, esperado {expected_commit}")
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from mosaic.config import load_column_map, load_sources, release_map
    from mosaic.hashing import sha256_file
    from mosaic.labels import parse_scvs, pb_whitelist_scvs
    from mosaic.orgs import build_org_matcher, canonical_keys, load_br_freeze, resolve_submitter

    cfg = root / "config"
    freeze = load_br_freeze(cfg / "clinvar-organizations-br.yaml")
    key_to_org = build_org_matcher(freeze, cfg / "submitter-name-to-org.yaml")
    return SimpleNamespace(
        root=root,
        commit=commit,
        commit_conferido=bool(expected_commit),
        release=release,
        spec=release_map(load_column_map(cfg), release),
        sources=load_sources(cfg)["sources"],
        freeze_date=freeze.get("freeze_date"),
        include_ids={str(o["org_id"]) for o in freeze.get("organizations") or [] if o.get("include")},
        parse_scvs=parse_scvs,
        sha256_file=sha256_file,
        is_whitelisted=lambda scv: bool(pb_whitelist_scvs([scv])),
        org_of=lambda name: resolve_submitter(name, key_to_org),
        canon=lambda name: canonical_keys(name)[1],
    )


def verify_source(mosaic: SimpleNamespace, path: Path) -> dict:
    """O submission_summary tem de ser o arquivo do release: sha256 contra config/sources.yaml do Mosaic."""
    key = f"clinvar_submission_summary_{mosaic.release}"
    spec = mosaic.sources.get(key) or {}
    expected = spec.get("sha256")
    if not expected or expected == "pending":
        raise SystemExit(f"config/sources.yaml sem sha256 para {key}: nao ha como provar que o arquivo e o do release")
    digest = mosaic.sha256_file(path)
    if digest != expected:
        raise SystemExit(f"{path}: sha256 {digest} diferente do fixado em {key} ({expected})")
    return {"chave": key, "sha256": digest, "bytes": path.stat().st_size, "url_no_sources_yaml": spec.get("url")}


# ------------------------------------------------------------------------------------------------
# reproducao do is_br (mesma agregacao de labels.build_labels + examples.assemble_pb_examples)
# ------------------------------------------------------------------------------------------------


def reproduce_br_flags(vids_by_variant, scvs_by_vid, *, is_whitelisted, org_of, include_ids) -> dict:
    """(br_lab_any, br_lab_only, br_lab_shared) por variante: so SCVs do filtro P/B, sem repetir a mesma SCV entre
    VariationIDs; uma SCV e marcada brasileira se o submissor resolve para uma instituicao incluida na lista."""
    out = {}
    for variant_id, vids in vids_by_variant.items():
        seen: set[str] = set()
        flags: list[bool] = []
        for vid in vids:
            for scv in scvs_by_vid.get(vid, ()):
                if not is_whitelisted(scv) or scv.scv in seen:
                    continue
                seen.add(scv.scv)
                org_id = org_of(scv.submitter)
                flags.append(bool(org_id and org_id in include_ids))
        any_br = any(flags)
        only_br = bool(flags) and all(flags)
        out[variant_id] = (any_br, bool(any_br and only_br), bool(any_br and not only_br))
    return out


def compare_with_published(reproduced: dict, published) -> tuple[int, list[dict]]:
    """published: (variant_id, any, only, shared). Devolve (n que concordam, lista das divergencias)."""
    agree, diverge = 0, []
    for variant_id, p_any, p_only, p_shared in published:
        pub = (bool(p_any), bool(p_only), bool(p_shared))
        got = reproduced.get(variant_id)
        if got == pub:
            agree += 1
        else:
            diverge.append({"variant_id": variant_id, "publicado": list(pub),
                            "reproduzido": list(got) if got is not None else None})
    return agree, diverge


# ------------------------------------------------------------------------------------------------
# classificacao, instituicoes e evidencia
# ------------------------------------------------------------------------------------------------


def classify(m_any: bool, m_only: bool, m_shared: bool, in_master: bool, v1_br: bool, v1_non_br: bool,
             covered: bool) -> str:
    if not in_master:
        return OUTSIDE_MASTER
    if m_any and not v1_br:
        return A2 if covered else A1
    if v1_br and not m_any:
        return B
    if m_any and v1_br:
        if m_shared and not v1_non_br:
            return C1
        if m_only and v1_non_br:
            return C2
        return BOTH_MARK
    return NONE_MARK_COVERED if covered else NONE_MARK_UNCOVERED


def assign_set(key: str, sets: dict[str, set[str]]) -> str:
    for name in SET_PRIORITY:
        if key in sets.get(name, ()):
            return name
    return "nenhum"


def fold_name(text: str) -> str:
    """Casefold, sem acentos, espacos colapsados. Mantem ';' (separador da lista regional)."""
    text = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def submitter_presence(submitter: str, joined) -> str:
    """Presenca do submissor em `regional_submitters` da v1 (nomes unicos unidos por ';', truncada com ';...(+N)').

    Presenca exata: o nome inteiro entre separadores. Nomes da lista sao unidos sem espaco, entao um nome com '; '
    continua identificavel; ';' sem espaco pode ser dois nomes vizinhos (indicio). Substring e so indicio.
    Lista ausente, ou truncada sem o nome, e indeterminado."""
    if not isinstance(joined, str) or not joined.strip():
        return NO_LIST
    body, truncated = joined, False
    cut = joined.rfind(TRUNCATION_SEP)
    if cut >= 0:
        body, truncated = joined[:cut], True
    name, listed = fold_name(submitter), fold_name(body)
    if not name:
        return NO_LIST
    if f";{name};" in f";{listed};":
        return SEMICOLON_AMBIGUOUS if re.search(r";(?! )", name) else EXACT
    if name in listed:
        return PARTIAL
    return TRUNCATED if truncated else ABSENT


def load_organization_countries(path: Path, canon) -> tuple[dict[str, str], dict]:
    """organization_summary do NCBI -> {nome canonico: pais}. Mesmo nome com paises diferentes vira 'ambiguo'."""
    countries: dict[str, set[str]] = defaultdict(set)
    header, name_i, country_i, n, skipped = None, None, None, 0, 0
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if header is None:
                if line.startswith("#") and "\t" in line:
                    header = [c.strip().lower() for c in line[1:].rstrip("\r\n").split("\t")]
                    if "organization" not in header or "country" not in header:
                        raise SystemExit(f"{path}: cabecalho sem organization/country: {header[:8]}")
                    name_i, country_i = header.index("organization"), header.index("country")
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                skipped += 1
                continue
            n += 1
            countries[canon(fields[name_i])].add(fields[country_i].strip())
    if header is None:
        raise SystemExit(f"{path}: sem cabecalho '#...'")
    out = {k: (next(iter(v)) if len(v) == 1 else "ambiguo") for k, v in countries.items()}
    return out, {"organizacoes": n, "linhas_ignoradas": skipped}


def institution_status(org_id, include_ids, country) -> str:
    """Nao junta 'pais nao resolvido' com 'nao brasileira pelo registro' nem 'fora da lista' com 'nao brasileira'."""
    if org_id and org_id in include_ids:
        return LISTED
    if not country:
        return COUNTRY_UNRESOLVED
    if country == "ambiguo":
        return COUNTRY_AMBIGUOUS
    if country == "Brazil":
        return REGISTRY_BRAZIL_UNLISTED
    return REGISTRY_NOT_BRAZIL


def status_summary(rows) -> str:
    present = {r["status_instituicao"] for r in rows}
    return "+".join(s for s in STATUS_ORDER if s in present) or "nenhuma_scv"


def evidence_rows(vids, scvs_by_vid, *, is_whitelisted, org_of, include_ids, canon, countries, joined) -> list[dict]:
    """Todas as SCVs do release para a variante (qualquer classe), uma vez cada."""
    rows, seen = [], set()
    for vid in vids:
        for scv in scvs_by_vid.get(vid, ()):
            if scv.scv in seen:
                continue
            seen.add(scv.scv)
            org_id = org_of(scv.submitter)
            country = None if countries is None else countries.get(canon(scv.submitter))
            rows.append({
                "variation_id": scv.variation_id,
                "scv": scv.scv,
                "submitter": scv.submitter,
                "clinical_significance": scv.classification,
                "review_status": scv.review_status,
                "origin_counts": scv.origin_counts,
                "contributes_raw": scv.contributes_raw,
                "date_last_evaluated": scv.date_last_evaluated,
                "passa_filtro_pb": bool(is_whitelisted(scv)),
                "org_id_mosaic": org_id,
                "is_br_mosaic": bool(org_id and org_id in include_ids),
                "pais_ncbi": country,
                "status_instituicao": institution_status(org_id, include_ids, country),
                "submissor_na_tabela_v1": submitter_presence(scv.submitter, joined),
            })
    return rows


def table_presence(rows, *, marked_br: bool) -> str:
    """Presenca, na lista da v1, dos submissores das SCVs do filtro P/B do lado pedido (marcadas BR ou nao)."""
    statuses = {r["submissor_na_tabela_v1"] for r in rows if r["passa_filtro_pb"] and r["is_br_mosaic"] == marked_br}
    if EXACT in statuses:
        return "presente_exato"
    if statuses & {PARTIAL, SEMICOLON_AMBIGUOUS}:
        return "so_indicio_parcial_ou_ambiguo"
    if statuses & {TRUNCATED, NO_LIST}:
        return "indeterminado"
    return "ausente" if statuses else "sem_scv_do_filtro_nesse_lado"


def b_evidence(rows) -> str:
    """v1 marca BR, Mosaic nao: flags nao exclusivas; pais nao resolvido nao vira 'nao brasileira'."""
    flags = []
    if any(r["org_id_mosaic"] is not None and not r["passa_filtro_pb"] for r in rows):
        flags.append("scv_da_lista_fora_do_filtro_pb")
    if any(r["status_instituicao"] == REGISTRY_BRAZIL_UNLISTED for r in rows):
        flags.append(REGISTRY_BRAZIL_UNLISTED)
    if flags:
        return "+".join(flags)
    if any(r["status_instituicao"] in UNRESOLVED for r in rows):
        return "pais_nao_resolvido_em_alguma_scv"
    return "so_nao_brasileiras_pelo_registro_ncbi" if rows else "sem_scv_no_release"


def variant_evidence(category: str, rows) -> str:
    if category == A1:
        return "v1_sem_linhas_da_variante"
    if category == A2:
        return "submissor_marcado:" + table_presence(rows, marked_br=True)
    if category == B:
        return b_evidence(rows)
    if category == C1:
        not_marked = [r for r in rows if r["passa_filtro_pb"] and not r["is_br_mosaic"]]
        return (f"submissor_nao_marcado:{table_presence(rows, marked_br=False)}"
                f"|status:{status_summary(not_marked)}")
    if category == C2:
        not_marked = [r for r in rows if not r["is_br_mosaic"]]
        if not not_marked:
            return "sem_scv_nao_marcada_no_release"
        # br_lab_only com SCV nao marcada => ela esta fora do filtro P/B; o status diz se ha pais resolvido
        return "scv_nao_marcada_fora_do_filtro_pb|status:" + status_summary(not_marked)
    return ""


def scan_unlisted_brazil(vids_by_variant, scvs_by_vid, *, is_whitelisted, org_of, include_ids, canon, countries):
    """Todas as SCVs do filtro P/B de TODOS os exemplos: status da instituicao (cache por submissor)."""
    cache: dict[str, str] = {}
    flagged: dict[str, bool] = {}
    per_status: Counter[str] = Counter()
    unlisted: Counter[str] = Counter()
    for variant_id, vids in vids_by_variant.items():
        seen: set[str] = set()
        hit = False
        for vid in vids:
            for scv in scvs_by_vid.get(vid, ()):
                if scv.scv in seen or not is_whitelisted(scv):
                    continue
                seen.add(scv.scv)
                status = cache.get(scv.submitter)
                if status is None:
                    country = None if countries is None else countries.get(canon(scv.submitter))
                    status = institution_status(org_of(scv.submitter), include_ids, country)
                    cache[scv.submitter] = status
                per_status[status] += 1
                if status == REGISTRY_BRAZIL_UNLISTED:
                    hit = True
                    unlisted[scv.submitter] += 1
        flagged[variant_id] = hit
    return flagged, per_status, unlisted


# ------------------------------------------------------------------------------------------------
# nosso lado (v1)
# ------------------------------------------------------------------------------------------------


def read_master(uri: str, columns: list[str]):
    import pyarrow.fs as fs
    import pyarrow.parquet as pq

    if uri.startswith("s3://"):
        path = uri[len("s3://"):]
        filesystem = fs.S3FileSystem(region=fs.resolve_s3_region(path.split("/", 1)[0]))
    else:
        path, filesystem = str(Path(uri).expanduser()), None
    available = set(pq.ParquetFile(path, filesystem=filesystem).schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise SystemExit(f"{uri}: faltam colunas {missing}")
    return pq.read_table(path, columns=columns, filesystem=filesystem).to_pandas()


def our_sets(br: Path, pairs: Path, splits: Path, exclude_chrom: str) -> dict[str, set[str]]:
    import pandas as pd
    import pyarrow.parquet as pq

    br_df, _ = load_slice(br, exclude_chrom)
    sets = {"t_br_slice": set(br_df["variant_key"].astype(str))}
    pairs_df = pd.read_parquet(pairs)
    for side in ("br", "nonbr"):
        col = f"{side}_variant_key"
        if col not in pairs_df.columns:
            raise SystemExit(f"{pairs}: sem {col}")
        sets[f"t_{side}_pareado"] = set(pairs_df[col].astype(str))
    names = set(pq.ParquetFile(splits).schema_arrow.names)
    if not {"variant_key", "split_within_gene"} <= names:
        raise SystemExit(f"{splits}: precisa de variant_key e split_within_gene")
    splits_df = pd.read_parquet(splits, columns=["variant_key", "split_within_gene"])
    for name, grp in splits_df.groupby("split_within_gene"):
        sets[f"split_{name}"] = set(grp["variant_key"].astype(str))
    return sets


def is_snv_key(key: str) -> bool:
    parts = key.split(":")
    return len(parts) == 4 and len(parts[2]) == 1 and len(parts[3]) == 1


def outside_mosaic(sets: dict[str, set[str]], example_keys: set[str]) -> dict:
    out = {}
    for name in TEST_SETS:
        keys = sets.get(name, set())
        outside = [k for k in keys if k not in example_keys]
        snv = sum(1 for k in outside if is_snv_key(k))
        out[name] = {"n": len(keys), "no_pb_examples": len(keys) - len(outside), "fora_snv": snv,
                     "fora_nao_snv": len(outside) - snv}
    return out


def crosstab(rows_a, rows_b) -> dict:
    table: dict[str, Counter] = defaultdict(Counter)
    for a, b in zip(rows_a, rows_b):
        table[str(a)][str(b)] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}


def top_submitters(evidence, categories: tuple[str, ...], select, top: int) -> list:
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for r in evidence:
        if r["categoria"] not in categories or not select(r):
            continue
        k = (r["variant_id"], r["submitter"])
        if k not in seen:
            seen.add(k)
            counts[r["submitter"]] += 1
    return counts.most_common(top)


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    import pyarrow.parquet as pq

    args = parse_args(argv)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    excl = str(args.exclude_chrom).strip().lower().removeprefix("chr")
    report: dict = {"entradas": {k: (str(v) if v is not None else None) for k, v in vars(args).items()}}

    def write_report() -> None:
        path = out_dir / "diagnostico_submissor_br.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\n[diag] -> {path}")

    mosaic = load_mosaic(args.mosaic_root, release=args.clinvar_release, expected_commit=args.expected_mosaic_commit)
    ss_path = args.submission_summary.expanduser()
    report["mosaic"] = {"raiz": str(mosaic.root), "commit": mosaic.commit, "commit_conferido": mosaic.commit_conferido,
                        "release_clinvar": mosaic.release, "instituicoes_incluidas": len(mosaic.include_ids),
                        "freeze_date": mosaic.freeze_date}
    report["submission_summary"] = verify_source(mosaic, ss_path)
    print(f"[diag] Mosaic {mosaic.commit} · {len(mosaic.include_ids)} instituicoes incluidas · "
          f"submission_summary {mosaic.release} conferido por sha256")

    countries = None
    if args.organization_summary is not None:
        org_path = args.organization_summary.expanduser()
        countries, org_meta = load_organization_countries(org_path, mosaic.canon)
        report["organization_summary"] = org_meta | {
            "sha256": mosaic.sha256_file(org_path),
            "nota": "versao atual do NCBI, nao a congelada pelo Mosaic; casada pelo nome canonico",
        }

    names = set(pq.ParquetFile(args.pb_examples).schema_arrow.names)
    missing = [c for c in EXAMPLE_COLUMNS if c not in names]
    if missing:
        raise SystemExit(f"{args.pb_examples}: faltam {missing}")
    ex = pd.read_parquet(args.pb_examples, columns=EXAMPLE_COLUMNS)
    ex["key"] = build_key(ex["chrom"], ex["pos_1based"], ex["ref"], ex["alt"]).astype(str)
    vids_by_variant = {v: [str(x) for x in (vids if vids is not None else ())]
                       for v, vids in zip(ex["variant_id"], ex["clinvar_variation_ids"])}
    keep = {vid for vids in vids_by_variant.values() for vid in vids}
    print(f"[diag] pb_examples: {len(ex):,} exemplos, {len(keep):,} VariationIDs · lendo o submission_summary "
          "(alguns minutos)")
    scvs_by_vid, _ = mosaic.parse_scvs(ss_path, mosaic.spec["submission_summary"], keep)

    reproduced = reproduce_br_flags(vids_by_variant, scvs_by_vid, is_whitelisted=mosaic.is_whitelisted,
                                    org_of=mosaic.org_of, include_ids=mosaic.include_ids)
    agree, divergent = compare_with_published(
        reproduced, zip(ex["variant_id"], ex["br_lab_any"], ex["br_lab_only"], ex["br_lab_shared"]))
    report["reproducao"] = {"exemplos": len(ex), "concordam": agree, "divergem": len(divergent)}
    print(f"\n[1] reproducao do is_br do Mosaic: {agree:,} de {len(ex):,} exemplos batem com o publicado")
    if divergent:
        report["reproducao"]["status"] = "NAO_FIEL: parar; pedir a pasta interim/ do build ao Eduardo"
        report["reproducao"]["amostra"] = divergent[:50]
        (out_dir / "reproducao_divergente.json").write_text(json.dumps(divergent, ensure_ascii=False), encoding="utf-8")
        print(f"    -> {len(divergent):,} divergem: reproducao NAO fiel, diagnostico interrompido")
        write_report()
        return 2
    report["reproducao"]["status"] = "fiel (compatibilidade com as flags publicadas, nao correcao geografica)"

    master = read_master(args.master_uri, MASTER_COLUMNS)
    master["variant_key"] = master["variant_key"].astype(str)
    master = master.drop_duplicates("variant_key")
    sets = our_sets(args.br, args.pairs, args.splits, excl)
    membership = pd.read_parquet(args.membership, columns=["variant_id", "study_id", "member_role"])
    roles = (membership.assign(papel=membership["study_id"].astype(str) + "/" + membership["member_role"].astype(str))
             .groupby("variant_id")["papel"].agg(lambda s: "+".join(sorted(set(s)))))

    df = ex.drop(columns=["clinvar_variation_ids"]).merge(master, left_on="key", right_on="variant_key", how="left",
                                                          indicator=True)
    df["no_master_v1"] = df["_merge"].eq("both")
    df["v1_br"] = as_bool(df["has_brazilian_submitter"])
    df["v1_linha_nao_br"] = as_bool(df["has_non_brazilian_submitter"])
    df["coberta_v1"] = pd.to_numeric(df["regional_submission_rows"], errors="coerce").fillna(0) > 0
    df["conjunto_v1"] = [assign_set(k, sets) for k in df["key"]]
    df["papel_mosaic"] = df["variant_id"].map(roles).fillna("nenhum")
    flags = [reproduced[v] for v in df["variant_id"]]
    df["mosaic_marca_br"] = [f[0] for f in flags]
    df["mosaic_so_marcadas"] = [f[1] for f in flags]
    df["mosaic_compartilhada"] = [f[2] for f in flags]
    df["categoria"] = [
        classify(a, o, s, m, b, nb, c)
        for a, o, s, m, b, nb, c in zip(df["mosaic_marca_br"], df["mosaic_so_marcadas"], df["mosaic_compartilhada"],
                                        df["no_master_v1"], df["v1_br"], df["v1_linha_nao_br"], df["coberta_v1"])
    ]

    if countries is not None:
        scan_flags, per_status, unlisted = scan_unlisted_brazil(
            vids_by_variant, scvs_by_vid, is_whitelisted=mosaic.is_whitelisted, org_of=mosaic.org_of,
            include_ids=mosaic.include_ids, canon=mosaic.canon, countries=countries)
        df["scv_pb_pais_brazil_fora_da_lista"] = df["variant_id"].map(scan_flags).map(
            {True: "sim", False: "nao"}).fillna("nao_verificado")
        report["escopo_lista_br"] = {
            "nota": ("SCVs do filtro P/B de TODOS os exemplos, inclusive onde os dois concordam; pais do "
                     "organization_summary atual casado por nome. Nao valida a lista brasileira de forma completa."),
            "scvs_do_filtro_por_status": dict(sorted(per_status.items())),
            "variantes_por_categoria": crosstab(df["categoria"], df["scv_pb_pais_brazil_fora_da_lista"]),
            "submissores_pais_brazil_fora_da_lista": unlisted.most_common(args.top),
        }
    else:
        df["scv_pb_pais_brazil_fora_da_lista"] = "nao_verificado"
        report["escopo_lista_br"] = {"nota": "sem --organization-summary: varredura de escopo nao executada"}

    evidence: list[dict] = []
    per_variant: dict[str, str] = {}
    for row in df[~df["categoria"].isin(NOT_DIVERGENT)].itertuples(index=False):
        rows = evidence_rows(vids_by_variant[row.variant_id], scvs_by_vid, is_whitelisted=mosaic.is_whitelisted,
                             org_of=mosaic.org_of, include_ids=mosaic.include_ids, canon=mosaic.canon,
                             countries=countries, joined=row.regional_submitters)
        per_variant[row.variant_id] = variant_evidence(row.categoria, rows)
        for r in rows:
            r.update(variant_id=row.variant_id, key=row.key, categoria=row.categoria, conjunto_v1=row.conjunto_v1,
                     papel_mosaic=row.papel_mosaic, binary_label=int(row.binary_label))
        evidence.extend(rows)
    df["evidencia"] = df["variant_id"].map(per_variant).fillna("")

    divergent_df = df[~df["categoria"].isin(NOT_DIVERGENT)]
    label = df["binary_label"].map({1: "P", 0: "B"}).fillna("?")
    members = df["papel_mosaic"] != "nenhum"
    report["definicoes"] = DEFINITIONS
    report["categorias"] = dict(sorted(Counter(df["categoria"]).items()))
    report["categorias_por_conjunto_v1"] = crosstab(df["conjunto_v1"], df["categoria"])
    report["categorias_por_papel_mosaic"] = crosstab(df["papel_mosaic"], df["categoria"])
    report["categorias_por_rotulo"] = crosstab(df["categoria"], label)
    report["evidencia_por_categoria"] = crosstab(divergent_df["categoria"], divergent_df["evidencia"])
    report["membros_mosaic_por_conjunto_v1"] = crosstab(df.loc[members, "papel_mosaic"], df.loc[members, "conjunto_v1"])
    report["submissores"] = {
        "A_marcados_br_pelo_mosaic": top_submitters(
            evidence, (A1, A2), lambda r: r["is_br_mosaic"] and r["passa_filtro_pb"], args.top),
        "B_da_lista_fora_do_filtro_pb": top_submitters(
            evidence, (B,), lambda r: r["org_id_mosaic"] is not None and not r["passa_filtro_pb"], args.top),
        "B_pais_brazil_no_ncbi_fora_da_lista": top_submitters(
            evidence, (B,), lambda r: r["status_instituicao"] == REGISTRY_BRAZIL_UNLISTED, args.top),
        "C1_nao_marcados_no_filtro_pb": top_submitters(
            evidence, (C1,), lambda r: (not r["is_br_mosaic"]) and r["passa_filtro_pb"], args.top),
    }
    report["status_instituicao_nas_scvs_divergentes"] = dict(sorted(Counter(r["status_instituicao"] for r in evidence).items()))
    report["presenca_na_tabela_v1_nas_scvs_divergentes"] = dict(sorted(Counter(r["submissor_na_tabela_v1"] for r in evidence).items()))
    report["fora_do_mosaic"] = outside_mosaic(sets, set(df["key"]))
    report["ressalvas"] = [
        "Reproducao fiel confirma compatibilidade com as flags publicadas, nao a correcao geografica das instituicoes.",
        "Categorias descrevem MARCACOES dos dois lados; 'nenhum_marca_br' nao e nacionalidade nao brasileira.",
        "Presenca na tabela v1: so 'presente_exato' e presenca; indicio e indeterminado nao sao evidencia de mapeamento.",
        "Pais do organization_summary atual do NCBI, casado por nome; 'pais_nao_resolvido' nao prova nada.",
        "A evidencia por SCV cobre so as divergentes; a varredura de escopo so ve SCVs do filtro P/B dos exemplos e "
        "instituicoes com registro Brazil. A lista brasileira nao fica validada de forma completa.",
        "Release da tabela regional v1 desconhecido: ausencia da submissao nao separa release de cobertura.",
    ]

    keep_cols = ["variant_id", "key", "label_tier", "binary_label", "mosaic_marca_br", "mosaic_so_marcadas",
                 "mosaic_compartilhada", "no_master_v1", "v1_br", "v1_linha_nao_br", "coberta_v1",
                 "regional_submission_rows", "clinvar_regional_cohort", "regional_submitters", "conjunto_v1",
                 "papel_mosaic", "categoria", "evidencia", "scv_pb_pais_brazil_fora_da_lista"]
    divergent_df[keep_cols].to_parquet(out_dir / "variantes_divergentes.parquet", index=False)
    ev = pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS)
    ev.to_parquet(out_dir / "evidencia_scv.parquet", index=False)
    ev.to_csv(out_dir / "evidencia_scv.csv", index=False)

    print(f"\n[2] categorias (todos os {len(df):,} exemplos do Mosaic; marcacoes, nao nacionalidade)")
    for cat, n in report["categorias"].items():
        print(f"    {cat:<58} {n:>8,}")
    print("\n[3] evidencia por categoria divergente")
    for cat, blk in report["evidencia_por_categoria"].items():
        print(f"    {cat}")
        for ev_name, n in sorted(blk.items(), key=lambda t: -t[1]):
            print(f"        {ev_name:<86} {n:>6,}")
    print("\n[4] categorias divergentes por conjunto v1")
    for conj, blk in report["categorias_por_conjunto_v1"].items():
        shown = {k: v for k, v in blk.items() if k not in NOT_DIVERGENT}
        if shown:
            print(f"    {conj:<20} " + "  ".join(f"{k.split('_')[0]}={v:,}" for k, v in shown.items()))
    print("\n[5] escopo da lista brasileira (SCVs do filtro P/B de todos os exemplos)")
    scope = report["escopo_lista_br"]
    if "scvs_do_filtro_por_status" in scope:
        print(f"    SCVs por status: {scope['scvs_do_filtro_por_status']}")
        print("    submissores com pais Brazil no NCBI fora da lista: "
              + "; ".join(f"{s} ({n})" for s, n in scope["submissores_pais_brazil_fora_da_lista"][:10]))
    else:
        print(f"    {scope['nota']}")
    print("\n[6] submissores nas divergentes")
    for name, pairs in report["submissores"].items():
        print(f"    {name}: " + "; ".join(f"{s} ({n})" for s, n in pairs[:8]))
    print(f"\n[7] presenca na tabela v1 (SCVs das divergentes): {report['presenca_na_tabela_v1_nas_scvs_divergentes']}")
    print(f"\n[8] fora do Mosaic: {report['fora_do_mosaic']}")
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
