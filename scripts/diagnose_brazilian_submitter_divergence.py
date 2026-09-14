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
2. Cruza cada exemplo do Mosaic com o master regional v1 (has_brazilian_submitter, cobertura, submissores) e com
   os conjuntos v1 (T_BR, pares, splits).
3. Classifica cada divergencia por motivo e escreve a evidencia por SCV:
     A1  Mosaic BR; a tabela regional v1 nao tinha nenhuma linha da variante (cobertura)
     A2  Mosaic BR; a tabela cobria a variante, sem linha brasileira (mapeamento, release)
     B   tabela v1 BR; Mosaic nao (classe de SCV, submissor nao reconhecido, sem evidencia no release)
     C1  as duas BR; v1 so brasileira, Mosaic compartilhada
     C2  as duas BR; v1 mista, Mosaic so brasileira

O QUE NAO PROVA
---------------
- A tabela regional v1 nao guarda o cohort por submissor e o eval_unified nao esta no notebook: "submissor
  presente na tabela" e busca por nome canonico na lista `regional_submitters` (aproximada; truncada em 20).
- O pais vem do organization_summary ATUAL do NCBI, nao do congelado pelo Mosaic (2026-08-22); nome de
  submissor e nome de organizacao nem sempre coincidem (sem registro nao e prova de nada).
- O release da tabela regional v1 e desconhecido: "submissao ausente da tabela" nao separa release de cobertura.
- Nenhuma correcao automatica: o relatorio alimenta a definicao e a reconstrucao dos dados.

USO (notebook)
--------------
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
        --out-dir ~/artifacts/redesenho/diagnostico_submissor_br
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_regionalization_data import as_bool, build_key, load_slice  # noqa: E402

EXPECTED_MOSAIC_COMMIT = "814e7f0"
DEFAULT_RELEASE = "2026-06"
TRUNCATION_MARK = "...(+"
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

A1 = "A1_mosaic_br_sem_cobertura_na_tabela_v1"
A2 = "A2_mosaic_br_tabela_v1_sem_linha_br"
B = "B_tabela_v1_br_mosaic_nao"
C1 = "C1_v1_so_br_mosaic_compartilhada"
C2 = "C2_v1_mista_mosaic_so_br"
AGREE_BR = "concordam_br"
AGREE_NON_BR = "concordam_nao_br"
OUTSIDE_MASTER = "fora_do_master_v1"
NOT_DIVERGENT = (AGREE_BR, AGREE_NON_BR, OUTSIDE_MASTER)


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
    VariationIDs; uma SCV e brasileira se o submissor resolve para uma instituicao incluida na lista."""
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
# classificacao e evidencia
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
        return AGREE_BR
    return AGREE_NON_BR


def assign_set(key: str, sets: dict[str, set[str]]) -> str:
    for name in SET_PRIORITY:
        if key in sets.get(name, ()):
            return name
    return "nenhum"


def submitter_in_list(submitter_canon: str, joined, canon) -> bool | None:
    """O submissor esta na lista `regional_submitters` (nomes unidos por ';', truncada com ';...(+N)')?
    None quando a lista esta truncada e o nome nao aparece: nao da para afirmar ausencia."""
    if not isinstance(joined, str) or not joined.strip():
        return False
    if submitter_canon and submitter_canon in canon(joined):
        return True
    return None if TRUNCATION_MARK in joined else False


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
    """Separa 'nao reconhecida pelo matcher' de 'nao brasileira pelo registro do NCBI'."""
    if org_id and org_id in include_ids:
        return "brasileira_na_lista_do_mosaic"
    if not country:
        return "sem_registro_de_pais"
    if country == "ambiguo":
        return "pais_ambiguo"
    if country == "Brazil":
        return "brasileira_nao_reconhecida_pelo_matcher"
    return "nao_brasileira_pelo_registro"


def evidence_rows(vids, scvs_by_vid, *, is_whitelisted, org_of, include_ids, canon, countries, joined) -> list[dict]:
    """Todas as SCVs do release para a variante (qualquer classe), uma vez cada."""
    rows, seen = [], set()
    for vid in vids:
        for scv in scvs_by_vid.get(vid, ()):
            if scv.scv in seen:
                continue
            seen.add(scv.scv)
            org_id = org_of(scv.submitter)
            name = canon(scv.submitter)
            country = None if countries is None else countries.get(name)
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
                "submissor_na_tabela_v1": submitter_in_list(name, joined, canon),
            })
    return rows


def table_presence(rows, *, brazilian: bool) -> str:
    """As SCVs do filtro P/B do lado pedido (brasileiras ou nao) aparecem na lista de submissores da tabela v1?"""
    hits = [r["submissor_na_tabela_v1"] for r in rows if r["passa_filtro_pb"] and r["is_br_mosaic"] == brazilian]
    side = "br" if brazilian else "nao_br"
    if any(h is True for h in hits):
        return f"submissor_{side}_presente_na_tabela_v1"
    if any(h is None for h in hits):
        return "indeterminado_lista_truncada"
    return f"submissor_{side}_ausente_da_tabela_v1"


def b_evidence(rows) -> str:
    """Tabela v1 BR, Mosaic nao: o que o release mostra de brasileiro na variante (flags nao exclusivas)."""
    flags = []
    if any(r["org_id_mosaic"] is not None and not r["passa_filtro_pb"] for r in rows):
        flags.append("instituicao_da_lista_fora_do_filtro_pb")
    if any(r["status_instituicao"] == "brasileira_nao_reconhecida_pelo_matcher" for r in rows):
        flags.append("submissor_brasileiro_nao_reconhecido")
    return "+".join(flags) if flags else "sem_evidencia_brasileira_no_release"


def c2_evidence(rows) -> str:
    """Tabela v1 mista, Mosaic so BR: se ha SCV nao brasileira, ela necessariamente fica fora do filtro P/B."""
    if any(not r["is_br_mosaic"] for r in rows):
        return "nao_br_so_fora_do_filtro_pb"
    return "sem_scv_nao_br_no_release"


def variant_evidence(category: str, rows) -> str:
    if category == A1:
        return "tabela_v1_sem_linhas_da_variante"
    if category == A2:
        return table_presence(rows, brazilian=True)
    if category == B:
        return b_evidence(rows)
    if category == C1:
        return table_presence(rows, brazilian=False)
    if category == C2:
        return c2_evidence(rows)
    return ""


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
            "nota": "versao atual do NCBI, nao a congelada pelo Mosaic; nome de submissor pode diferir do da organizacao",
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
    report["reproducao"]["status"] = "fiel"

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
    df["v1_nao_br"] = as_bool(df["has_non_brazilian_submitter"])
    df["coberta_v1"] = pd.to_numeric(df["regional_submission_rows"], errors="coerce").fillna(0) > 0
    df["conjunto_v1"] = [assign_set(k, sets) for k in df["key"]]
    df["papel_mosaic"] = df["variant_id"].map(roles).fillna("nenhum")
    flags = [reproduced[v] for v in df["variant_id"]]
    df["m_any"] = [f[0] for f in flags]
    df["m_only"] = [f[1] for f in flags]
    df["m_shared"] = [f[2] for f in flags]
    df["categoria"] = [
        classify(a, o, s, m, b, nb, c)
        for a, o, s, m, b, nb, c in zip(df["m_any"], df["m_only"], df["m_shared"], df["no_master_v1"], df["v1_br"],
                                        df["v1_nao_br"], df["coberta_v1"])
    ]

    evidence: list[dict] = []
    per_variant: dict[str, str] = {}
    divergent_df = df[~df["categoria"].isin(NOT_DIVERGENT)]
    for row in divergent_df.itertuples(index=False):
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
    report["categorias"] = dict(sorted(Counter(df["categoria"]).items()))
    report["categorias_por_conjunto_v1"] = crosstab(df["conjunto_v1"], df["categoria"])
    report["categorias_por_papel_mosaic"] = crosstab(df["papel_mosaic"], df["categoria"])
    report["categorias_por_rotulo"] = crosstab(df["categoria"], label)
    report["evidencia_por_categoria"] = crosstab(divergent_df["categoria"], divergent_df["evidencia"])
    report["membros_mosaic_por_conjunto_v1"] = crosstab(df.loc[df["papel_mosaic"] != "nenhum", "papel_mosaic"],
                                                        df.loc[df["papel_mosaic"] != "nenhum", "conjunto_v1"])
    report["submissores"] = {
        "A_instituicoes_br_do_mosaic": top_submitters(
            evidence, (A1, A2), lambda r: r["is_br_mosaic"] and r["passa_filtro_pb"], args.top),
        "B_instituicao_da_lista_fora_do_filtro_pb": top_submitters(
            evidence, (B,), lambda r: r["org_id_mosaic"] is not None and not r["passa_filtro_pb"], args.top),
        "B_brasileiros_nao_reconhecidos": top_submitters(
            evidence, (B,), lambda r: r["status_instituicao"] == "brasileira_nao_reconhecida_pelo_matcher", args.top),
        "C1_nao_br_do_filtro_pb": top_submitters(
            evidence, (C1,), lambda r: (not r["is_br_mosaic"]) and r["passa_filtro_pb"], args.top),
    }
    report["status_instituicao_nas_scvs_divergentes"] = dict(Counter(r["status_instituicao"] for r in evidence))
    report["fora_do_mosaic"] = outside_mosaic(sets, set(df["key"]))
    report["ressalvas"] = [
        "Reproducao validada contra os flags publicados; sem isso nada abaixo vale.",
        "submissor_na_tabela_v1 e busca por nome canonico em regional_submitters (aproximada; None = lista truncada).",
        "Pais do organization_summary atual do NCBI, nao do congelado pelo Mosaic; sem registro nao prova nada.",
        "Release da tabela regional v1 desconhecido: ausencia da submissao nao separa release de cobertura.",
        "Categorias descrevem as duas definicoes; nao dizem qual esta certa. Nenhuma correcao automatica.",
    ]

    keep_cols = ["variant_id", "key", "label_tier", "binary_label", "m_any", "m_only", "m_shared", "no_master_v1",
                 "v1_br", "v1_nao_br", "coberta_v1", "regional_submission_rows", "clinvar_regional_cohort",
                 "regional_submitters", "conjunto_v1", "papel_mosaic", "categoria", "evidencia"]
    divergent_df[keep_cols].to_parquet(out_dir / "variantes_divergentes.parquet", index=False)
    ev = pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS)
    ev.to_parquet(out_dir / "evidencia_scv.parquet", index=False)
    ev.to_csv(out_dir / "evidencia_scv.csv", index=False)

    print(f"\n[2] categorias (todos os {len(df):,} exemplos do Mosaic)")
    for cat, n in report["categorias"].items():
        print(f"    {cat:<44} {n:>8,}")
    print("\n[3] evidencia por categoria divergente")
    for cat, blk in report["evidencia_por_categoria"].items():
        print(f"    {cat}")
        for ev_name, n in sorted(blk.items(), key=lambda t: -t[1]):
            print(f"        {ev_name:<60} {n:>6,}")
    print("\n[4] categorias por conjunto v1 (so as divergentes)")
    for conj, blk in report["categorias_por_conjunto_v1"].items():
        shown = {k: v for k, v in blk.items() if k not in NOT_DIVERGENT}
        if shown:
            print(f"    {conj:<20} " + "  ".join(f"{k.split('_')[0]}={v:,}" for k, v in shown.items()))
    print("\n[5] submissores mais frequentes")
    for name, pairs in report["submissores"].items():
        print(f"    {name}: " + "; ".join(f"{s} ({n})" for s, n in pairs[:8]))
    print(f"\n[6] fora do Mosaic: {report['fora_do_mosaic']}")
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
