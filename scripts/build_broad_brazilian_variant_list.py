#!/usr/bin/env python3
"""Lista da REGRA AMPLA brasileira: toda variante com qualquer SCV de instituicao da lista do Mosaic.

Roda no notebook (pandas + pyarrow + pyyaml; sem GPU). So le; so escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secoes 4.2 (exclusao 2) e 6.4.

POR QUE EXISTE
--------------
O `br_lab_any` publicado no release usa o filtro P/B do Mosaic: SCV que contribui para o agregado, germinativa e
com direcao P ou B. O diagnostico de 14-15/09 mostrou que isso deixa de fora 1.336 SCVs brasileiras (959 por
origem nao germinativa, 630 por nao contribuir). Para EXCLUIR do treino nao queremos o filtro: qualquer submissao
brasileira ja e motivo de exclusao, porque a duvida e sobre exposicao, nao sobre rotulo. Ausencia de informacao
nao vira "nao brasileira".

Esta lista tem dois usos declarados no plano:
  1. exclusao do snapshot de treino da cabeca (G2, `--broad-br-variant-ids`);
  2. a analise de sensibilidade da secao 6.4: quantos CONTROLES do estudo clinico carregam alguma SCV brasileira.

COMO
----
Reusa o carregamento do Mosaic do diagnostico (mesmo clone, mesmo commit conferido, mesmo `submission_summary`
com sha256 conferido contra `config/sources.yaml`) e percorre as SCVs de cada VariationID publicada em
`pb_examples.clinvar_variation_ids`. Marca a variante se o submissor de QUALQUER SCV resolver para uma
instituicao `include: true` da lista brasileira -- sem filtro de classificacao, origem ou contribuicao.

O QUE NAO PROVA
---------------
- Marcacao nao e nacionalidade: o matcher do Mosaic so indexa instituicoes incluidas, entao "nao reconhecida" e
  "nao brasileira" dao o mesmo resultado. O diagnostico mediu 15% de SCVs P/B com pais nao resolvido no NCBI.
- A lista e do freeze do Mosaic (2026-08-22); instituicoes brasileiras fora dela nao sao marcadas.

USO (notebook)
--------------
    set -o pipefail
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/build_broad_brazilian_variant_list.py \
        --mosaic-root ~/testeArq/lumina-mosaic \
        --submission-summary ~/clinvar/2026-06/submission_summary_2026-06.txt.gz \
        --pb-examples ~/mosaic-v1/pb_examples.parquet \
        --membership ~/mosaic-v1/studies/brazil/membership.parquet \
        --out-dir ~/artifacts/redesenho/g2_regra_ampla | tee ~/regra_ampla.log
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_brazilian_submitter_divergence import (  # noqa: E402
    EXPECTED_MOSAIC_COMMIT,
    load_mosaic,
    verify_source,
)
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

ROLE_CONTROL = "control"
STUDY_CLINICAL = "br_clinical_evidence"


def broad_br_variants(vids_by_variant, scvs_by_vid, *, org_of, include_ids) -> dict[str, list[str]]:
    """variant_id -> submissores brasileiros encontrados, SEM filtro de classe, origem ou contribuicao."""
    out: dict[str, list[str]] = {}
    for variant_id, vids in vids_by_variant.items():
        seen: set[str] = set()
        submitters: list[str] = []
        for vid in vids:
            for scv in scvs_by_vid.get(vid, ()):
                if scv.scv in seen:
                    continue
                seen.add(scv.scv)
                org_id = org_of(scv.submitter)
                if org_id and org_id in include_ids:
                    submitters.append(scv.submitter)
        if submitters:
            out[variant_id] = submitters
    return out


def compare_with_published(broad: dict[str, list[str]], published_any: dict[str, bool]) -> dict[str, Any]:
    """Quanto a regra ampla acrescenta ao `br_lab_any` do release (que passa pelo filtro P/B)."""
    marked = {vid for vid, flag in published_any.items() if flag}
    broad_ids = set(broad)
    return {
        "regra_ampla": len(broad_ids),
        "br_lab_any_publicado": len(marked),
        "so_na_regra_ampla": len(broad_ids - marked),
        "so_no_br_lab_any": len(marked - broad_ids),
        "nota_so_no_br_lab_any": "esperado 0: o filtro P/B e um subconjunto da regra ampla",
    }


def validate_broad_result(broad: dict[str, list[str]], published_any: dict[str, bool]) -> list[str]:
    """A regra ampla TEM de conter tudo que o release ja marca: o filtro P/B e um subconjunto dela.

    Se alguma variante com `br_lab_any` ficar de fora, algo esta errado (parsing, lista de instituicoes, release
    trocado) e publicar a lista mesmo assim faria o G2 excluir de menos.
    """
    problems: list[str] = []
    if not broad:
        problems.append("nenhuma variante marcada: parsing ou lista de instituicoes provavelmente errados")
    missing = sorted({vid for vid, flag in published_any.items() if flag} - set(broad))
    if missing:
        problems.append(
            f"{len(missing)} variantes com br_lab_any no release ficaram fora da regra ampla, ex.: {missing[:3]}"
        )
    return problems


def controls_with_brazilian_scv(membership, broad: dict[str, list[str]]) -> dict[str, Any]:
    """Secao 6.4: controles do estudo clinico que carregam alguma SCV brasileira pela regra ampla."""
    controls = membership[(membership["study_id"] == STUDY_CLINICAL)
                          & (membership["member_role"] == ROLE_CONTROL)]
    hit = [vid for vid in controls["variant_id"] if vid in broad]
    return {
        "controles_do_estudo_clinico": int(len(controls)),
        "com_scv_brasileira_regra_ampla": len(hit),
        "fracao": round(len(hit) / len(controls), 4) if len(controls) else None,
        "exemplos": hit[:5],
        "nota": "analise de sensibilidade pre-declarada: repetir a interacao sem os pares desses controles, "
                "sem refazer o pareamento; a direcao do efeito nao e assumida",
    }


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mosaic-root", required=True, type=Path)
    parser.add_argument("--submission-summary", required=True, type=Path)
    parser.add_argument("--pb-examples", required=True, type=Path)
    parser.add_argument("--membership", type=Path, help="para a contagem de controles da secao 6.4")
    parser.add_argument("--clinvar-release", default="2026-06")
    parser.add_argument("--expected-mosaic-commit", default=EXPECTED_MOSAIC_COMMIT)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    mosaic = load_mosaic(args.mosaic_root, release=args.clinvar_release,
                         expected_commit=args.expected_mosaic_commit)
    ss_path = args.submission_summary.expanduser()
    source = verify_source(mosaic, ss_path)
    print(f"[ampla] Mosaic {mosaic.commit} · {len(mosaic.include_ids)} instituicoes incluidas · "
          f"submission_summary conferido por sha256")

    columns = ["variant_id", "clinvar_variation_ids", "br_lab_any"]
    examples = pd.read_parquet(args.pb_examples.expanduser(), columns=columns)
    vids_by_variant = {
        vid: [str(x) for x in (vids if vids is not None else ())]
        for vid, vids in zip(examples["variant_id"], examples["clinvar_variation_ids"])
    }
    keep = {vid for vids in vids_by_variant.values() for vid in vids}
    print(f"[ampla] pb_examples: {len(examples):,} exemplos, {len(keep):,} VariationIDs · lendo o "
          "submission_summary (alguns minutos)")
    scvs_by_vid, _ = mosaic.parse_scvs(ss_path, mosaic.spec["submission_summary"], keep)

    broad = broad_br_variants(vids_by_variant, scvs_by_vid, org_of=mosaic.org_of,
                              include_ids=mosaic.include_ids)
    published = dict(zip(examples["variant_id"], examples["br_lab_any"].astype(bool)))

    report: dict[str, Any] = {
        "entradas": {"mosaic_root": str(mosaic.root), "commit": mosaic.commit,
                     "submission_summary": source, "pb_examples": str(args.pb_examples)},
        "regra": "qualquer SCV cujo submissor resolve para instituicao include:true da lista brasileira do Mosaic, "
                 "sem filtro de classificacao, origem ou contribuicao",
        "comparacao_com_o_publicado": compare_with_published(broad, published),
        "submissores_mais_frequentes": Counter(
            name for names in broad.values() for name in names
        ).most_common(15),
        "o_que_nao_prova": [
            "marcacao nao e nacionalidade: o matcher so indexa instituicoes incluidas",
            "instituicoes brasileiras fora do freeze de 2026-08-22 nao sao marcadas",
        ],
    }

    if args.membership is not None:
        membership = pd.read_parquet(args.membership.expanduser(),
                                     columns=["variant_id", "study_id", "member_role"])
        report["controles_com_scv_brasileira"] = controls_with_brazilian_scv(membership, broad)

    list_path = out_dir / "broad_brazilian_variant_ids.txt"
    manifest_path = out_dir / "broad_brazilian_variant_ids.txt.manifest.json"
    report_path = out_dir / "regra_ampla_brasileira.json"

    problems = validate_broad_result(broad, published)
    if problems:
        report["status"] = "FALHOU"
        report["problemas"] = problems
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"FALHOU: {len(problems)} problema(s); a lista NAO foi publicada.")
        for problem in problems:
            print(f"  - {problem}")
        if list_path.exists() or manifest_path.exists():
            print(f"  ATENCAO: ha arquivos antigos em {out_dir}; apague antes de usar no G2.")
        return 2

    pb_examples_sha256 = sha256_file(args.pb_examples.expanduser())
    list_path.write_text("\n".join(sorted(broad)) + "\n", encoding="utf-8")
    manifest = {
        "regra": report["regra"],
        "n_variantes": len(broad),
        "lista_sha256": sha256_file(list_path),
        "pb_examples": str(args.pb_examples),
        "pb_examples_sha256": pb_examples_sha256,
        "submission_summary_sha256": source.get("sha256"),
        "clinvar_release": args.clinvar_release,
        "mosaic_commit": mosaic.commit,
        "instituicoes_incluidas": len(mosaic.include_ids),
        "superset_do_br_lab_any": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report["status"] = "OK"
    report["manifesto"] = manifest
    report["saidas"] = {"lista": str(list_path), "manifesto": str(manifest_path), "relatorio": str(report_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("comparacao_com_o_publicado", "saidas")
                      if k in report}, ensure_ascii=False, indent=2))
    if "controles_com_scv_brasileira" in report:
        print(json.dumps(report["controles_com_scv_brasileira"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
