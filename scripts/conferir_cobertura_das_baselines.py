#!/usr/bin/env python3
"""Baselines do G7: confere, SEM calcular metrica, se as anotacoes do release cobrem os membros dos estudos.

POR QUE
    A revisao de 24/09 recusou retirar a baseline de frequencia continua porque "nao esta na membership": o release
    do Mosaic traz `pb_annotations.parquet`, uma linha por `variant_id` da suite, com `gnomad_v4_af`, `gnomad_status`,
    `abraom_af`, `abraom_status` e `present_abraom`, e o comparador OFICIAL `gnomad_rarity` do Mosaic sai dessas
    colunas (`config/comparators.yaml`: -gnomad_v4_af, not_found e ac0 com AF 0). Antes de escrever a baseline no
    consumidor, este script confere:
      - PROVENIENCIA: o hash logico de `pb_annotations.parquet` e de `studies/brazil/membership.parquet`,
        recalculado com o `logical_contract` do proprio Mosaic, confere com a referencia declarada
        (`g6.proveniencia.release_do_mosaic`: os invariantes da ADR 0006, iguais nos dois layouts) e, se houver, com o
        manifesto da copia (`release.manifest.json` ou, no layout anterior a ADR 0006, `bundle.manifest.json`); e a
        especificacao do `gnomad_rarity` no `comparators.yaml` e a declarada (coluna, formula, regra de ausente);
      - COBERTURA: por estudo e papel, quantos membros tem linha de anotacao, `gnomad_rarity` definido pela regra
        oficial e `abraom_af` finito; a distribuicao de `gnomad_status` e de `abraom_status`; e se a `present_abraom`
        da membership e a das anotacoes concordam.

    So CONTAGENS. Nenhum rotulo e cruzado com anotacao ou score e nenhuma metrica e calculada: a baseline nos estudos
    so se mede no G7, depois do congelamento.

USO (notebook)
    PYTHONPATH="$PWD" python3 scripts/conferir_cobertura_das_baselines.py --release-root ~/mosaic-v1 \\
        --mosaic-root ~/testeArq/lumina-mosaic --out ~/artifacts/redesenho/g6_baselines/cobertura.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.recortes import carregar_campanha  # noqa: E402

ANOTACOES = "pb_annotations.parquet"
MEMBERSHIP = "studies/brazil/membership.parquet"
CHAVES = {ANOTACOES: ("variant_id",), MEMBERSHIP: ("variant_id", "study_id")}
COLUNAS_DAS_ANOTACOES = ("variant_id", "gnomad_v4_af", "gnomad_status", "abraom_af", "abraom_status", "present_abraom")
#: A especificacao do comparador oficial que a baseline usa, como declarada em `analises_secundarias`.
GNOMAD_RARITY = {"id": "gnomad_rarity", "column": "gnomad_v4_af", "formula": "neg_gnomad_v4_af",
                 "missing": "impute_af0_for_not_found_and_ac0"}
CAMPOS_DO_CONTRATO = ("logical_hash_version", "logical_hash", "n", "schema", "primary_key")
#: O manifesto da copia do release: o nome novo (ADR 0006) ou o do layout anterior, em que o S3 ainda esta.
MANIFESTOS = ("release.manifest.json", "bundle.manifest.json")


def comparar_contrato(publicado: dict[str, Any] | None, recalculado: dict[str, Any], caminho: str,
                      campos: tuple[str, ...] = CAMPOS_DO_CONTRATO, origem: str = "manifesto") -> list[str]:
    """Campos do contrato de referencia que o recalculado nao reproduz. Pura."""
    if publicado is None:
        return [f"{caminho}: sem contrato no {origem}"]
    return [f"{caminho}: {campo} recalculado != {origem}" for campo in campos
            if publicado.get(campo) != recalculado.get(campo)]


def manifesto_da_copia(raiz: Path) -> tuple[str | None, dict[str, Any]]:
    """(nome, conteudo) do manifesto da copia, no layout novo ou no anterior; (None, {}) se nao houver."""
    for nome in MANIFESTOS:
        if (raiz / nome).is_file():
            return nome, json.loads((raiz / nome).read_text(encoding="utf-8"))
    return None, {}


def conferir_especificacao(spec: dict[str, Any] | None) -> list[str]:
    """A especificacao do `gnomad_rarity` no comparators.yaml tem de ser a declarada. Pura."""
    if spec is None:
        return ["gnomad_rarity ausente dos comparadores oficiais"]
    return [f"gnomad_rarity.{campo}: {spec.get(campo)!r} != {valor!r}" for campo, valor in GNOMAD_RARITY.items()
            if spec.get(campo) != valor]


def _contagem(serie: pd.Series) -> dict[str, int]:
    return {("ausente" if pd.isna(k) else str(k)): int(v) for k, v in serie.value_counts(dropna=False).items()}


def cobertura(membros: pd.DataFrame, anotacoes: pd.DataFrame,
              pontuar: Callable[[dict[str, Any]], float | None]) -> dict[str, Any]:
    """Contagens por estudo e papel -- nenhum rotulo, nenhuma metrica. `pontuar` e o `comparator_score` do Mosaic com
    a especificacao oficial do `gnomad_rarity`. Pura."""
    if not anotacoes["variant_id"].is_unique:
        raise ValueError("pb_annotations com variant_id repetido")
    juntos = membros[["variant_id", "study_id", "member_role", "present_abraom"]].merge(
        anotacoes[list(COLUNAS_DAS_ANOTACOES)], on="variant_id", how="left", suffixes=("", "_anotacao"),
        indicator=True)
    saida: dict[str, Any] = {}
    for (estudo, papel), grupo in juntos.groupby(["study_id", "member_role"], sort=True):
        anotado = grupo["_merge"] == "both"
        registros = grupo.loc[anotado].to_dict("records")
        definidos = sum(pontuar(r) is not None for r in registros)
        abraom = pd.to_numeric(grupo["abraom_af"], errors="coerce").to_numpy(dtype=float)
        da_membership = grupo["present_abraom"].astype(bool)
        da_anotacao = grupo["present_abraom_anotacao"]
        saida[f"{estudo}/{papel}"] = {
            "membros": int(len(grupo)), "com_anotacao": int(anotado.sum()),
            "gnomad_rarity_definido": int(definidos), "gnomad_status": _contagem(grupo["gnomad_status"]),
            "abraom_af_finito": int(np.isfinite(abraom).sum()), "abraom_status": _contagem(grupo["abraom_status"]),
            "present_abraom": {"na_membership": int(da_membership.sum()),
                               "nas_anotacoes": int(da_anotacao.fillna(False).astype(bool).sum()),
                               "discordantes": int((anotado & (da_membership != da_anotacao.fillna(False)
                                                               .astype(bool))).sum())}}
    return saida


def problemas_da_cobertura(por_grupo: dict[str, Any]) -> list[str]:
    """Membro sem linha de anotacao ou com presenca no ABraOM discordante: sinal de chave ou release errados, que
    reprova. Cobertura incompleta do `gnomad_rarity` NAO reprova: sai nas contagens, e o consumidor usa a intersecao
    de cobertura, como o Mosaic. Pura."""
    problemas = []
    for grupo, c in por_grupo.items():
        if c["com_anotacao"] != c["membros"]:
            problemas.append(f"{grupo}: {c['membros'] - c['com_anotacao']} membros sem linha em pb_annotations")
        if c["present_abraom"]["discordantes"]:
            problemas.append(f"{grupo}: {c['present_abraom']['discordantes']} membros com present_abraom discordante")
    return problemas


def carregar_mosaic(mosaic_root: Path) -> tuple[Callable[..., dict[str, Any]], dict[str, Any], Callable, dict]:
    """(logical_contract, especificacao do gnomad_rarity, comparator_score, identidade do codigo do Mosaic)."""
    sys.path.insert(0, str(mosaic_root / "src"))
    from mosaic.comparators import comparator_score, comparator_specs
    from mosaic.hashing import logical_contract

    especificacao = next((s for s in comparator_specs(config_dir=mosaic_root / "config", official_only=True)
                          if s.get("id") == "gnomad_rarity"), None)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=mosaic_root, capture_output=True, text=True,
                                timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        commit = f"nao conferido ({type(exc).__name__})"
    codigo = {"mosaic_root": str(mosaic_root), "commit": commit,
              "comparators_yaml_sha256": sha256_do_arquivo(mosaic_root / "config" / "comparators.yaml")}
    return logical_contract, especificacao, comparator_score, codigo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--mosaic-root", required=True, type=Path)
    parser.add_argument("--campanha", type=Path, default=RAIZ / "configs" / "campanha_r03_desenvolvimento.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    raiz, mosaic_root, destino = (args.release_root.expanduser(), args.mosaic_root.expanduser(),
                                  args.out.expanduser())
    if destino.exists():
        print(f"FALHOU: {destino} ja existe")
        return 2
    referencia = carregar_campanha(args.campanha)["g6"]["proveniencia"]["release_do_mosaic"]
    logical_contract, especificacao, comparator_score, codigo = carregar_mosaic(mosaic_root)
    nome_do_manifesto, manifesto = manifesto_da_copia(raiz)
    publicados = {c.get("path"): c for c in manifesto.get("artifact_contracts") or []}
    problemas, contratos = [], {}
    for caminho, chave in CHAVES.items():
        recalculado = logical_contract(pq.read_table(raiz / caminho), chave)
        problemas += comparar_contrato(referencia["logical_hash"].get(caminho), recalculado, caminho,
                                       campos=("logical_hash", "n"), origem="referencia declarada")
        if nome_do_manifesto:
            problemas += comparar_contrato(publicados.get(caminho), recalculado, caminho, origem=nome_do_manifesto)
        contratos[caminho] = {"logical_hash": recalculado["logical_hash"], "n": recalculado["n"],
                              "file_sha256": sha256_do_arquivo(raiz / caminho)}
    if nome_do_manifesto and manifesto.get("artifact_contracts_hash") not in (None,
                                                                              referencia["artifact_contracts_hash"]):
        problemas.append(f"{nome_do_manifesto}: artifact_contracts_hash != o declarado")
    problemas += conferir_especificacao(especificacao)
    anotacoes = pd.read_parquet(raiz / ANOTACOES, columns=list(COLUNAS_DAS_ANOTACOES))
    membros = pd.read_parquet(raiz / MEMBERSHIP, columns=["variant_id", "study_id", "member_role", "present_abraom"])
    por_grupo = cobertura(membros, anotacoes, lambda linha: comparator_score(linha, especificacao or GNOMAD_RARITY))
    problemas += problemas_da_cobertura(por_grupo)

    saida = {"formato": "campanha_r03_cobertura_das_baselines_v1", "passou": not problemas, "problemas": problemas,
             "release": {"referencia": referencia, "manifesto_da_copia": nome_do_manifesto,
                         "manifesto_sha256": sha256_do_arquivo(raiz / nome_do_manifesto) if nome_do_manifesto else None,
                         "artifact_contracts_hash_do_manifesto": manifesto.get("artifact_contracts_hash"),
                         "contratos": contratos},
             "comparador": {"especificacao": especificacao, "codigo": codigo}, "cobertura": por_grupo,
             "leitura": "so contagens: nenhum rotulo cruzado com anotacao ou score, nenhuma metrica; a baseline nos "
                        "estudos so se mede no G7"}
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[baselines] release {referencia['release_id']} | manifesto da copia: {nome_do_manifesto or 'nenhum'} | "
          f"Mosaic {codigo['commit'][:12]} | comparators.yaml {codigo['comparators_yaml_sha256'][:12]}")
    for caminho, c in contratos.items():
        print(f"  {caminho:<36} hash logico {c['logical_hash'][:12]}  n {c['n']}")
    print("  grupo                                  membros  anotados  gnomad_rarity  abraom_af  present (memb/anot/disc)")
    for grupo, c in por_grupo.items():
        p = c["present_abraom"]
        print(f"  {grupo:<38} {c['membros']:>7}  {c['com_anotacao']:>8}  {c['gnomad_rarity_definido']:>13}  "
              f"{c['abraom_af_finito']:>9}  {p['na_membership']}/{p['nas_anotacoes']}/{p['discordantes']}")
        print(f"    gnomad_status {c['gnomad_status']} | abraom_status {c['abraom_status']}")
    if problemas:
        print(f"\nFALHOU: {len(problemas)} problema(s)")
        for problema in problemas:
            print(f"  - {problema}")
        return 2
    print(f"\nPASSOU: anotacoes do release conferidas e cobrindo os membros. Detalhe em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
