#!/usr/bin/env python3
"""Acha no S3 o arquivo do ABraOM que o source-lock do Mosaic exige, conferindo por sha256.

Roda no notebook (aws cli + stdlib + pyyaml; sem GPU). Le o S3 e escreve so em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 1 (identidades) e gate G0.

POR QUE EXISTE
--------------
O estudo brasileiro do Mosaic exige a fonte `abraom_sabe1171` do source-lock -- nao "um arquivo do ABraOM". Varios
artefatos antigos da campanha v10 parecem ABraOM e NAO sao o mesmo objeto: o dataset do adapter de frequencia, o
master regional, os adapters A_BR/A_gnomAD. Usar um deles quebraria a identidade declarada no manifesto sem dar
erro nenhum.

A unica pergunta valida e: qual arquivo tem o sha256 que o `config/sources.yaml` do Mosaic declara? Este script
le esse valor do PROPRIO Mosaic (nao aceita hash digitado), lista candidatos nos prefixos indicados, baixa e
compara.

Arquivo comprimido: o hash declarado pode ser do conteudo descomprimido. Para `.gz` o script confere os dois.

O QUE NAO PROVA
---------------
- Nao valida o conteudo do ABraOM, so a identidade. Bater o sha256 e prova suficiente; nao bater nao prova que o
  arquivo e inutil -- prova que ele nao e o objeto declarado, e ai a decisao e do Eduardo.
- Nao baixa nada sem `--download`.

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    # 1) so listar candidatos:
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/locate_abraom_source.py --mosaic-root ~/testeArq/lumina-mosaic
    # 2) baixar e conferir:
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/locate_abraom_source.py --mosaic-root ~/testeArq/lumina-mosaic \\
        --download --out-dir ~/abraom_candidatos
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

SOURCE_KEY = "abraom_sabe1171"

#: Prefixos onde o arquivo pode estar. O primeiro e a aposta principal: os brutos do ClinVar do Mosaic ficam em
#: `.../mosaic/data/raw/clinvar/`, entao o bruto do ABraOM deve seguir a mesma simetria.
PREFIXOS_PADRAO = (
    "s3://ai4bio-lumina/benchmarks/mosaic/data/raw/",
    "s3://ai4bio-lumina/benchmarks/mosaic/data/processed/gen-abraom-seqs/",
    "s3://ai4bio-lumina/data/external/",
    "s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/abraom_frequency_adapter/",
)

#: Artefatos derivados que PARECEM ABraOM e nao sao a fonte declarada. Listados para o relatorio avisar.
DERIVADOS_CONHECIDOS = {
    "clinvar_regional_abraom_master.parquet": "master regional v1 (derivado, join so-SNV)",
    "abraom_frequency_adapter": "dataset do adapter de frequencia da campanha v10",
}

PISTAS = ("abraom", "sabe")
EXTENSOES = (".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz", ".vcf", ".vcf.gz")


def expected_source(mosaic_root: Path, key: str = SOURCE_KEY) -> dict[str, Any]:
    """Le o sha256 e o caminho declarados no `config/sources.yaml` do Mosaic. Fonte unica da verdade."""
    import yaml

    caminho = mosaic_root.expanduser() / "config" / "sources.yaml"
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    fontes = dados.get("sources") or {}
    if key not in fontes:
        raise SystemExit(f"{caminho}: fonte {key!r} ausente; chaves: {sorted(fontes)[:10]}")
    spec = dict(fontes[key])
    if not spec.get("sha256") or spec["sha256"] == "pending":
        raise SystemExit(f"{caminho}: fonte {key!r} sem sha256 fixado; nada a conferir")
    spec["arquivo_de_origem"] = str(caminho)
    return spec


def parse_ls(saida: str, prefixo: str) -> list[tuple[str, int]]:
    """Converte a saida de `aws s3 ls --recursive` em (uri, bytes)."""
    bucket_raiz = prefixo.split("/", 3)
    bucket = bucket_raiz[2]
    itens: list[tuple[str, int]] = []
    for linha in saida.splitlines():
        partes = linha.split()
        if len(partes) < 4:
            continue
        try:
            tamanho = int(partes[2])
        except ValueError:
            continue
        chave = " ".join(partes[3:])
        itens.append((f"s3://{bucket}/{chave}", tamanho))
    return itens


def parece_abraom(uri: str) -> bool:
    nome = uri.rsplit("/", 1)[-1].lower()
    if not nome:
        return False
    tem_pista = any(pista in uri.lower() for pista in PISTAS)
    tem_extensao = nome.endswith(EXTENSOES)
    return tem_pista and tem_extensao


def e_derivado_conhecido(uri: str) -> str | None:
    for marca, explicacao in DERIVADOS_CONHECIDOS.items():
        if marca in uri:
            return explicacao
    return None


def hashes_do_arquivo(caminho: Path) -> dict[str, str]:
    """sha256 do arquivo e, se for `.gz`, tambem do conteudo descomprimido."""
    out = {"arquivo": sha256_file(caminho)}
    if caminho.suffix == ".gz":
        digest = hashlib.sha256()
        with gzip.open(caminho, "rb") as handle:
            for bloco in iter(lambda: handle.read(1 << 20), b""):
                digest.update(bloco)
        out["descomprimido"] = digest.hexdigest()
    return out


def confere(caminho: Path, esperado: str) -> tuple[bool, dict[str, str]]:
    calculados = hashes_do_arquivo(caminho)
    return esperado in calculados.values(), calculados


def listar(prefixo: str) -> list[tuple[str, int]]:
    try:
        saida = subprocess.run(["aws", "s3", "ls", "--recursive", prefixo],
                               check=True, capture_output=True, text=True).stdout
    except FileNotFoundError as exc:
        raise SystemExit(f"aws cli nao encontrado: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        return []
    return parse_ls(saida, prefixo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mosaic-root", required=True, type=Path)
    parser.add_argument("--source-key", default=SOURCE_KEY)
    parser.add_argument("--prefix", action="append", default=None,
                        help="pode repetir; sem isto usa a lista padrao")
    parser.add_argument("--download", action="store_true", help="baixa os candidatos e confere o sha256")
    parser.add_argument("--max-mb", type=int, default=4000, help="nao baixa candidato maior que isto")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    spec = expected_source(args.mosaic_root, args.source_key)
    esperado = spec["sha256"]
    print(f"[abraom] fonte {args.source_key}: path declarado {spec.get('path')!r}, "
          f"sha256 {esperado[:16]}…, schema {spec.get('schema')}")
    print(f"[abraom] valor lido de {spec['arquivo_de_origem']}")

    prefixos = args.prefix or list(PREFIXOS_PADRAO)
    candidatos: list[tuple[str, int]] = []
    for prefixo in prefixos:
        itens = listar(prefixo)
        achados = [item for item in itens if parece_abraom(item[0])]
        print(f"[abraom] {prefixo}: {len(itens)} objetos, {len(achados)} candidatos")
        candidatos.extend(achados)

    relatorio: dict[str, Any] = {
        "fonte_esperada": {k: spec[k] for k in ("path", "sha256", "schema") if k in spec},
        "prefixos": prefixos,
        "candidatos": [{"uri": uri, "bytes": tamanho,
                        "derivado_conhecido": e_derivado_conhecido(uri)} for uri, tamanho in candidatos],
        "conferidos": [],
        "encontrado": None,
    }

    if not candidatos:
        print("\nNenhum candidato. Tente outros prefixos com --prefix, ou peca o caminho ao Eduardo.")
    for uri, tamanho in candidatos:
        marca = e_derivado_conhecido(uri)
        print(f"  {tamanho / 1e6:10.1f} MB  {uri}" + (f"   [{marca}]" if marca else ""))

    if args.download and candidatos:
        destino = (args.out_dir or Path("./abraom_candidatos")).expanduser()
        destino.mkdir(parents=True, exist_ok=True)
        for uri, tamanho in candidatos:
            if tamanho > args.max_mb * 1e6:
                print(f"[abraom] pulando {uri} ({tamanho / 1e6:.0f} MB > --max-mb)")
                continue
            local = destino / uri.rsplit("/", 1)[-1]
            if not local.exists():
                print(f"[abraom] baixando {uri}")
                subprocess.run(["aws", "s3", "cp", uri, str(local)], check=True)
            bate, calculados = confere(local, esperado)
            relatorio["conferidos"].append({"uri": uri, "local": str(local), "hashes": calculados, "bate": bate})
            print(f"[abraom] {'BATE' if bate else 'nao bate'}  {uri}  {calculados}")
            if bate:
                relatorio["encontrado"] = uri
                break

    if args.out_dir is not None:
        args.out_dir.expanduser().mkdir(parents=True, exist_ok=True)
        (args.out_dir.expanduser() / "localizacao_abraom.json").write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    if relatorio["encontrado"]:
        print(f"\nENCONTRADO: {relatorio['encontrado']}\nRegistrar como a fonte `{args.source_key}` no G0.")
        return 0
    if args.download:
        print("\nNenhum candidato bateu o sha256 declarado. Isso NAO significa que o dado e inutil: significa "
              "que nao e o objeto do source-lock. Levar ao Eduardo antes de usar qualquer substituto.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
