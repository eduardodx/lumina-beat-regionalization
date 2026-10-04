#!/usr/bin/env python3
"""Fase 1, passo 2: leitura do R03 congelado (M0) no release novo, pelo MESMO caminho numerico dos caches antigos.

Dois modos, com a mesma regra (docs/fase1_r03_congelado.md):

- ``conferencia``: reextrai uma amostra deterministica de variantes que os caches antigos ja tem e compara os
  vetores, extracao por extracao, contra a tolerancia do extrator. E o que autoriza reaproveitar esses caches:
  achar o id no cache nao valida o vetor.
- ``complemento``: extrai as variantes do release elegiveis em 4 kb que nenhum cache antigo cobre.

Antes de extrair, a identidade nova tem de ser a do cache de referencia em tudo menos a tabela: codigo, pacote
`lumina`, ambiente, R03, FASTA, janela, lote e fragmento. Outro caminho numerico e recusado; nesse caso a saida e
extrair tudo de novo, com identidade propria. A janela, o lote e o fragmento vem da identidade da referencia.

USO (notebook, GPU; primeiro a conferencia, depois o complemento)
    PYTHONPATH="$PWD" python3 scripts/extrair_mosaic_v1.py --modo conferencia --entrega ~/mosaic-v1-2026-09-30 \\
        --referencia ~/artifacts/redesenho/g3_cache/M0 \\
        --cache-antigo ~/artifacts/redesenho/g3_cache/M0 --cache-antigo ~/artifacts/redesenho/g7_cache/M0 \\
        --checkpoint ~/artifacts/r03/best_checkpoint.pt --fasta ~/hg38/hg38.fa \\
        --out-dir ~/artifacts/mosaic_v1/cache_conferencia

SAIDAS: o cache em --out-dir (fragmentos, tabela, identidade e manifesto, como na campanha), os hashes das fontes
em fontes_da_extracao.json e, na conferencia, conferencia.json com a maior diferenca por extracao e cache antigo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import indices_existentes, ler_fragmentos, nome_do_fragmento, sha256_do_arquivo  # noqa: E402
from eval.campanha.recortes import COLUNAS, hash_da_tabela, hash_do_conteudo  # noqa: E402

RELEASE = "artifacts/mosaic-v1-2026-09-30"
VISTA = "views/4kb/partitions.parquet"
MODOS = ("conferencia", "complemento")
PAPEL = {"conferencia": "conferencia", "complemento": "mosaic_v1_complemento"}
#: Campos da identidade que so dependem da tabela extraida (e o commit, que nao determina numero).
CAMPOS_DA_TABELA = ("tabela_sha256_composicao", "tabela_sha256_conteudo", "papeis", "revisao_do_codigo")
SEMENTE_DA_AMOSTRA = "conferencia_mosaic_v1"


def tabela_do_release(raiz: Path) -> pd.DataFrame:
    """Variantes elegiveis na vista de 4 kb, com as colunas que a extracao e o hash de conteudo leem."""
    exemplos = pd.read_parquet(raiz / "clinical-variants.parquet",
                               columns=["variant_id", "chrom", "pos_1based", "ref", "alt", "binary_label", "label_tier"])
    paineis = pd.read_parquet(raiz / "evaluation-panels.parquet", columns=["variant_id", "primary_panel"])
    vista = pd.read_parquet(raiz / VISTA, columns=["variant_id", "sequence_eligible", "overlap_cluster_id"])
    tabela = (exemplos.merge(paineis, on="variant_id", validate="one_to_one")
              .merge(vista, on="variant_id", validate="one_to_one"))
    if len(tabela) != len(exemplos):
        raise ValueError(f"o release perdeu linhas no join: {len(tabela)} de {len(exemplos)}")
    tabela = tabela[tabela["sequence_eligible"].astype(bool)].drop(columns="sequence_eligible")
    return tabela[list(COLUNAS)].reset_index(drop=True)


def ids_do_cache(pasta: Path) -> set[str]:
    ids: set[str] = set()
    for indice in indices_existentes(pasta):
        with open(pasta / nome_do_fragmento(indice), "rb") as arquivo, np.load(arquivo, allow_pickle=False) as dados:
            ids.update(dados["variant_id"].astype(str).tolist())
    return ids


def validar_cache_antigo(pasta: Path, referencia: dict[str, Any], release: pd.DataFrame) -> tuple[set[str], dict]:
    """Confere CADA fonte antes de descontar seus IDs do complemento; nao modifica o cache antigo."""
    for nome in ("identidade.json", "manifesto.json", "tabela.parquet"):
        if not (pasta / nome).is_file():
            raise ValueError(f"{pasta}: falta {nome}")
    ident = json.loads((pasta / "identidade.json").read_text(encoding="utf-8"))
    diferentes = sorted(k for k in set(referencia) | set(ident)
                        if k not in CAMPOS_DA_TABELA and referencia.get(k) != ident.get(k))
    if diferentes:
        raise ValueError(f"{pasta}: identidade difere da referencia em {diferentes}")
    if ident.get("sistema") != "M0" or ident.get("adapter_sha256") is not None:
        raise ValueError(f"{pasta}: o cache tem de ser M0, sem adapter")
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    if manifesto.get("completo") is not True:
        raise ValueError(f"{pasta}: manifesto nao declara cache completo")
    declarada = manifesto.get("identidade", {})
    if any(ident.get(k) != declarada.get(k) for k in (set(ident) | set(declarada)) - {"revisao_do_codigo"}):
        raise ValueError(f"{pasta}: identidade do manifesto diverge de identidade.json")
    tabela = pd.read_parquet(pasta / "tabela.parquet")
    if tabela["variant_id"].duplicated().any():
        raise ValueError(f"{pasta}: tabela com variant_id repetido")
    if (hash_do_conteudo(tabela) != ident.get("tabela_sha256_conteudo")
            or hash_da_tabela(tabela) != ident.get("tabela_sha256_composicao")):
        raise ValueError(f"{pasta}: hashes da tabela nao conferem")
    ids, problemas = ler_fragmentos(pasta, tabela)
    if problemas:
        raise ValueError(f"{pasta}: fragmentos invalidos: {'; '.join(problemas[:5])}")
    if ids != set(tabela["variant_id"].astype(str)):
        raise ValueError(f"{pasta}: fragmentos nao cobrem a tabela inteira")
    if (manifesto.get("variantes_na_tabela") != len(tabela)
            or manifesto.get("variantes_no_cache") != len(ids) or manifesto.get("faltando") != 0):
        raise ValueError(f"{pasta}: contagens do manifesto nao conferem")

    # Rotulo, painel, cluster e papel pertencem ao release atual. Para reutilizar o vetor,
    # o mesmo ID precisa continuar identificando a MESMA sequencia e o mesmo alelo.
    coordenadas = ["chrom", "pos_1based", "ref", "alt"]
    antiga = tabela.assign(variant_id=tabela["variant_id"].astype(str)).set_index("variant_id")
    atual = release.assign(variant_id=release["variant_id"].astype(str)).set_index("variant_id")
    comuns = sorted(ids & set(atual.index))
    a, b = antiga.loc[comuns, coordenadas], atual.loc[comuns, coordenadas]
    divergentes = a.ne(b).any(axis=1) | a.isna().any(axis=1) | b.isna().any(axis=1)
    if divergentes.any():
        raise ValueError(f"{pasta}: coordenadas/alelos divergem do release para {list(a.index[divergentes])[:5]}")
    arquivos = ["identidade.json", "manifesto.json", "tabela.parquet"]
    arquivos += [nome_do_fragmento(i) for i in indices_existentes(pasta)]
    return ids, {"n": len(ids), "no_release_elegivel": len(comuns),
                 "sha256": {nome: sha256_do_arquivo(pasta / nome) for nome in arquivos}}


def _menores_hashes(ids: list[str], n: int) -> list[str]:
    chave = {v: hashlib.sha256(f"{SEMENTE_DA_AMOSTRA}|{v}".encode()).hexdigest() for v in ids}
    return sorted(ids, key=lambda v: chave[v])[:n]


def selecionar(tabela: pd.DataFrame, modo: str, caches: dict[str, set[str]], n_amostra: int) -> pd.DataFrame:
    """A tabela a extrair: o complemento dos caches, ou uma amostra deterministica de cada cache."""
    elegiveis = set(tabela["variant_id"].astype(str))
    if modo == "complemento":
        cobertos = set().union(*caches.values()) if caches else set()
        escolhidos = elegiveis - cobertos
    else:
        por_cache = max(1, n_amostra // max(1, len(caches)))
        escolhidos = set()
        for ids in caches.values():
            escolhidos |= set(_menores_hashes(sorted(ids & elegiveis), por_cache))
    saida = tabela[tabela["variant_id"].astype(str).isin(escolhidos)].assign(papel=PAPEL[modo])
    return saida.sort_values(["papel", "variant_id"], kind="mergesort").reset_index(drop=True)


def vetores(pasta: Path, ids: set[str], extracoes: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """{extracao: {variant_id: vetor}} das variantes pedidas, lidas dos fragmentos do cache."""
    saida: dict[str, dict[str, np.ndarray]] = {nome: {} for nome in extracoes}
    for indice in indices_existentes(pasta):
        with open(pasta / nome_do_fragmento(indice), "rb") as arquivo, np.load(arquivo, allow_pickle=False) as dados:
            vids = dados["variant_id"].astype(str)
            linhas = [i for i, v in enumerate(vids) if v in ids]
            if not linhas:
                continue
            for nome in extracoes:
                matriz = np.asarray(dados[nome], dtype=np.float32)
                for i in linhas:
                    saida[nome][vids[i]] = matriz[i]
    return saida


def comparar(novos: dict[str, dict[str, np.ndarray]], antigos: dict[str, dict[str, np.ndarray]],
             tolerancia: float) -> dict[str, Any]:
    """Maior diferenca absoluta por extracao entre o vetor reextraido e o do cache antigo, variante a variante."""
    saida: dict[str, Any] = {"tolerancia": tolerancia, "extracoes": {}}
    passou = bool(novos) and set(novos) == set(antigos) and bool(np.isfinite(tolerancia) and tolerancia >= 0)
    for nome, por_id in novos.items():
        comuns = sorted(set(por_id) & set(antigos.get(nome, {})))
        faltando = sorted(set(por_id) - set(antigos.get(nome, {})))
        faltando_no_novo = sorted(set(antigos.get(nome, {})) - set(por_id))
        if not comuns:
            saida["extracoes"][nome] = {"comparadas": 0, "faltando_no_antigo": len(faltando)}
            passou = False
            continue
        diferencas, invalidos = [], []
        for v in comuns:
            a, b = np.asarray(por_id[v]), np.asarray(antigos[nome][v])
            if a.ndim != 1 or a.size == 0 or a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
                invalidos.append(v)
            else:
                diferencas.append(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        diferencas = np.asarray(diferencas)
        bloco = {"comparadas": len(comuns), "faltando_no_antigo": len(faltando),
                 "faltando_no_novo": len(faltando_no_novo), "vetores_invalidos": len(invalidos),
                 "max_abs": float(diferencas.max()) if diferencas.size else None,
                 "mediana_abs": float(np.median(diferencas)) if diferencas.size else None,
                 "acima_da_tolerancia": int((diferencas > tolerancia).sum())}
        saida["extracoes"][nome] = bloco
        passou &= bloco["acima_da_tolerancia"] == 0 and not faltando and not faltando_no_novo and not invalidos
    saida["passou"] = bool(passou)
    return saida


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modo", required=True, choices=MODOS)
    parser.add_argument("--entrega", required=True, type=Path, help="checkout dedicado do Mosaic com a entrega baixada")
    parser.add_argument("--referencia", required=True, type=Path, help="cache antigo cuja identidade fixa o caminho numerico")
    parser.add_argument("--cache-antigo", action="append", default=[], type=Path)
    parser.add_argument("--n-amostra", type=int, default=512)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--campanha", type=Path, default=RAIZ / "configs" / "campanha_r03_desenvolvimento.json")
    parser.add_argument("--superficie", type=Path, default=RAIZ / "configs" / "adapter_r03_superficie.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.cache_antigo or args.n_amostra <= 0:
        return _falhar(["declare pelo menos um --cache-antigo e --n-amostra positivo"])

    referencia = args.referencia.expanduser()
    identidade_ref = json.loads((referencia / "identidade.json").read_text(encoding="utf-8"))
    if identidade_ref.get("sistema") != "M0" or identidade_ref.get("adapter_sha256") is not None:
        return _falhar(["a referencia tem de ser um cache do M0 (R03 congelado, sem adapter)"])
    try:
        release = tabela_do_release(args.entrega.expanduser() / RELEASE)
        caches, fontes = {}, {}
        for p in args.cache_antigo:
            pasta = p.expanduser()
            ids, fonte = validar_cache_antigo(pasta, identidade_ref, release)
            if not (ids & set(release["variant_id"].astype(str))):
                raise ValueError(f"{pasta}: nenhum ID elegivel em comum com o release para conferir")
            caches[str(pasta)], fontes[str(pasta)] = ids, fonte
            print(f"[cache antigo] {pasta}: {len(ids):,} variantes, identidade e fragmentos conferidos")
    except (ValueError, KeyError, OSError) as exc:
        return _falhar([str(exc)])
    tabela = selecionar(release, args.modo, caches, args.n_amostra)
    print(f"[mosaic_v1] modo {args.modo}: {len(tabela):,} variantes a extrair "
          f"(caches antigos: {', '.join(f'{len(v):,}' for v in caches.values())})")
    if tabela.empty:
        return _falhar(["nada a extrair"])

    checkpoint_sha = sha256_do_arquivo(args.checkpoint.expanduser())
    if checkpoint_sha != identidade_ref["checkpoint_sha256"]:
        return _falhar([f"checkpoint {checkpoint_sha[:12]} nao e o da referencia"])
    parametros = argparse.Namespace(
        sistema="M0", semente_do_adapter=None, adapter=None, checkpoint=args.checkpoint.expanduser(),
        superficie=args.superficie, window_bp=int(identidade_ref["janela_bp"]),
        variantes_por_lote=int(identidade_ref["lote"]["variantes_por_lote"]),
        fragmento=int(identidade_ref["fragmento"]), out_dir=args.out_dir.expanduser())

    import torch

    from eval.campanha.recortes import carregar_campanha
    from scripts.audit_variant_windows import abrir_fasta
    from scripts.extract_campaign_features import (
        TOLERANCIA_NUMERICA,
        ambiente_de_execucao,
        codigo_da_extracao,
        identidade,
        montar_sistema,
        revisao_do_codigo,
        rodar_extracao,
    )

    device = torch.device(args.device)
    modelo, _ = montar_sistema(parametros, device, carregar_campanha(args.campanha))
    identidade_nova = identidade(parametros, tabela=tabela, checkpoint_sha=checkpoint_sha, adapter_sha=None,
                                 fasta_sha=sha256_do_arquivo(args.fasta.expanduser()), revisao=revisao_do_codigo(),
                                 codigo=codigo_da_extracao(), ambiente=ambiente_de_execucao(device))
    diferentes = sorted(k for k in set(identidade_ref) | set(identidade_nova)
                        if k not in CAMPOS_DA_TABELA and identidade_ref.get(k) != identidade_nova.get(k))
    if diferentes:
        return _falhar([f"a identidade difere da referencia em {diferentes}: outro caminho numerico. Nada foi "
                        f"extraido; reaproveitar os caches antigos deixa de valer (extrair tudo de novo)"])
    fetch, _leitor = abrir_fasta(args.fasta.expanduser())
    codigo = rodar_extracao(parametros, modelo, tabela, fetch, identidade_nova)
    if codigo == 0:
        proveniencia = {"modo": args.modo, "caches_antigos": fontes,
                       "referencia": str(referencia), "script_sha256": sha256_do_arquivo(Path(__file__)),
                       "release_sha256": {nome: sha256_do_arquivo(args.entrega.expanduser() / RELEASE / nome)
                                          for nome in ("clinical-variants.parquet", "evaluation-panels.parquet", VISTA)}}
        (parametros.out_dir / "fontes_da_extracao.json").write_text(json.dumps(proveniencia, indent=2), encoding="utf-8")
    if codigo != 0 or args.modo == "complemento":
        return codigo

    extracoes = sorted(identidade_nova["extracoes"])
    ids = set(tabela["variant_id"].astype(str))
    novos = vetores(parametros.out_dir, ids, extracoes)
    por_cache = {}
    for pasta, cobertos in caches.items():
        comuns = ids & cobertos
        recorte = {nome: {v: vetor for v, vetor in por_id.items() if v in comuns}
                   for nome, por_id in novos.items()}
        por_cache[pasta] = comparar(recorte, vetores(Path(pasta), comuns, extracoes), TOLERANCIA_NUMERICA)
    resultado = {"tolerancia": TOLERANCIA_NUMERICA, "por_cache": por_cache,
                 "passou": all(r["passou"] for r in por_cache.values())}
    (parametros.out_dir / "conferencia.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(json.dumps(resultado, indent=2))
    if not resultado["passou"]:
        return _falhar(["os vetores reextraidos diferem dos caches antigos acima da tolerancia: nao reaproveitar"])
    print("PASSOU (conferencia): os vetores dos caches antigos podem ser reaproveitados")
    return 0


def _falhar(problemas: list[str]) -> int:
    print(f"FALHOU: {len(problemas)} problema(s)")
    for problema in problemas:
        print(f"  - {problema}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
