#!/usr/bin/env python3
"""Depois de reconstruir o ambiente, confere que ele REPRODUZ um cache ja gravado: re-extrai as primeiras N
variantes do primeiro fragmento, nos MESMOS lotes da extracao original, e compara numero a numero.

POR QUE
    Um reinicio do espaco do SageMaker apaga o que foi instalado fora da home (24/09: o `mamba_ssm` sumiu do
    `/opt/conda`). `conferir_codigo_do_cache.py` compara versoes e arquivos; isto compara o NUMERO -- e e o que
    decide se retomar o MR_a2 (12 fragmentos gravados antes do reinicio) e parear com o M0 continuam validos.

COMO
    - le a identidade do cache (sistema, janela, variantes por lote, adapter) e recusa argumentos que a contradigam;
    - exige que as N primeiras linhas da tabela sejam as N primeiras do `fragmento_00000` (execucao nova) e que nao
      haja `falhas.json`: so assim os lotes re-extraidos tem a MESMA vizinhanca dos originais -- a vizinhanca no lote
      muda o numero na casa de 1e-6;
    - monta o sistema com as funcoes do proprio extrator (nenhum dos 12 arquivos da identidade muda), re-extrai e
      compara cada leitura: diferenca maxima, igualdade exata e a tolerancia declarada do extrator (1e-5).
Nada e gravado no cache.

USO (notebook)
    PYTHONPATH="$WORK" python3 scripts/conferir_reproducao_do_cache.py --cache ~/artifacts/redesenho/g3_cache/M0 \\
        --checkpoint ~/artifacts/r03/best_checkpoint.pt --fasta ~/hg38/hg38.fa
    MR: mais --adapter ~/artifacts/redesenho/g4_a2/adapter_melhor.pt --semente-do-adapter 20260922
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import nome_do_fragmento  # noqa: E402
from eval.campanha.layout import EXTRACOES  # noqa: E402
from eval.campanha.recortes import carregar_campanha  # noqa: E402
from scripts.extract_campaign_features import (  # noqa: E402
    TOLERANCIA_NUMERICA,
    extrair_lote,
    janelas_da_tabela,
    montar_sistema,
)


def problemas_do_alinhamento(ids_da_tabela: np.ndarray, ids_do_fragmento: np.ndarray, n: int,
                             lote: int) -> list[str]:
    """O que impede a comparacao EXATA: N tem de formar lotes completos, e as N primeiras do fragmento tem de ser as
    N primeiras da tabela, na mesma ordem (o que so vale numa execucao nova e sem falha de janela). Pura."""
    problemas = []
    if n <= 0 or n % lote:
        problemas.append(f"--variantes {n} tem de ser multiplo de {lote} (lotes completos, como na extracao)")
    if len(ids_do_fragmento) < n or len(ids_da_tabela) < n:
        problemas.append(f"o fragmento ({len(ids_do_fragmento)}) ou a tabela ({len(ids_da_tabela)}) tem menos de {n}")
    elif list(map(str, ids_do_fragmento[:n])) != list(map(str, ids_da_tabela[:n])):
        problemas.append("as primeiras variantes do fragmento_00000 nao sao as primeiras da tabela: os lotes "
                         "re-extraidos nao teriam a vizinhanca original")
    return problemas


def comparar_leituras(nova: dict[str, np.ndarray], gravada: dict[str, np.ndarray],
                      tolerancia: float) -> dict[str, Any]:
    """Diferenca maxima e igualdade exata por leitura; passa se toda leitura ficar dentro da tolerancia. Pura."""
    por_leitura = {}
    for nome in sorted(gravada):
        a, b = np.asarray(nova.get(nome)), np.asarray(gravada[nome])
        if a.shape != b.shape:
            por_leitura[nome] = {"forma": f"{a.shape} != {b.shape}", "passou": False}
            continue
        diferenca = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        por_leitura[nome] = {"diferenca_maxima": diferenca, "identica": bool(np.array_equal(a, b)),
                             "passou": diferenca <= tolerancia}
    return {"por_leitura": por_leitura, "tolerancia": tolerancia,
            "passou": bool(por_leitura) and all(v["passou"] for v in por_leitura.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, help="adapter_melhor.pt da semente (so MR)")
    parser.add_argument("--semente-do-adapter", type=int, help="so MR")
    parser.add_argument("--variantes", type=int, default=64)
    parser.add_argument("--superficie", type=Path, default=Path("configs/adapter_r03_superficie.json"))
    parser.add_argument("--campanha", type=Path, default=Path("configs/campanha_r03_desenvolvimento.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    cache = args.cache.expanduser()
    identidade = json.loads((cache / "identidade.json").read_text(encoding="utf-8"))
    sistema, janela = identidade["sistema"], int(identidade["janela_bp"])
    lote, foco = int(identidade["lote"]["variantes_por_lote"]), int(identidade["indice_focal"])
    problemas = []
    if sistema == "MR" and (args.adapter is None or args.semente_do_adapter != identidade.get("semente_do_adapter")):
        problemas.append(f"cache MR da semente {identidade.get('semente_do_adapter')}: passe --adapter e a MESMA "
                         f"--semente-do-adapter")
    if sistema == "M0" and (args.adapter is not None or args.semente_do_adapter is not None):
        problemas.append("cache M0: sem --adapter nem --semente-do-adapter")
    if (cache / "falhas.json").exists():
        problemas.append("o cache tem falhas.json: os lotes originais podem nao corresponder as linhas da tabela")
    tabela = pd.read_parquet(cache / "tabela.parquet")
    with np.load(cache / nome_do_fragmento(0), allow_pickle=False) as dados:
        fragmento = {chave: dados[chave] for chave in dados.files}
    problemas += problemas_do_alinhamento(tabela["variant_id"].to_numpy(), fragmento["variant_id"], args.variantes,
                                          lote)
    if problemas:
        print("FALHOU: a reproducao nao pode ser conferida:")
        for problema in problemas:
            print(f"  - {problema}")
        return 2

    import torch

    from scripts.audit_variant_windows import abrir_fasta

    configuracao = argparse.Namespace(sistema=sistema, checkpoint=args.checkpoint.expanduser(),
                                      adapter=args.adapter.expanduser() if args.adapter else None,
                                      semente_do_adapter=args.semente_do_adapter, superficie=args.superficie)
    modelo, adapter_sha = montar_sistema(configuracao, torch.device(args.device), carregar_campanha(args.campanha))
    if sistema == "MR" and adapter_sha != identidade.get("adapter_sha256"):
        print(f"FALHOU: o adapter passado ({str(adapter_sha)[:12]}) nao e o do cache "
              f"({str(identidade.get('adapter_sha256'))[:12]})")
        return 2
    fetch, _leitor = abrir_fasta(args.fasta.expanduser())
    janelas, falhas = janelas_da_tabela(tabela.iloc[:args.variantes], fetch, janela)
    if falhas:
        print(f"FALHOU: {len(falhas)} janelas nao se construiram agora: {falhas[:3]}")
        return 2
    blocos: dict[str, list[np.ndarray]] = {}
    for inicio in range(0, args.variantes, lote):
        grupo = [j for _, j in janelas[inicio:inicio + lote]]
        for nome, matriz in extrair_lote(modelo, grupo, variantes_por_lote=lote, focal=foco).items():
            blocos.setdefault(nome, []).append(matriz)
    nova = {nome: np.concatenate(partes) for nome, partes in blocos.items()}
    resultado = comparar_leituras(nova, {nome: fragmento[nome][:args.variantes] for nome in EXTRACOES},
                                  TOLERANCIA_NUMERICA)

    print(f"[reproducao] {cache} ({sistema}{' ' + str(args.semente_do_adapter) if sistema == 'MR' else ''}): "
          f"{args.variantes} variantes, {args.variantes // lote} lotes de {lote}")
    for nome, valores in resultado["por_leitura"].items():
        if "forma" in valores:
            print(f"  {nome:<22} forma diferente: {valores['forma']}")
            continue
        print(f"  {nome:<22} diferenca maxima {valores['diferenca_maxima']:.3e}  "
              f"{'IDENTICA' if valores['identica'] else 'nao identica'}  "
              f"({'ok' if valores['passou'] else 'FORA da tolerancia'} {resultado['tolerancia']:.0e})")
    if not resultado["passou"]:
        print("FALHOU: o ambiente atual NAO reproduz este cache. Nao retome nem extraia com ele.")
        return 2
    print("PASSOU: o ambiente atual reproduz o cache gravado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
