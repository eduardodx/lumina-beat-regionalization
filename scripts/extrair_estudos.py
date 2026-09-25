#!/usr/bin/env python3
"""G7, passo 1: extrai as variantes dos dois estudos pelo MESMO caminho numerico do desenvolvimento. Exige o G6
CONGELADO.

O QUE FAZ
    1. le o manifesto congelado (sha256 a parte; rascunho e recusado) e confere que a declaracao ainda e a dele;
    2. tabela RECONSTRUIDA das tabelas oficiais do release -- membership + coordenadas e alelos do `pb_examples`,
       os dois com o hash logico do Mosaic igual a referencia congelada no manifesto (`g7.tabela_oficial`): uma
       linha por variante, papel `estudo`. Um arquivo intermediario com os mesmos ids e outra sequencia nao passa.
       `--membros` (a saida do G1), se dado, e conferido campo a campo contra ela. Membros no chr8 entram
       (`g7.LEITURA_DO_CHR8`);
    3. monta o sistema (M0, ou MR da semente) com as funcoes do extrator de desenvolvimento -- nenhum dos 12 arquivos
       da identidade muda --, com a janela, o lote e o fragmento do cache de desenvolvimento do MESMO sistema (lidos
       da identidade dele, cujo sha256 o manifesto registrou), e o adapter declarado congelado para a semente;
    4. ANTES de extrair, a identidade nova tem de ser a do cache de desenvolvimento em tudo menos a tabela (codigo,
       pacote `lumina`, ambiente, R03, adapter, FASTA, janela, lote): outro caminho numerico e recusado;
    5. extrai com o mesmo `rodar_extracao` (retomavel, gravacao atomica, completude estrita).

`--so-conferir` faz 1 a 3 sem GPU (sem montar o sistema) e para: use antes de gastar GPU.

USO (notebook, GPU; um sistema por vez)
    PYTHONPATH="$PWD" python3 scripts/extrair_estudos.py --manifesto ~/artifacts/redesenho/g6_final \\
        --sistema MR --semente-do-adapter 20260922 \\
        --release-root ~/mosaic-v1 --mosaic-root ~/testeArq/lumina-mosaic \\
        --membros ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \\
        --checkpoint ~/artifacts/r03/best_checkpoint.pt --fasta ~/hg38/hg38.fa \\
        --out-dir ~/artifacts/redesenho/g7_cache/MR_a2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import g6, g7  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.recortes import carregar_campanha, hash_do_conteudo  # noqa: E402
from scripts import conferir_cobertura_das_baselines as cobertura  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifesto", required=True, type=Path, help="pasta do G6 congelado")
    parser.add_argument("--sistema", required=True, choices=("M0", "MR"))
    parser.add_argument("--semente-do-adapter", type=int, help="so MR")
    parser.add_argument("--release-root", required=True, type=Path, help="o release do Mosaic (~/mosaic-v1)")
    parser.add_argument("--mosaic-root", required=True, type=Path, help="o repositorio do Mosaic (hash logico)")
    parser.add_argument("--membros", type=Path, help="brazil_study_variants.parquet do G1: conferido contra a oficial")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--campanha", type=Path, default=RAIZ / "configs" / "campanha_r03_desenvolvimento.json")
    parser.add_argument("--superficie", type=Path, default=RAIZ / "configs" / "adapter_r03_superficie.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--so-conferir", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if (args.sistema == "MR") != (args.semente_do_adapter is not None):
        print("FALHOU: MR exige --semente-do-adapter; M0 nao aceita")
        return 2
    try:
        manifesto, sha_do_manifesto = g6.ler_manifesto_congelado(args.manifesto)
    except g6.ManifestoInvalido as exc:
        print(f"FALHOU: {exc}")
        return 2
    problemas = []
    if sha256_do_arquivo(args.campanha.expanduser()) != manifesto["declaracao"]["sha256"]:
        problemas.append("a declaracao mudou depois do congelamento do G6")
    chave = "M0" if args.sistema == "M0" else str(args.semente_do_adapter)
    desenvolvimento = manifesto["caches_de_desenvolvimento"].get(chave)
    if desenvolvimento is None:
        return _falhar([f"o manifesto nao tem cache de desenvolvimento para {chave}"])
    arquivo_da_identidade = Path(desenvolvimento["pasta"]).expanduser() / "identidade.json"
    if not arquivo_da_identidade.exists() or sha256_do_arquivo(arquivo_da_identidade) != desenvolvimento[
            "identidade_sha256"]:
        problemas.append(f"{arquivo_da_identidade} ausente ou diferente do registrado no manifesto")
    if problemas:
        return _falhar(problemas)
    identidade_dev = json.loads(arquivo_da_identidade.read_text(encoding="utf-8"))
    faltando = [c for c in ("janela_bp", "lote", "fragmento", "checkpoint_sha256") if c not in identidade_dev]
    if faltando:
        return _falhar([f"a identidade de desenvolvimento nao tem {faltando}: nao ha parametros a reproduzir"])
    release = args.release_root.expanduser()
    problemas = cobertura.conferir_contratos(
        release, cobertura.carregar_hash_logico(args.mosaic_root.expanduser()),
        manifesto["proveniencia"]["declarada"]["release_do_mosaic"], (cobertura.MEMBERSHIP, cobertura.EXEMPLOS))
    if problemas:
        return _falhar(problemas)
    tabela = g7.tabela_oficial(pd.read_parquet(release / cobertura.MEMBERSHIP),
                               pd.read_parquet(release / cobertura.EXEMPLOS,
                                               columns=["variant_id", *g7.COLUNAS_DE_SEQUENCIA, "binary_label",
                                                        "label_tier"]))
    if args.membros is not None:
        diferentes = g7.diferencas_de_tabela(tabela, g7.tabela_dos_estudos(pd.read_parquet(args.membros.expanduser())))
        if diferentes:
            return _falhar([f"--membros difere da tabela oficial em {diferentes}"])
    adapter = None
    if args.sistema == "MR":
        adapter = Path(manifesto["adapters_congelados"][chave]["arquivo"]).expanduser()
        if sha256_do_arquivo(adapter) != manifesto["adapters_congelados"][chave]["sha256"]:
            return _falhar([f"{adapter}: sha256 diferente do adapter congelado"])
    checkpoint_sha = sha256_do_arquivo(args.checkpoint.expanduser())
    if checkpoint_sha != identidade_dev["checkpoint_sha256"]:
        return _falhar([f"checkpoint {checkpoint_sha[:12]} nao e o do cache de desenvolvimento"])
    parametros = argparse.Namespace(
        sistema=args.sistema, semente_do_adapter=args.semente_do_adapter if args.sistema == "MR" else None,
        adapter=adapter, checkpoint=args.checkpoint.expanduser(), superficie=args.superficie,
        window_bp=int(identidade_dev["janela_bp"]), variantes_por_lote=int(identidade_dev["lote"]["variantes_por_lote"]),
        fragmento=int(identidade_dev["fragmento"]), out_dir=args.out_dir.expanduser())
    print(f"[estudos] manifesto {sha_do_manifesto[:12]} | {chave} | {len(tabela):,} variantes | janela "
          f"{parametros.window_bp} | lote {parametros.variantes_por_lote} | fragmento {parametros.fragmento}")
    print(f"  tabela oficial: {len(tabela):,} variantes, {int((tabela['chrom'] == 'chr8').sum())} no chr8, "
          f"conteudo {hash_do_conteudo(tabela)[:12]}")
    print(f"  {g7.LEITURA_DO_CHR8}")
    if args.so_conferir:
        print("PASSOU (so conferir): manifesto, declaracao, release, tabela oficial, cache de desenvolvimento, "
              "adapter e R03 conferem")
        return 0

    import torch

    from scripts.audit_variant_windows import abrir_fasta
    from scripts.extract_campaign_features import (
        ambiente_de_execucao,
        codigo_da_extracao,
        identidade,
        montar_sistema,
        revisao_do_codigo,
        rodar_extracao,
    )

    device = torch.device(args.device)
    modelo, adapter_sha = montar_sistema(parametros, device, carregar_campanha(args.campanha))
    identidade_nova = identidade(parametros, tabela=tabela, checkpoint_sha=checkpoint_sha, adapter_sha=adapter_sha,
                                 fasta_sha=sha256_do_arquivo(args.fasta.expanduser()), revisao=revisao_do_codigo(),
                                 codigo=codigo_da_extracao(), ambiente=ambiente_de_execucao(device))
    diferentes = g7.diferencas_do_estudo(identidade_dev, identidade_nova)
    if diferentes:
        return _falhar([f"a identidade dos estudos difere da de desenvolvimento em {diferentes}: outro caminho "
                        f"numerico, nada foi extraido"])
    fetch, _leitor = abrir_fasta(args.fasta.expanduser())
    return rodar_extracao(parametros, modelo, tabela, fetch, identidade_nova)


def _falhar(problemas: list[str]) -> int:
    print(f"FALHOU: {len(problemas)} problema(s)")
    for problema in problemas:
        print(f"  - {problema}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
