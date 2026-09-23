#!/usr/bin/env python3
"""G3: extrai as duas leituras candidatas de UM sistema (M0 ou MR) e grava um cache com identidade.

POR QUE CACHE, E NAO O TREINO PONTA A PONTA DE `eval/clinvar/train.py`
    Em M0 e em MR o backbone e congelado: a representacao de uma variante nao muda entre epocas. Refazer o
    forward a cada epoca daria o mesmo numero a um custo de horas por epoca. Extrair uma vez e treinar as cabecas
    sobre o cache e exato e barato -- e e o que o plano previa ("cache por sistema", secao 5.2).

O QUE ENTRA
    Os papeis de desenvolvimento declarados em `configs/campanha_r03_desenvolvimento.json`: `train` e
    `validation` do snapshot (o de `nenhum` cobre as tres politicas, que sao aninhadas) e o conjunto de selecao
    comum. O papel `test` (fold 0) e os estudos brasileiros sao RECUSADOS por `eval/campanha/recortes.py`.

O QUE SAI (em --out-dir)
    identidade.json      -- checkpoint, adapter, FASTA, extrator, janela, lote e tabela, com sha256. Uma retomada
                            com identidade diferente e recusada: um cache nao mistura dois sistemas.
    fragmento_NNNNN.npz  -- variant_id, papel e uma matriz float32 por extracao. Gravado a cada --fragmento
                            variantes: uma queda perde no maximo um fragmento.
    tabela.parquet       -- a tabela extraida (rotulos, paineis, clusters), para a cabeca nao reler o snapshot.
    manifesto.json       -- contagens, falhas, tempo e checagens, escrito no fim.

MODO --smoke
    Extrai poucas variantes e mede: custo por variante (com a estimativa para a tabela inteira), determinismo
    (o mesmo lote duas vezes), independencia da posicao no lote e, em MR, que o adapter esta ATIVO (as leituras
    mudam quando a escala do rsLoRA vai a zero). Nao grava cache.

USO (notebook)
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 scripts/extract_campaign_features.py --sistema M0 --smoke \\
        --checkpoint ~/artifacts/r03/best_checkpoint.pt --fasta ~/hg38/hg38.fa \\
        --snapshot ~/artifacts/redesenho/g2_final_nenhum/core_head_snapshot.parquet \\
        --selecao ~/artifacts/redesenho/g5_comum/selecao_comum.parquet \\
        --estudos ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \\
        --out-dir ~/artifacts/redesenho/g3_cache/M0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.campanha.recortes import (  # noqa: E402
    PAPEIS_DE_DESENVOLVIMENTO,
    adapter_congelado,
    carregar_campanha,
    hash_da_tabela,
    resumo_da_tabela,
    tabela_de_extracao,
)
from eval.embedding_probe.windows import WindowError, build_window, focal_offset  # noqa: E402
from scripts.audit_variant_windows import abrir_fasta  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

FAMILIA = "lumina-r03"
#: A pesquisa mediu diferenca 0 exata nas duas checagens de numerica; a tolerancia so evita reprovar por um kernel
#: nao deterministico da ordem de 1e-7, que nao afeta nenhuma comparacao da campanha. O valor medido e sempre
#: impresso.
TOLERANCIA_NUMERICA = 1e-5
VERSAO_DO_EXTRATOR = "campanha_r03_extracao_v1"
PREFIXO_DO_CHECKPOINT = "f2983560"  # R03 best_checkpoint.pt, passo 71.000 (contrato)


# ----------------------------------------------------------------------------------------------- identidade

def identidade(args: argparse.Namespace, *, tabela: pd.DataFrame, checkpoint_sha: str, adapter_sha: str | None,
               fasta_sha: str, revisao: str) -> dict[str, Any]:
    """O que torna duas extracoes o MESMO objeto. Tudo que muda um numero do cache esta aqui."""
    from eval.campanha.layout import EXTRACOES, RAIO_DO_CONTEXTO

    return {
        "versao_do_extrator": VERSAO_DO_EXTRATOR,
        "revisao_do_codigo": revisao,
        "sistema": args.sistema,
        "checkpoint_sha256": checkpoint_sha,
        "adapter_sha256": adapter_sha,
        "semente_do_adapter": args.semente_do_adapter if args.sistema == "MR" else None,
        "fasta_sha256": fasta_sha,
        "janela_bp": args.window_bp,
        "indice_focal": focal_offset(args.window_bp),
        "layout_da_janela": "centrada (L//2 - 1, a convencao do Mosaic)",
        "orientacao": "so a fita direta; sem media com o complemento reverso",
        "lote": {"layout": "[ref_0, alt_0, ref_1, alt_1, ...]", "variantes_por_lote": args.variantes_por_lote,
                 "sequencias_por_forward": 2 * args.variantes_por_lote,
                 "ultimo_lote": "completado com copias: o tamanho do forward nunca muda"},
        "precisao": {"dtype": "float32", "tf32": False},
        "extracoes": {nome: {"blocos": list(blocos), "dims": dims} for nome, (blocos, dims) in EXTRACOES.items()},
        "raio_do_contexto_local": RAIO_DO_CONTEXTO,
        "tabela_sha256_composicao": hash_da_tabela(tabela),
        "papeis": sorted(tabela["papel"].unique().tolist()),
    }


def revisao_do_codigo() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "desconhecida"
    except Exception:  # noqa: BLE001
        return "desconhecida"


# ----------------------------------------------------------------------------------------------- sistema

def montar_sistema(args: argparse.Namespace, device: Any, campanha: dict[str, Any]) -> tuple[Any, str | None]:
    """R03 congelado; em MR, mais o adapter declarado, carregado com o conjunto EXATO de chaves."""
    import torch

    from eval.adapter import treino
    from eval.clinvar.r03_adapter import install_tilelang_fallback_shim

    install_tilelang_fallback_shim()
    from eval.clinvar.adapters import build_finetune_adapter

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = build_finetune_adapter(FAMILIA, "r03", device, checkpoint_path=str(args.checkpoint))
    treino.congelar_tudo(adapter.backbone)
    adapter_sha = None
    if args.sistema == "MR":
        from eval.clinvar.lora import apply_lora

        declarado = adapter_congelado(campanha, args.semente_do_adapter)
        adapter_sha = sha256_file(args.adapter.expanduser())
        if adapter_sha != declarado["sha256"]:
            raise SystemExit(f"FALHOU: {args.adapter} tem sha256 {adapter_sha[:12]}, a declaracao da semente "
                             f"{args.semente_do_adapter} diz {declarado['sha256'][:12]}")
        lora = campanha["adapter_do_mr"]["receita"]["lora"]
        resumo = apply_lora(adapter.backbone, rank=lora["rank"], alpha=lora["alpha"], dropout=0.0,
                            use_rslora=lora["rslora"])
        esperados = json.loads(Path(args.superficie).read_text(encoding="utf-8"))
        if sorted(resumo.module_names) != sorted(esperados):
            raise SystemExit(f"FALHOU: superficie montada ({len(resumo.module_names)}) difere da aprovada "
                             f"({len(esperados)})")
        treino.carregar_adapter(args.adapter.expanduser(), adapter.backbone)
        treino.congelar_tudo(adapter.backbone)  # o adapter tambem fica congelado: aqui so se le
    from eval.embedding_probe.rich import assert_r03_head_layout

    assert_r03_head_layout(adapter.backbone)  # um layout de cabecas diferente desloca todas as colunas
    adapter.backbone.eval()
    return adapter, adapter_sha


def escalas_do_lora(backbone: Any, valor: float | None = None) -> list[float]:
    """Le (ou troca) a escala de todo LoRALinear. Zerar a escala desliga o delta sem descarregar nada."""
    from eval.clinvar.lora import LoRALinear

    antigas = []
    for modulo in backbone.modules():
        if isinstance(modulo, LoRALinear):
            antigas.append(modulo.scaling)
            if valor is not None:
                modulo.scaling = valor
    return antigas


def restaurar_escalas(backbone: Any, escalas: list[float]) -> None:
    from eval.clinvar.lora import LoRALinear

    modulos = [m for m in backbone.modules() if isinstance(m, LoRALinear)]
    for modulo, escala in zip(modulos, escalas):
        modulo.scaling = escala


# ----------------------------------------------------------------------------------------------- leitura

def extrair_lote(adapter: Any, janelas: list[Any], *, variantes_por_lote: int, focal: int) -> dict[str, np.ndarray]:
    """Um forward de tamanho fixo e as duas leituras das variantes reais do lote."""
    import torch

    from eval.campanha.layout import lote_pareado
    from eval.campanha.leituras import ler_extracoes

    sequencias, reais = lote_pareado([j.ref_seq for j in janelas], [j.alt_seq for j in janelas],
                                     variantes_por_lote)
    with torch.inference_mode():
        post = adapter.forward_hidden_states(adapter.tokenize(sequencias))
        leituras = ler_extracoes(adapter.backbone, post, focal=focal, reais=reais,
                                 refs=[j.ref for j in janelas], alts=[j.alt for j in janelas])
    return {nome: tensor.float().cpu().numpy() for nome, tensor in leituras.items()}


def janelas_da_tabela(tabela: pd.DataFrame, fetch, window_bp: int) -> tuple[list[tuple[int, Any]], dict[str, int]]:
    """(posicao na tabela, janela) das variantes que constroem; falhas contadas por motivo."""
    boas, falhas = [], {}
    for posicao, linha in enumerate(tabela.itertuples(index=False)):
        try:
            boas.append((posicao, build_window(fetch, chrom=str(linha.chrom), pos_1based=int(linha.pos_1based),
                                               ref=str(linha.ref), alt=str(linha.alt), window_bp=window_bp)))
        except WindowError as exc:
            falhas[exc.reason] = falhas.get(exc.reason, 0) + 1
    return boas, falhas


# ----------------------------------------------------------------------------------------------- smoke

def rodar_smoke(args: argparse.Namespace, adapter: Any, tabela: pd.DataFrame, fetch) -> int:
    focal = focal_offset(args.window_bp)
    amostra = tabela.groupby("papel", sort=True).head(args.limite).reset_index(drop=True)
    janelas, falhas = janelas_da_tabela(amostra, fetch, args.window_bp)
    if falhas.get("ref_mismatch"):
        print(f"FALHOU: {falhas['ref_mismatch']} ref_mismatch -- FASTA ou build errado")
        return 2
    lote = [j for _, j in janelas[:args.variantes_por_lote]]
    checagens: list[dict[str, Any]] = []

    def checar(nome: str, ok: bool, detalhe: Any) -> None:
        checagens.append({"checagem": nome, "ok": bool(ok), "detalhe": detalhe})
        print(f"  [{'ok ' if ok else 'FALHA'}] {nome}  {detalhe}")

    primeira = extrair_lote(adapter, lote, variantes_por_lote=args.variantes_por_lote, focal=focal)
    segunda = extrair_lote(adapter, lote, variantes_por_lote=args.variantes_por_lote, focal=focal)
    diferenca = max(float(np.abs(primeira[n] - segunda[n]).max()) for n in primeira)
    checar("determinismo: o mesmo lote duas vezes", diferenca <= TOLERANCIA_NUMERICA,
           f"max |diferenca| = {diferenca:.3e} (0 esperado)")

    sozinha = extrair_lote(adapter, lote[1:2], variantes_por_lote=args.variantes_por_lote, focal=focal)
    posicao = max(float(np.abs(primeira[n][1] - sozinha[n][0]).max()) for n in primeira)
    checar("independencia da posicao no lote (mesmo tamanho de forward)", posicao <= TOLERANCIA_NUMERICA,
           f"max |diferenca| = {posicao:.3e} (0 esperado)")

    finito = all(np.isfinite(v).all() for v in primeira.values())
    checar("leituras finitas", finito, {n: list(v.shape) for n, v in primeira.items()})

    if args.sistema == "MR":
        escalas = escalas_do_lora(adapter.backbone, 0.0)
        try:
            sem_delta = extrair_lote(adapter, lote, variantes_por_lote=args.variantes_por_lote, focal=focal)
        finally:
            restaurar_escalas(adapter.backbone, escalas)
        efeito = {n: float(np.abs(primeira[n] - sem_delta[n]).mean()) for n in primeira}
        checar("adapter ATIVO: as leituras mudam com a escala do rsLoRA em zero",
               all(v > 0 for v in efeito.values()), {n: f"{v:.3e}" for n, v in efeito.items()})

    import time as _time
    inicio = _time.perf_counter()
    medidos = [j for _, j in janelas]
    for comeco in range(0, len(medidos), args.variantes_por_lote):
        extrair_lote(adapter, medidos[comeco:comeco + args.variantes_por_lote],
                     variantes_por_lote=args.variantes_por_lote, focal=focal)
    por_variante = (_time.perf_counter() - inicio) / max(1, len(medidos))
    total_horas = por_variante * len(tabela) / 3600
    checar("custo medido", True, f"{por_variante:.4f} s/variante em {len(medidos)} variantes -> "
                                 f"~{total_horas:.1f} h para as {len(tabela):,} da tabela")

    relatorio = {"sistema": args.sistema, "variantes_por_lote": args.variantes_por_lote, "falhas": falhas,
                 "segundos_por_variante": por_variante, "horas_estimadas_para_a_tabela": total_horas,
                 "tabela": resumo_da_tabela(tabela), "checagens": checagens,
                 "passou": all(c["ok"] for c in checagens)}
    destino = args.out_dir.expanduser()
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "smoke_da_extracao.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("sistema", "variantes_por_lote", "falhas",
                                                "horas_estimadas_para_a_tabela", "tabela", "passou")},
                     ensure_ascii=False, indent=2))
    return 0 if relatorio["passou"] else 2


# ----------------------------------------------------------------------------------------------- extracao

def conferir_identidade(destino: Path, ident: dict[str, Any]) -> str | None:
    """Grava a identidade num cache novo; num cache existente, recusa se ela mudou (menos a revisao do codigo,
    que muda sem mudar numero -- quem muda numero tem de subir VERSAO_DO_EXTRATOR)."""
    arquivo = destino / "identidade.json"
    if not arquivo.exists():
        destino.mkdir(parents=True, exist_ok=True)
        arquivo.write_text(json.dumps(ident, ensure_ascii=False, indent=2), encoding="utf-8")
        return None
    gravada = json.loads(arquivo.read_text(encoding="utf-8"))
    diferentes = sorted(k for k in set(gravada) | set(ident)
                        if k != "revisao_do_codigo" and gravada.get(k) != ident.get(k))
    if diferentes:
        return (f"{destino} ja tem um cache com outra identidade (difere em {diferentes}). Use outro --out-dir: "
                f"um cache nao mistura dois objetos")
    print(f"[retomada] identidade confere; revisao gravada {str(gravada.get('revisao_do_codigo'))[:12]}")
    return None


def ja_extraidas(destino: Path) -> set[str]:
    feitas: set[str] = set()
    for arquivo in sorted(destino.glob("fragmento_*.npz")):
        with np.load(arquivo, allow_pickle=False) as dados:
            feitas.update(dados["variant_id"].tolist())
    return feitas


def rodar_extracao(args: argparse.Namespace, adapter: Any, tabela: pd.DataFrame, fetch,
                   ident: dict[str, Any]) -> int:
    destino = args.out_dir.expanduser()
    destino.mkdir(parents=True, exist_ok=True)
    problema = conferir_identidade(destino, ident)
    if problema:
        print(f"FALHOU: {problema}")
        return 2
    tabela.to_parquet(destino / "tabela.parquet", index=False)

    feitas = ja_extraidas(destino)
    pendentes = tabela[~tabela["variant_id"].isin(feitas)].reset_index(drop=True)
    print(f"[extracao] {len(tabela):,} na tabela, {len(feitas):,} ja no cache, {len(pendentes):,} pendentes")
    focal = focal_offset(args.window_bp)
    proximo = len(list(destino.glob("fragmento_*.npz")))
    falhas: dict[str, int] = {}
    inicio = time.perf_counter()
    feitas_agora = 0
    for comeco in range(0, len(pendentes), args.fragmento):
        pedaco = pendentes.iloc[comeco:comeco + args.fragmento]
        janelas, falhas_do_pedaco = janelas_da_tabela(pedaco, fetch, args.window_bp)
        for motivo, n in falhas_do_pedaco.items():
            falhas[motivo] = falhas.get(motivo, 0) + n
        if falhas.get("ref_mismatch"):
            print(f"FALHOU: {falhas['ref_mismatch']} ref_mismatch -- FASTA ou build errado")
            return 2
        blocos: dict[str, list[np.ndarray]] = {}
        for i in range(0, len(janelas), args.variantes_por_lote):
            grupo = janelas[i:i + args.variantes_por_lote]
            for nome, matriz in extrair_lote(adapter, [j for _, j in grupo],
                                             variantes_por_lote=args.variantes_por_lote, focal=focal).items():
                blocos.setdefault(nome, []).append(matriz)
        posicoes = [p for p, _ in janelas]
        np.savez(destino / f"fragmento_{proximo:05d}.npz",
                 variant_id=pedaco["variant_id"].to_numpy()[posicoes].astype(str),
                 papel=pedaco["papel"].to_numpy()[posicoes].astype(str),
                 **{nome: np.concatenate(partes) for nome, partes in blocos.items()})
        proximo += 1
        feitas_agora += len(janelas)
        taxa = (time.perf_counter() - inicio) / max(1, feitas_agora)
        print(f"  fragmento {proximo - 1:05d}: {feitas_agora:,}/{len(pendentes):,}  ({taxa:.4f} s/variante, "
              f"faltam ~{taxa * (len(pendentes) - comeco - len(pedaco)) / 3600:.1f} h)")

    total = ja_extraidas(destino)
    manifesto = {
        "identidade": ident,
        "variantes_na_tabela": int(len(tabela)),
        "variantes_no_cache": int(len(total)),
        "faltando": int(len(tabela) - len(total)),
        "falhas_desta_execucao": falhas,
        "segundos_desta_execucao": round(time.perf_counter() - inicio, 1),
        "tabela": resumo_da_tabela(tabela),
        "completo": len(total) == len(tabela) - sum(falhas.values()),
    }
    (destino / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifesto[k] for k in ("variantes_na_tabela", "variantes_no_cache", "faltando",
                                                 "falhas_desta_execucao", "completo")}, ensure_ascii=False))
    return 0 if manifesto["completo"] else 2


# ----------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sistema", required=True, choices=("M0", "MR"))
    parser.add_argument("--adapter", type=Path, help="adapter_melhor.pt da semente (so MR)")
    parser.add_argument("--semente-do-adapter", type=int, help="semente declarada do adapter (so MR)")
    parser.add_argument("--superficie", type=Path, default=Path("configs/adapter_r03_superficie.json"))
    parser.add_argument("--campanha", type=Path, default=Path("configs/campanha_r03_desenvolvimento.json"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path, help="g2_final_nenhum: cobre as tres politicas")
    parser.add_argument("--selecao", required=True, type=Path)
    parser.add_argument("--estudos", required=True, type=Path, help="membros dos estudos: recusados na tabela")
    parser.add_argument("--papeis", default=",".join(PAPEIS_DE_DESENVOLVIMENTO))
    parser.add_argument("--window-bp", type=int, default=4096)
    parser.add_argument("--variantes-por-lote", type=int, default=8)
    parser.add_argument("--fragmento", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limite", type=int, default=16, help="variantes por papel no smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if (args.sistema == "MR") != (args.adapter is not None and args.semente_do_adapter is not None):
        print("FALHOU: MR exige --adapter e --semente-do-adapter; M0 nao aceita nenhum dos dois")
        return 2
    campanha = carregar_campanha(args.campanha)
    papeis = tuple(p.strip() for p in args.papeis.split(",") if p.strip())
    tabela = tabela_de_extracao(pd.read_parquet(args.snapshot.expanduser()),
                                pd.read_parquet(args.selecao.expanduser()),
                                pd.read_parquet(args.estudos.expanduser(), columns=["variant_id"]),
                                papeis=papeis)
    print(f"[tabela] {len(tabela):,} variantes: "
          + ", ".join(f"{p} {n:,}" for p, n in tabela["papel"].value_counts().sort_index().items()))

    checkpoint_sha = sha256_file(args.checkpoint.expanduser())
    if not checkpoint_sha.startswith(PREFIXO_DO_CHECKPOINT):
        print(f"FALHOU: checkpoint {checkpoint_sha[:12]} nao e o R03 do contrato ({PREFIXO_DO_CHECKPOINT})")
        return 2
    import torch

    device = torch.device(args.device)
    adapter, adapter_sha = montar_sistema(args, device, campanha)
    fetch, _leitor = abrir_fasta(args.fasta.expanduser())
    if args.smoke:
        return rodar_smoke(args, adapter, tabela, fetch)
    ident = identidade(args, tabela=tabela, checkpoint_sha=checkpoint_sha, adapter_sha=adapter_sha,
                       fasta_sha=sha256_file(args.fasta.expanduser()), revisao=revisao_do_codigo())
    return rodar_extracao(args, adapter, tabela, fetch, ident)


if __name__ == "__main__":
    sys.exit(main())
