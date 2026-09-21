#!/usr/bin/env python3
"""G4: treina o adapter populacional (rsLoRA + MLM) sobre o plano de janelas, e o SMOKE que o valida no R03 real.

Roda no notebook COM GPU (torch + pysam + o pacote `lumina`). Le o plano, o split e o FASTA; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.1.

    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/train_population_adapter.py --smoke \\
        --checkpoint ~/artifacts/r03/best_checkpoint.pt --fasta ~/hg38/hg38.fa \\
        --plano-treino ~/artifacts/redesenho/g4_split/plano_treino.parquet \\
        --plano-validacao ~/artifacts/redesenho/g4_split/plano_validacao.parquet \\
        --out-dir ~/artifacts/redesenho/g4_smoke
    echo "exit=$?"

O QUE O SMOKE PROVA (e o que nenhum teste sintetico provava)
------------------------------------------------------------
1. Carrega o R03 CERTO e aplica rsLoRA nos modulos previstos -- com a familia pedida explicitamente.
2. Reconstroi a janela DESLOCADA do plano, aplica o ALT e mascara nos indices declarados.
3. Calcula a loss so nas posicoes previstas, com focal e contexto separados.
4. So o adapter esta no otimizador; base e cabecas nativas ficam identicas apos o passo.
5. O adapter MUDA apos o passo.
6. Salvar e recarregar pelo caminho real reproduz as predicoes.

**Nada disso demonstra ganho de regionalizacao.** Prova que a peca funciona como declarado.

DECISOES QUE FICAM REGISTRADAS, NAO IMPLICITAS
----------------------------------------------
- **Familia `lumina-r03` exigida.** `build_finetune_adapter` despacha por familia, e `"lumina"` e outro ramo:
  receber um checkpoint nao seleciona o R03.
- **`lumina.__file__` no manifesto.** Ha uma copia do pacote neste repo e outra no `lumina-inference`; sem isso
  nao se sabe qual implementacao rodou.
- **Modo do backbone (`train`/`eval`) declarado.** Congelar peso zera gradiente, nao desliga dropout: sao
  decisoes diferentes e o dropout muda o que o adapter ve.
- **A validacao agrega por SOMA E CONTAGEM**, nunca por media de medias entre lotes.
- **O scheduler conta atualizacoes do otimizador**, nao lotes -- com acumulacao os dois divergem.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedding_probe.windows import WindowError, build_window  # noqa: E402
from scripts.audit_variant_windows import abrir_fasta  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

FAMILIA = "lumina-r03"
COLUNAS = ("variant_id", "chrom", "pos_1based", "ref", "alt", "focal_index", "window_start", "spans", "fonte")


def exemplos_do_plano(plano: pd.DataFrame, fetch, *, window_bp: int) -> Iterator[Any]:
    """Reconstroi a janela DECLARADA de cada linha e monta o exemplo. Pula o que nao constroi, contando."""
    from eval.adapter import mlm

    for linha in plano.itertuples(index=False):
        try:
            janela = build_window(fetch, chrom=str(linha.chrom), pos_1based=int(linha.pos_1based),
                                  ref=str(linha.ref), alt=str(linha.alt), window_bp=window_bp,
                                  focal_index=int(linha.focal_index))
        except WindowError as exc:
            yield ("falha", exc.reason, str(linha.variant_id))
            continue
        if janela.window_start != int(linha.window_start):
            yield ("falha", "window_start_incoerente", str(linha.variant_id))
            continue
        yield ("ok", mlm.montar_exemplo(
            janela.alt_seq, mlm.spans_do_plano(linha.spans), variant_id=str(linha.variant_id),
            fonte=str(linha.fonte), focal_index=janela.focal_index), None)


def em_lotes(fonte: Iterator[Any], *, tamanho: int) -> Iterator[tuple[list[Any], dict[str, int]]]:
    lote: list[Any] = []
    falhas: dict[str, int] = {}
    for estado, carga, _ in fonte:
        if estado == "falha":
            falhas[carga] = falhas.get(carga, 0) + 1
            continue
        lote.append(carga)
        if len(lote) == tamanho:
            yield lote, falhas
            lote, falhas = [], {}
    if lote:
        yield lote, falhas


def escala_cosseno(passo: int, *, total: int, aquecimento: int, minimo: float = 0.01) -> float:
    """Conta ATUALIZACOES do otimizador, nao lotes: com acumulacao os dois divergem."""
    if aquecimento > 0 and passo < aquecimento:
        return (passo + 1) / aquecimento
    restante = max(1, total - aquecimento)
    progresso = min(1.0, max(0.0, (passo - aquecimento) / restante))
    return minimo + (1 - minimo) * 0.5 * (1 + math.cos(math.pi * progresso))


def montar(config: argparse.Namespace, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Carrega o R03, congela TUDO e so entao insere o rsLoRA. A ordem nao e negociavel."""
    from eval.adapter import treino
    from eval.clinvar.lora import apply_lora
    from eval.clinvar.r03_adapter import install_tilelang_fallback_shim

    install_tilelang_fallback_shim()
    from eval.clinvar.adapters import build_finetune_adapter

    adapter = build_finetune_adapter(FAMILIA, config.model_version, device,
                                     checkpoint_path=str(config.checkpoint))
    import lumina  # depois do shim, e so para registrar QUAL pacote foi importado

    congelados = treino.congelar_tudo(adapter.backbone)
    resumo = apply_lora(adapter.backbone, rank=config.lora_rank, alpha=config.lora_alpha,
                        dropout=config.lora_dropout, use_rslora=not config.sem_rslora)
    nomes = treino.assert_so_o_adapter_treina(adapter.backbone)

    adapter.backbone.train(not config.backbone_em_eval)
    proveniencia = {
        "familia": FAMILIA,
        "lumina_file": getattr(lumina, "__file__", None),
        "lumina_version": getattr(lumina, "__version__", None),
        "checkpoint": str(config.checkpoint),
        "parametros_congelados": congelados,
        "parametros_do_adapter": len(nomes),
        "modulos_adaptados": list(resumo.module_names),
        "use_rslora": resumo.use_rslora,
        "rank": resumo.rank, "alpha": resumo.alpha, "dropout": resumo.dropout,
        "modo_do_backbone": "eval" if config.backbone_em_eval else "train",
    }
    return adapter, resumo, proveniencia


def rodar_smoke(config: argparse.Namespace) -> int:
    import torch

    from eval.adapter import mlm, treino

    device = torch.device(config.device)
    adapter, resumo, proveniencia = montar(config, device)
    backbone = adapter.backbone

    fetch, leitor = abrir_fasta(config.fasta.expanduser())
    plano = pd.read_parquet(config.plano_treino.expanduser()).head(config.smoke_exemplos)
    faltando = [c for c in COLUNAS if c not in plano.columns]
    if faltando:
        print(f"FALHOU: faltam colunas {faltando} no plano")
        return 2

    checagens: list[dict[str, Any]] = []

    def checar(nome: str, ok: bool, detalhe: Any = "") -> bool:
        checagens.append({"checagem": nome, "ok": bool(ok), "detalhe": str(detalhe)})
        print(f"  [{'ok ' if ok else 'FALHA'}] {nome}  {detalhe}")
        return bool(ok)

    tudo = True
    tudo &= checar("1. familia e rsLoRA nos modulos previstos",
                   proveniencia["parametros_do_adapter"] > 0 and resumo.module_names,
                   f"{len(resumo.module_names)} modulos, rank={resumo.rank}, rslora={resumo.use_rslora}")
    tudo &= checar("1b. mlm_head NAO foi embrulhada",
                   not any("mlm_head" in n for n in resumo.module_names), "fica congelada e no grafo")

    lotes = list(em_lotes(exemplos_do_plano(plano, fetch, window_bp=config.window_bp),
                          tamanho=config.batch))
    if not lotes:
        return 2 if not checar("2. janelas reconstruidas", False, "nenhum exemplo") else 2
    exemplos, falhas = lotes[0]
    tudo &= checar("2. janela deslocada reconstruida com ALT e mascaras", not falhas and exemplos,
                   f"{len(exemplos)} exemplos, falhas={falhas}")

    lote = treino.montar_lote(exemplos, device=device)
    logits = treino.logits_mlm(adapter, lote.input_ids)
    tudo &= checar("3a. logits com 4 classes por posicao", logits.shape[-1] == len(mlm.SNV_BASES),
                   tuple(logits.shape))
    tudo &= checar("3b. alvos dentro de [0,3]", int(lote.alvo.min()) >= 0 and int(lote.alvo.max()) <= 3,
                   f"min={int(lote.alvo.min())} max={int(lote.alvo.max())}")

    pesos = {mlm.CATEGORIA_FOCAL: config.peso_focal, mlm.CATEGORIA_CONTEXTO: config.peso_contexto,
             mlm.CATEGORIA_REFERENCIA: config.peso_referencia}
    perda, decomposicao = treino.perda_do_lote(logits, lote, pesos)
    referencia = mlm.perda_ponderada(decomposicao, pesos)
    tudo &= checar("3c. loss em tensores bate com o nucleo sem torch",
                   abs(float(perda.detach()) - referencia) < 1e-4,
                   f"{float(perda.detach()):.6f} vs {referencia:.6f}")
    tudo &= checar("3d. focal e contexto separados",
                   decomposicao[mlm.CATEGORIA_FOCAL]["posicoes"] == len(exemplos)
                   and decomposicao[mlm.CATEGORIA_CONTEXTO]["posicoes"] > 0,
                   {c: decomposicao[c]["posicoes"] for c in mlm.CATEGORIAS})

    parametros = list(treino.parametros_do_adapter(backbone).values())
    identidades = [id(p) for p in parametros]
    tudo &= checar("4a. cada parametro do adapter uma unica vez no otimizador",
                   len(identidades) == len(set(identidades)), f"{len(parametros)} tensores")
    antes_congelados = treino.impressao_dos_congelados(backbone)
    copia = [p.detach().clone() for p in parametros]

    otimizador = torch.optim.AdamW(parametros, lr=config.lr)
    perda.backward()
    otimizador.step()
    otimizador.zero_grad(set_to_none=True)

    tudo &= checar("4b. backbone congelado identico apos o passo",
                   treino.impressao_dos_congelados(backbone) == antes_congelados,
                   antes_congelados[:16])
    mudou = any(not torch.equal(p.detach(), c) for p, c in zip(parametros, copia))
    tudo &= checar("5. adapter mudou apos o passo", mudou)

    backbone.eval()
    with torch.no_grad():
        antes = treino.logits_mlm(adapter, lote.input_ids).detach().clone()
    out_dir = config.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    caminho = out_dir / "adapter_smoke.pt"
    treino.salvar_adapter(caminho, backbone=backbone, resumo_lora=resumo,
                          config={k: str(v) for k, v in vars(config).items()},
                          identidades={"checkpoint_r03": str(config.checkpoint)}, metricas={}, passo=1)
    with torch.no_grad():
        for nome, parametro in backbone.named_parameters():
            if treino.e_do_adapter(nome):
                parametro.zero_()
        zerado = treino.logits_mlm(adapter, lote.input_ids)
        mudou_ao_zerar = not torch.allclose(zerado, antes)
    treino.carregar_adapter(caminho, backbone)
    with torch.no_grad():
        depois = treino.logits_mlm(adapter, lote.input_ids)
    tudo &= checar("6a. zerar o adapter muda as predicoes", mudou_ao_zerar)
    tudo &= checar("6b. recarregar reproduz as predicoes",
                   bool(torch.allclose(depois, antes, atol=1e-5)),
                   f"maior diferenca {float((depois - antes).abs().max()):.2e}")

    relatorio = {
        "proveniencia": proveniencia,
        "entradas": {
            "plano_treino": str(config.plano_treino),
            "plano_treino_sha256": sha256_file(config.plano_treino.expanduser()),
            "fasta": str(config.fasta), "leitor": leitor,
            "checkpoint_sha256": sha256_file(config.checkpoint.expanduser())
            if config.hashear_checkpoint else "nao calculado (--hashear-checkpoint)",
        },
        "receita": {"pesos": pesos, "criterio_primario": mlm.CRITERIO_PRIMARIO,
                    "window_bp": config.window_bp, "batch": config.batch, "lr": config.lr},
        "decomposicao_do_primeiro_lote": decomposicao,
        "por_fonte": treino.decomposicao_por_fonte(logits.detach(), lote),
        "checagens": checagens,
        "passou": tudo,
        "o_que_nao_prova": [
            "nao demonstra ganho de regionalizacao nem aprendizado de estrutura populacional",
            "mede reconstrucao mascarada num unico lote, com um unico passo",
        ],
        "saidas": {"adapter": str(caminho), "adapter_sha256": sha256_file(caminho)},
    }
    (out_dir / "smoke_do_adapter.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("proveniencia", "decomposicao_do_primeiro_lote", "por_fonte",
                                                "passou", "saidas")},
                     ensure_ascii=False, indent=2, default=str))
    if not tudo:
        print("\nFALHOU: o smoke do adapter nao passou (veja `checagens`).")
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path, help="best_checkpoint.pt do R03")
    parser.add_argument("--model-version", default="r03")
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--plano-treino", required=True, type=Path)
    parser.add_argument("--plano-validacao", type=Path)
    parser.add_argument("--window-bp", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--sem-rslora", action="store_true", help="usa LoRA classico (alpha/r) em vez de rsLoRA")
    parser.add_argument("--backbone-em-eval", action="store_true",
                        help="desliga dropout do backbone; congelar peso NAO faz isso sozinho")
    parser.add_argument("--peso-focal", type=float, default=1.0)
    parser.add_argument("--peso-contexto", type=float, default=0.5)
    parser.add_argument("--peso-referencia", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="roda so as checagens de integridade no R03 real")
    parser.add_argument("--smoke-exemplos", type=int, default=8)
    parser.add_argument("--hashear-checkpoint", action="store_true",
                        help="sha256 do checkpoint do R03; custa minutos num arquivo grande")
    parser.add_argument("--out-dir", required=True, type=Path)
    config = parser.parse_args(argv)

    if not config.smoke:
        print("FALHOU: por enquanto so --smoke esta implementado; o laco de treino entra depois que ele passar.")
        return 2
    return rodar_smoke(config)


if __name__ == "__main__":
    sys.exit(main())
