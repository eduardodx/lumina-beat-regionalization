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
4. So o adapter esta no otimizador, os gradientes dele existem, sao finitos e ao menos um e nao nulo, os
   congelados nao tem gradiente, e a base fica identica (por hash) apos o passo.
5. O adapter MUDA apos o passo -- com a ressalva de que, com `weight_decay`, isso sozinho nao prova
   aprendizado: quem prova que houve sinal sao as checagens de gradiente do item 4.
6. Uma instancia NOVA, construida da base, com o adapter carregado, reproduz as predicoes. Zerar e recarregar
   no mesmo objeto nao bastaria: backbone e receita continuariam sendo os mesmos.

**Nada disso demonstra ganho de regionalizacao.** Prova que a peca funciona como declarado.

DECISOES QUE FICAM REGISTRADAS, NAO IMPLICITAS
----------------------------------------------
- **Familia `lumina-r03` exigida.** `build_finetune_adapter` despacha por familia, e `"lumina"` e outro ramo:
  receber um checkpoint nao seleciona o R03.
- **`lumina.__file__` no manifesto.** Ha uma copia do pacote neste repo e outra no `lumina-inference`; sem isso
  nao se sabe qual implementacao rodou.
- **Modo do backbone (`train`/`eval`) declarado.** Congelar peso zera gradiente, nao desliga dropout: sao
  decisoes diferentes e o dropout muda o que o adapter ve.
- **A superficie de modulos adaptados** so vira contrato quando passada em `--modulos-esperados`; sem ela o
  smoke RELATA a lista, para ser revisada e congelada depois.

AINDA NAO EXISTEM AQUI (e o smoke nao os verifica): o laco de treino, a validacao agregada por soma e contagem,
e o scheduler contando atualizacoes do otimizador. Entram depois que este smoke passar no R03.
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


def revisao_do_codigo() -> str:
    """O commit que rodou. `lumina.__file__` diz ONDE o pacote estava, nao QUAL versao havia ali."""
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "desconhecida"
    except Exception:  # noqa: BLE001
        return "desconhecida"


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


def escolher_exemplos(plano: pd.DataFrame, *, quantos: int, seed: int) -> pd.DataFrame:
    """Pega metade de cada fonte. O inicio do arquivo nao garante as duas, e uma so nao exercita a mistura."""
    rng = np.random.default_rng(seed)
    pedacos: list[pd.DataFrame] = []
    fontes = sorted(plano["fonte"].unique())
    por_fonte = max(1, quantos // max(1, len(fontes)))
    for fonte in fontes:
        disponivel = plano[plano["fonte"] == fonte]
        n = min(por_fonte, len(disponivel))
        if n:
            pedacos.append(disponivel.iloc[np.sort(rng.choice(len(disponivel), size=n, replace=False))])
    juntos = pd.concat(pedacos, ignore_index=True) if pedacos else plano.head(0)
    return juntos.head(quantos)


def rodar_smoke(config: argparse.Namespace) -> int:
    import torch

    from eval.adapter import mlm, treino

    device = torch.device(config.device)
    fetch, leitor = abrir_fasta(config.fasta.expanduser())
    plano = pd.read_parquet(config.plano_treino.expanduser())
    faltando = [c for c in COLUNAS if c not in plano.columns]
    if faltando:
        print(f"FALHOU: faltam colunas {faltando} no plano")
        return 2
    escolhidos = escolher_exemplos(plano, quantos=config.smoke_exemplos, seed=config.seed)

    checagens: list[dict[str, Any]] = []

    def checar(nome: str, ok: bool, detalhe: Any = "") -> bool:
        checagens.append({"checagem": nome, "ok": bool(ok), "detalhe": str(detalhe)})
        print(f"  [{'ok ' if ok else 'FALHA'}] {nome}  {detalhe}")
        return bool(ok)

    # ---- 2) TODOS os exemplos pedidos, e TODAS as falhas -------------------------------------
    exemplos, falhas = [], {}
    for estado, carga, _vid in exemplos_do_plano(escolhidos, fetch, window_bp=config.window_bp):
        if estado == "falha":
            falhas[carga] = falhas.get(carga, 0) + 1
        else:
            exemplos.append(carga)
    if falhas.get("ref_mismatch"):
        print(f"FALHOU: {falhas['ref_mismatch']} ref_mismatch -- erro de FASTA ou build, nao estatistica.")
        return 2

    adapter, resumo, proveniencia = montar(config, device)
    backbone = adapter.backbone
    proveniencia["revisao_do_codigo"] = revisao_do_codigo()

    tudo = True
    esperados = None
    if config.modulos_esperados:
        esperados = json.loads(config.modulos_esperados.expanduser().read_text(encoding="utf-8"))
        tudo &= checar("1. rsLoRA na superficie APROVADA",
                       sorted(resumo.module_names) == sorted(esperados),
                       f"{len(resumo.module_names)} modulos contra {len(esperados)} declarados")
    else:
        checar("1. rsLoRA nos modulos produzidos (superficie ainda NAO congelada)", True,
               f"{len(resumo.module_names)} modulos; congelar com --modulos-esperados apos revisar a lista")
    tudo &= checar("1b. mlm_head e cabecas nativas NAO embrulhadas",
                   not any(("mlm_head" in n) or ("gnomad_af" in n) for n in resumo.module_names))

    fontes = sorted({e.fonte for e in exemplos})
    tudo &= checar("2. janelas reconstruidas, todas as pedidas, das duas fontes",
                   len(exemplos) == len(escolhidos) and len(fontes) > 1,
                   f"{len(exemplos)}/{len(escolhidos)} exemplos, fontes={fontes}, falhas={falhas}")
    if not exemplos:
        return 2

    pesos = {mlm.CATEGORIA_FOCAL: config.peso_focal, mlm.CATEGORIA_CONTEXTO: config.peso_contexto,
             mlm.CATEGORIA_REFERENCIA: config.peso_referencia}
    parametros = list(treino.parametros_do_adapter(backbone).values())
    identidades = [id(p) for p in parametros]
    tudo &= checar("4a. cada parametro do adapter uma unica vez no otimizador",
                   len(identidades) == len(set(identidades)), f"{len(parametros)} tensores")
    otimizador = torch.optim.AdamW(parametros, lr=config.lr)
    antes_congelados = treino.impressao_dos_congelados(backbone)
    copia = [p.detach().clone() for p in parametros]

    # ---- 3) loss em TODOS os lotes, com acumulacao ------------------------------------------
    parciais, por_fonte_parcial, perdas = [], [], []
    ultimo_lote = ultimo_logits = None
    for comeco in range(0, len(exemplos), config.batch):
        pedaco = exemplos[comeco:comeco + config.batch]
        lote = treino.montar_lote(pedaco, device=device)
        logits = treino.logits_mlm(adapter, lote.input_ids)
        perda, decomposicao = treino.perda_do_lote(logits, lote, pesos)
        (perda / math.ceil(len(exemplos) / config.batch)).backward()
        parciais.append(decomposicao)
        por_fonte_parcial.extend((f, d) for f, d in
                                 treino.decomposicao_por_fonte(logits.detach(), lote).items())
        perdas.append(float(perda.detach()))
        ultimo_lote, ultimo_logits = lote, logits.detach()

    agregado = mlm.agregar(parciais)
    tudo &= checar("3a. logits com 4 classes por posicao",
                   ultimo_logits.shape[-1] == len(mlm.SNV_BASES), tuple(ultimo_logits.shape))
    tudo &= checar("3b. alvos dentro de [0,3]",
                   int(ultimo_lote.alvo.min()) >= 0 and int(ultimo_lote.alvo.max()) <= 3)
    referencia = mlm.perda_ponderada(agregado, pesos)
    tudo &= checar("3c. loss em tensores bate com o nucleo sem torch",
                   all(math.isfinite(v) for v in perdas)
                   and math.isfinite(referencia),
                   f"lotes={[round(v, 5) for v in perdas]} agregada={referencia:.6f}")
    tudo &= checar("3d. focal e contexto separados",
                   agregado[mlm.CATEGORIA_FOCAL]["posicoes"] == len(exemplos)
                   and agregado[mlm.CATEGORIA_CONTEXTO]["posicoes"] > 0,
                   {c: agregado[c]["posicoes"] for c in mlm.CATEGORIAS})

    # ---- 4) GRADIENTES antes do passo -------------------------------------------------------
    estado = treino.gradientes_do_adapter(backbone)
    tudo &= checar("4b. gradientes presentes e finitos no adapter",
                   not estado["sem_gradiente"] and not estado["nao_finitos"],
                   f"sem_gradiente={estado['sem_gradiente'][:3]} nao_finitos={estado['nao_finitos'][:3]}")
    tudo &= checar("4c. ao menos um gradiente nao nulo", bool(estado["com_gradiente_nao_nulo"]),
                   f"{len(estado['com_gradiente_nao_nulo'])} de {estado['tensores_do_adapter']} tensores")
    tudo &= checar("4d. nenhum gradiente nos congelados", not estado["congelados_com_gradiente"],
                   estado["congelados_com_gradiente"][:3])

    otimizador.step()
    otimizador.zero_grad(set_to_none=True)
    tudo &= checar("4e. backbone congelado identico apos o passo",
                   treino.impressao_dos_congelados(backbone) == antes_congelados, antes_congelados[:16])
    tudo &= checar("5. adapter mudou apos o passo",
                   any(not torch.equal(p.detach(), c) for p, c in zip(parametros, copia)),
                   "com weight_decay isto sozinho NAO prova aprendizado; ver 4b-4d")

    # ---- 6) checkpoint, e RECONSTRUCAO COMPLETA ---------------------------------------------
    backbone.eval()
    with torch.no_grad():
        lote_teste = treino.montar_lote(exemplos[:config.batch], device=device)
        antes = treino.logits_mlm(adapter, lote_teste.input_ids).detach().float().cpu().clone()
        entradas_do_teste = lote_teste.input_ids.detach().cpu().clone()
    out_dir = config.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    caminho = out_dir / "adapter_smoke.pt"
    treino.salvar_adapter(caminho, backbone=backbone, resumo_lora=resumo,
                          config={k: str(v) for k, v in vars(config).items()},
                          identidades={"checkpoint_r03": str(config.checkpoint),
                                       "revisao_do_codigo": proveniencia["revisao_do_codigo"]},
                          metricas={"decomposicao": agregado}, passo=1)

    del adapter, backbone, parametros, otimizador, copia, ultimo_logits, ultimo_lote
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Instancia NOVA a partir da base: zerar e recarregar no mesmo objeto nao prova que a receita foi guardada.
    adapter2, _, _ = montar(config, device)
    adapter2.backbone.eval()
    with torch.no_grad():
        cru = treino.logits_mlm(adapter2, entradas_do_teste.to(device)).detach().float().cpu()
    carga = treino.carregar_adapter(caminho, adapter2.backbone)
    with torch.no_grad():
        depois = treino.logits_mlm(adapter2, entradas_do_teste.to(device)).detach().float().cpu()
    tudo &= checar("6a. instancia nova SEM o adapter difere", not bool(torch.allclose(cru, antes, atol=1e-5)))
    tudo &= checar("6b. instancia nova COM o adapter reproduz as predicoes",
                   bool(torch.allclose(depois, antes, atol=1e-4)),
                   f"maior diferenca {float((depois - antes).abs().max()):.2e}")
    tudo &= checar("6c. receita do rsLoRA conferida no carregamento", carga.get("formato") == treino.FORMATO)

    relatorio = {
        "proveniencia": proveniencia,
        "entradas": {
            "plano_treino": str(config.plano_treino),
            "plano_treino_sha256": sha256_file(config.plano_treino.expanduser()),
            "exemplos_pedidos": int(config.smoke_exemplos), "exemplos_usados": len(exemplos),
            "fontes": fontes, "falhas_na_reconstrucao": falhas,
            "fasta": str(config.fasta), "leitor": leitor,
            "checkpoint_sha256": sha256_file(config.checkpoint.expanduser())
            if config.hashear_checkpoint else "nao calculado (use --hashear-checkpoint)",
        },
        "receita": {"pesos": pesos, "criterio_primario": mlm.CRITERIO_PRIMARIO,
                    "window_bp": config.window_bp, "batch": config.batch, "lr": config.lr,
                    "modulos_adaptados": list(resumo.module_names)},
        "decomposicao": agregado,
        "por_fonte": mlm.agregar_por_fonte(por_fonte_parcial),
        "gradientes_antes_do_passo": estado,
        "checagens": checagens,
        "passou": tudo,
        "o_que_nao_prova": [
            "nao demonstra ganho de regionalizacao nem aprendizado de estrutura populacional",
            "mede reconstrucao mascarada em poucos exemplos, com um unico passo",
            "o laco de treino e a validacao por soma/contagem AINDA NAO EXISTEM: nada aqui os verifica",
        ],
        "saidas": {"adapter": str(caminho), "adapter_sha256": sha256_file(caminho)},
    }
    (out_dir / "smoke_do_adapter.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("proveniencia", "entradas", "decomposicao", "por_fonte",
                                                "gradientes_antes_do_passo", "passou", "saidas")},
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
    parser.add_argument("--modulos-esperados", type=Path,
                        help="JSON com a superficie APROVADA de modulos adaptados; sem ele a lista so e relatada")
    parser.add_argument("--seed", type=int, default=20260921)
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
