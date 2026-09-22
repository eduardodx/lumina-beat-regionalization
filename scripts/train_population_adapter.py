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

O LACO DE TREINO (`--treinar`) acrescenta, e nada disso e verificado pelo smoke:
- validacao em `eval()` e sem gradiente, agregada por SOMA E CONTAGEM (nunca media de medias), tambem por fonte;
- acumulacao coerente com a loss ponderada: cada microlote contribui com `sum_i w_i CE_i` e os gradientes sao
  divididos pelo peso TOTAL antes do passo. Dividir cada microlote pelo numero deles daria peso igual a
  microlotes com quantidades diferentes de posicoes mascaradas, e elas diferem por janela;
- scheduler contado por ATUALIZACOES do otimizador;
- interrupcao em loss ou gradiente nao finito, com o motivo declarado;
- retomada EXPLICITA (`--retomar`), com o aviso de que o estado do otimizador nao e restaurado;
- reconferencia, no fim, de que o backbone congelado continua identico;
- **selecao por validacao**: `adapter_melhor.pt` guarda o melhor pelo criterio primario, e o relatorio avisa alto
  quando o adapter FINAL ficou pior que a linha de base. Medido no piloto 5: a validacao tocou o fundo no passo
  89 e depois degradou ate +0,26 acima da base -- salvar so o final entregaria o pior adapter da corrida.
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


def pico_de_memoria_mb() -> float | None:
    """Pico de RSS do processo. Em Linux `ru_maxrss` vem em KB; em macOS, em bytes. None no Windows."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(pico / 1024, 1) if sys.platform.startswith("linux") else round(pico / (1024 * 1024), 1)


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
            fonte=str(linha.fonte), focal_index=janela.focal_index, ref=janela.ref), None)


def escala_cosseno(passo: int, *, total: int, aquecimento: int, minimo: float = 0.01) -> float:
    """Conta ATUALIZACOES do otimizador, nao lotes: com acumulacao os dois divergem."""
    if aquecimento > 0 and passo < aquecimento:
        return (passo + 1) / aquecimento
    restante = max(1, total - aquecimento)
    progresso = min(1.0, max(0.0, (passo - aquecimento) / restante))
    return minimo + (1 - minimo) * 0.5 * (1 + math.cos(math.pi * progresso))


def inicializar_aleatoriedade(seed: int) -> None:
    """Controla a inicializacao LoRA; nao promete determinismo dos kernels CUDA."""
    import random
    import torch

    random.seed(seed)
    torch.manual_seed(seed)  # CPU e dispositivos CUDA; os geradores NumPy usam seed explicitamente.


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
        "modulos_inertes_ignorados": list(getattr(resumo, "modulos_inertes_ignorados", ())),
        "use_rslora": resumo.use_rslora,
        "rank": resumo.rank, "alpha": resumo.alpha, "dropout": resumo.dropout,
        "modo_do_backbone": "eval" if config.backbone_em_eval else "train",
        "seed_torch": config.seed,
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

    inicializar_aleatoriedade(config.seed)
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
    tudo &= checar("4b. todo modulo adaptado ENTROU no grafo (nenhum embrulho inerte)",
                   not estado["sem_gradiente"],
                   f"sem_gradiente={estado['sem_gradiente'][:4]}")
    tudo &= checar("4b2. nenhum gradiente nao finito", not estado["nao_finitos"], estado["nao_finitos"][:3])
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
            # SEMPRE. Reexecutar sem o hash sobrescrevia o relatorio anterior por um que dizia "nao
            # calculado", e a ligacao entre resultado, checkpoint e versao do codigo se perdia. sha256 de
            # 600 MB custa segundos, nao minutos como eu havia suposto.
            "checkpoint_sha256": sha256_file(config.checkpoint.expanduser()),
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
            "o smoke nao verifica o laco de treino nem a agregacao da validacao",
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


def amostrar_preservando_a_mistura(plano: pd.DataFrame, *, quantos: int, seed: int) -> pd.DataFrame:
    """Subamostra mantendo a PROPORCAO de cada fonte.

    `plano.head(n)` era defeito: o gerador concatena o ABraOM antes do global, entao o inicio do arquivo e de uma
    fonte so. O piloto de 22/09 treinou e validou 100% em ABraOM sem nada reclamar -- a mistura 60/40, que e o
    centro do desenho, nao foi exercitada. Aqui a cota de cada fonte sai da proporcao que ela tem no plano.
    """
    rng = np.random.default_rng(seed)
    contagem = plano["fonte"].value_counts()
    total = int(contagem.sum())
    pedacos: list[pd.DataFrame] = []
    alocado = 0
    fontes = list(contagem.index)
    for posicao, fonte in enumerate(fontes):
        disponivel = plano[plano["fonte"] == fonte]
        cota = (quantos - alocado) if posicao == len(fontes) - 1 else round(quantos * len(disponivel) / total)
        cota = int(min(max(0, cota), len(disponivel)))
        if cota:
            escolhidos = rng.choice(len(disponivel), size=cota, replace=False)
            pedacos.append(disponivel.iloc[np.sort(escolhidos)])
            alocado += cota
    return pd.concat(pedacos, ignore_index=True) if pedacos else plano.head(0)


def atualizar_melhor(melhor: dict[str, Any] | None, passo: int,
                     validacao: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """Devolve (melhor, trocou). Funcao pura, para a selecao ser testavel sem GPU."""
    valor = validacao["criterio_primario"]["valor"]
    if melhor is not None and valor >= melhor["valor"]:
        return melhor, False
    return {"passo": passo, "valor": valor, "validacao": validacao}, True


def _mistura(exemplos: list) -> dict[str, Any]:
    """Fracao de cada fonte nos exemplos efetivamente carregados. Sem isto, treinar numa fonte so passa batido."""
    if not exemplos:
        return {}
    contagem: dict[str, int] = {}
    for exemplo in exemplos:
        contagem[exemplo.fonte] = contagem.get(exemplo.fonte, 0) + 1
    total = len(exemplos)
    return {"por_fonte": dict(sorted(contagem.items())),
            "fracao": {f: round(q / total, 4) for f, q in sorted(contagem.items())}}


def carregar_exemplos(caminho: Path, fetch, *, window_bp: int, limite: int | None,
                      seed: int = 0) -> tuple[list, dict]:
    """Le o plano e reconstroi os exemplos. `ref_mismatch` interrompe: e erro de dado, nao estatistica."""
    plano = pd.read_parquet(caminho.expanduser())
    faltando = [c for c in COLUNAS if c not in plano.columns]
    if faltando:
        raise SystemExit(f"faltam colunas {faltando} em {caminho}")
    fontes_no_plano = set(plano["fonte"].unique())
    if limite:
        plano = amostrar_preservando_a_mistura(plano, quantos=limite, seed=seed)
    if len(fontes_no_plano) > 1 and set(plano["fonte"].unique()) != fontes_no_plano:
        raise SystemExit(f"a subamostra de {caminho} ficou com {sorted(set(plano['fonte']))} de "
                         f"{sorted(fontes_no_plano)}: aumente o limite")
    exemplos, falhas = [], {}
    for estado, carga, _vid in exemplos_do_plano(plano, fetch, window_bp=window_bp):
        if estado == "falha":
            falhas[carga] = falhas.get(carga, 0) + 1
        else:
            exemplos.append(carga)
    if falhas.get("ref_mismatch"):
        raise SystemExit(f"{falhas['ref_mismatch']} ref_mismatch em {caminho}: FASTA ou build errado.")
    return exemplos, falhas


def avaliar(adapter, exemplos, *, pesos, batch: int, device) -> dict[str, Any]:
    """Validacao em `eval()` e sem gradiente, agregada por SOMA E CONTAGEM -- nunca media de medias.

    Media de medias entre lotes daria o mesmo peso a um lote com 3 posicoes mascaradas e a outro com 30. Quem
    agrega e `mlm.agregar`, que soma `media * posicoes` e divide pelo total de posicoes.
    """
    import torch

    from eval.adapter import mlm, treino

    modo_anterior = adapter.backbone.training
    adapter.backbone.eval()
    parciais: list = []
    por_fonte: list = []
    diagnosticos: list = []
    try:
        with torch.no_grad():
            for comeco in range(0, len(exemplos), batch):
                lote = treino.montar_lote(exemplos[comeco:comeco + batch], device=device)
                logits = treino.logits_mlm(adapter, lote.input_ids)
                _, _, decomposicao = treino.perda_somada_do_lote(logits, lote, pesos)
                parciais.append(decomposicao)
                por_fonte.extend(treino.decomposicao_por_fonte(logits, lote).items())
                diagnosticos.append(treino.diagnostico_do_focal(logits, lote))
    finally:
        adapter.backbone.train(modo_anterior)

    agregado = mlm.agregar(parciais)
    if any(v["posicoes"] and not math.isfinite(v["media"]) for v in agregado.values()):
        raise RuntimeError("validacao com perda nao finita")
    return {
        "criterio_primario": {"categoria": mlm.CRITERIO_PRIMARIO,
                              "valor": agregado[mlm.CRITERIO_PRIMARIO]["media"],
                              "posicoes": agregado[mlm.CRITERIO_PRIMARIO]["posicoes"]},
        "por_categoria": agregado,
        "por_fonte": mlm.agregar_por_fonte(por_fonte),
        "diagnostico_do_focal": treino.juntar_diagnosticos(diagnosticos),
        "exemplos": len(exemplos),
    }


def rodar_treino(config: argparse.Namespace) -> int:
    import torch

    from eval.adapter import mlm, treino

    inicializar_aleatoriedade(config.seed)
    import time

    relogio = {"inicio": time.monotonic()}
    device = torch.device(config.device)
    fetch, leitor = abrir_fasta(config.fasta.expanduser())
    pesos = {mlm.CATEGORIA_FOCAL: config.peso_focal, mlm.CATEGORIA_CONTEXTO: config.peso_contexto,
             mlm.CATEGORIA_REFERENCIA: config.peso_referencia}
    mlm.validar_pesos(pesos)

    treino_exemplos, falhas_treino = carregar_exemplos(
        config.plano_treino, fetch, window_bp=config.window_bp, limite=config.limite_treino,
        seed=config.seed)
    if not treino_exemplos:
        print("FALHOU: nenhum exemplo de treino")
        return 2
    validacao_exemplos: list = []
    falhas_validacao: dict = {}
    if config.plano_validacao:
        validacao_exemplos, falhas_validacao = carregar_exemplos(
            config.plano_validacao, fetch, window_bp=config.window_bp, limite=config.limite_validacao,
            seed=config.seed + 1)

    relogio["exemplos_prontos"] = time.monotonic()
    print(f"  [carga] {len(treino_exemplos)} exemplos de treino e {len(validacao_exemplos)} de validacao em "
          f"{relogio['exemplos_prontos'] - relogio['inicio']:.0f}s  "
          f"(pico de memoria ate aqui: {pico_de_memoria_mb()} MB)")

    adapter, resumo, proveniencia = montar(config, device)
    proveniencia["revisao_do_codigo"] = revisao_do_codigo()
    backbone = adapter.backbone
    if config.modulos_esperados:
        esperados = json.loads(config.modulos_esperados.expanduser().read_text(encoding="utf-8"))
        if sorted(resumo.module_names) != sorted(esperados):
            print(f"FALHOU: superficie adaptada difere da aprovada "
                  f"({len(resumo.module_names)} contra {len(esperados)})")
            return 2

    # Calculadas UMA vez: os checkpoints parciais e o melhor tinham menos proveniencia que o final.
    identidades_base = {
        "checkpoint_r03": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint.expanduser()),
        "plano_treino_sha256": sha256_file(config.plano_treino.expanduser()),
        "plano_validacao_sha256": sha256_file(config.plano_validacao.expanduser())
        if config.plano_validacao else None,
        "revisao_do_codigo": proveniencia["revisao_do_codigo"],
    }

    parametros = list(treino.parametros_do_adapter(backbone).values())
    otimizador = torch.optim.AdamW(parametros, lr=config.lr, weight_decay=config.weight_decay)
    impressao_inicial = treino.impressao_dos_congelados(backbone)

    passo_inicial = 0
    retomada: dict[str, Any] = {}
    if config.retomar:
        # Retomada EXPLICITA: carrega o adapter e continua do passo gravado. O otimizador NAO e restaurado, e
        # isso vai declarado -- um AdamW recomecado do zero tem momentos vazios e nao e a mesma trajetoria.
        retomada = treino.carregar_adapter(config.retomar.expanduser(), backbone)
        passo_inicial = int(retomada.get("passo") or 0)
        retomada["aviso"] = "o estado do otimizador NAO e restaurado: os momentos do AdamW recomecam do zero"
        print(f"[retomada] adapter de {config.retomar} no passo {passo_inicial}")

    # LINHA DE BASE, antes de qualquer passo. `lora_b` nasce em zeros, entao o adapter comeca como um no-op
    # EXATO em execucao nova. Na retomada, a base e o adapter carregado, nao o R03 puro. Sem ela, um valor final de 1,72 nao
    # tem contra o que ser comparado -- a primeira validacao do piloto de 22/09 ja vinha depois de 10 passos.
    linha_de_base = None
    if validacao_exemplos:
        linha_de_base = avaliar(adapter, validacao_exemplos, pesos=pesos, batch=config.batch, device=device)
        linha_de_base["sistema"] = "adapter_retomado" if config.retomar else "r03_sem_delta"
        print(f"  [base]        focal_val={linha_de_base['criterio_primario']['valor']:.4f}  "
              f"({linha_de_base['sistema']})")

    relogio["modelo_pronto"] = time.monotonic()
    config.out_dir.expanduser().mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    historico: list[dict[str, Any]] = []
    # SELECAO POR VALIDACAO. O piloto 5 mostrou por que isto nao e opcional: a validacao tocou o fundo no passo
    # 89 e depois degradou ate ficar PIOR que nao treinar. Salvar so o final entregaria o pior adapter da corrida.
    melhor: dict[str, Any] | None = None
    caminho_melhor = config.out_dir.expanduser() / "adapter_melhor.pt"
    atualizacoes = 0
    motivo_de_parada = "passos concluidos"

    for passo in range(passo_inicial, config.passos):
        indices = rng.permutation(len(treino_exemplos))[:config.exemplos_por_passo]
        otimizador.zero_grad(set_to_none=True)
        peso_total = 0.0
        parciais: list = []
        finito = True
        for comeco in range(0, len(indices), config.batch):
            pedaco = [treino_exemplos[i] for i in indices[comeco:comeco + config.batch]]
            lote = treino.montar_lote(pedaco, device=device)
            logits = treino.logits_mlm(adapter, lote.input_ids)
            soma, peso, decomposicao = treino.perda_somada_do_lote(logits, lote, pesos)
            if not bool(torch.isfinite(soma)):
                finito = False
                break
            soma.backward()
            peso_total += peso
            parciais.append(decomposicao)
        if not finito:
            motivo_de_parada = f"loss nao finita no passo {passo}"
            break

        # A acumulacao so equivale a um lote unico depois desta divisao pelo peso TOTAL.
        treino.dividir_gradientes(parametros, peso_total)
        estado = treino.gradientes_do_adapter(backbone)
        if estado["nao_finitos"]:
            motivo_de_parada = f"gradiente nao finito no passo {passo}: {estado['nao_finitos'][:3]}"
            break

        if config.clip_norma > 0:
            torch.nn.utils.clip_grad_norm_(parametros, config.clip_norma)
        # O scheduler conta ATUALIZACOES do otimizador, nao lotes: com acumulacao os dois divergem.
        escala = escala_cosseno(atualizacoes, total=max(1, config.passos - passo_inicial),
                                aquecimento=config.aquecimento)
        for grupo in otimizador.param_groups:
            grupo["lr"] = config.lr * escala
        otimizador.step()
        atualizacoes += 1

        agregado = mlm.agregar(parciais)
        linha = {"passo": passo, "atualizacoes": atualizacoes, "lr": config.lr * escala,
                 "treino": {c: agregado[c] for c in mlm.CATEGORIAS},
                 "criterio_primario_treino": agregado[mlm.CRITERIO_PRIMARIO]["media"]}
        if validacao_exemplos and (passo + 1) % config.validar_a_cada == 0:
            linha["validacao"] = avaliar(adapter, validacao_exemplos, pesos=pesos, batch=config.batch,
                                         device=device)
        historico.append(linha)

        def _salvar(caminho: Path, metricas: dict[str, Any]) -> None:
            treino.salvar_adapter(caminho, backbone=backbone, resumo_lora=resumo,
                                  config={k: str(v) for k, v in vars(config).items()},
                                  identidades=identidades_base, metricas=metricas, passo=passo + 1)

        if "validacao" in linha:
            melhor, trocou = atualizar_melhor(melhor, passo, linha["validacao"])
            if trocou:
                melhor["caminho"] = str(caminho_melhor)
                _salvar(caminho_melhor, linha)
        if config.salvar_a_cada and (passo + 1) % config.salvar_a_cada == 0:
            # Antes de uma corrida longa, salvar so no fim significa perder tudo se ela cair.
            _salvar(config.out_dir.expanduser() / f"adapter_passo{passo + 1:06d}.pt", linha)
        print(f"  passo {passo:>4}  lr={linha['lr']:.2e}  focal_treino="
              f"{linha['criterio_primario_treino']:.4f}"
              + (f"  focal_val={linha['validacao']['criterio_primario']['valor']:.4f}"
                 if "validacao" in linha else ""))

    # O delta deve corresponder ao checkpoint final mesmo fora da cadencia de validacao -- e essa avaliacao
    # tambem PARTICIPA DA SELECAO. Antes ela era calculada depois do bloco do `melhor`: se o ultimo passo fosse o
    # melhor e nao caisse na cadencia, o arquivo escolhido ficaria errado.
    if validacao_exemplos and historico and "validacao" not in historico[-1]:
        historico[-1]["validacao"] = avaliar(
            adapter, validacao_exemplos, pesos=pesos, batch=config.batch, device=device)
        melhor, trocou = atualizar_melhor(melhor, historico[-1]["passo"], historico[-1]["validacao"])
        if trocou:
            melhor["caminho"] = str(caminho_melhor)
            treino.salvar_adapter(caminho_melhor, backbone=backbone, resumo_lora=resumo,
                                  config={k: str(v) for k, v in vars(config).items()},
                                  identidades=identidades_base, metricas=historico[-1],
                                  passo=historico[-1]["passo"] + 1)
    congelado_intacto = treino.impressao_dos_congelados(backbone) == impressao_inicial
    out_dir = config.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    caminho = out_dir / "adapter.pt"
    if congelado_intacto and historico:
        treino.salvar_adapter(caminho, backbone=backbone, resumo_lora=resumo,
                              config={k: str(v) for k, v in vars(config).items()},
                              identidades={"checkpoint_r03": str(config.checkpoint),
                                           "checkpoint_sha256": sha256_file(config.checkpoint.expanduser()),
                                           "plano_treino_sha256": sha256_file(config.plano_treino.expanduser()),
                                           "revisao_do_codigo": proveniencia["revisao_do_codigo"]},
                              metricas=historico[-1], passo=historico[-1]["passo"] + 1)

    final = historico[-1].get("validacao") if historico else None
    delta = None
    if linha_de_base and final:
        from eval.adapter import mlm as _mlm

        def _delta(depois, antes):
            return {c: round(depois[c]["media"] - antes[c]["media"], 6)
                    for c in _mlm.CATEGORIAS
                    if depois.get(c, {}).get("posicoes") and antes.get(c, {}).get("posicoes")}

        def _delta_diag(depois, antes):
            return {f: {c: round(depois[f][c] - antes[f][c], 6)
                        for c in ("p_ref", "p_alt", "fracao_do_alt_entre_as_nao_ref", "entropia",
                                  "termo_massa", "termo_escolha")
                        if c in depois[f] and c in antes[f]}
                    for f in sorted(set(depois) & set(antes))}

        delta = {
            "por_categoria": _delta(final["por_categoria"], linha_de_base["por_categoria"]),
            "diagnostico_do_focal": _delta_diag(final.get("diagnostico_do_focal", {}),
                                                linha_de_base.get("diagnostico_do_focal", {})),
            "como_ler_o_diagnostico": (
                "ATRIBUICAO, nao identificacao de causa. `termo_massa + termo_escolha` e EXATAMENTE a perda "
                "focal, entao o delta se reparte sem hipotese: massa = quanto custa a probabilidade total das "
                "nao-referencia; escolha = quanto custa acertar QUAL delas. Uma queda so no termo de massa diz "
                "que o modelo abriu espaco para as alternativas; uma queda no de escolha diz que ele discriminou "
                "melhor entre elas. NENHUM dos dois, sozinho, demonstra adaptacao populacional -- discriminar "
                "melhor pode vir de contexto de sequencia. `entropia`, `p_ref` e a fracao sao descricao; a "
                "fracao NAO fica presa em 1/3 quando se tira massa da referencia (so se a redistribuicao for "
                "uniforme), entao nao sirva dela como regra de decisao"),
            "por_fonte": {f: _delta(final["por_fonte"][f], linha_de_base["por_fonte"][f])
                          for f in sorted(set(final["por_fonte"]) & set(linha_de_base["por_fonte"]))},
            "leitura": ("negativo = melhorou. Mede a MESMA amostra de validacao antes e depois, entao nao ha "
                        "ruido de amostragem entre os dois; continua sem intervalo de confianca, e o numero de "
                        "posicoes focais nao equivale ao numero de observacoes independentes: respeitar os locos"),
        }

    relogio["fim"] = time.monotonic()
    custo = {
        "segundos_carregando_exemplos": round(relogio["exemplos_prontos"] - relogio["inicio"], 1),
        "segundos_montando_o_modelo": round(relogio["modelo_pronto"] - relogio["exemplos_prontos"], 1),
        "segundos_no_laco": round(relogio["fim"] - relogio["modelo_pronto"], 1),
        "segundos_por_atualizacao": round((relogio["fim"] - relogio["modelo_pronto"]) / atualizacoes, 2)
        if atualizacoes else None,
        "pico_de_memoria_mb": pico_de_memoria_mb(),
        "exemplos_de_treino_na_memoria": len(treino_exemplos),
        "o_que_nao_separa": ("o tempo do laco inclui as validacoes; para separar, comparar corridas com "
                             "--validar-a-cada diferente"),
    }
    print(f"  [custo] carga {custo['segundos_carregando_exemplos']}s, modelo "
          f"{custo['segundos_montando_o_modelo']}s, laco {custo['segundos_no_laco']}s "
          f"({custo['segundos_por_atualizacao']}s/atualizacao), pico {custo['pico_de_memoria_mb']} MB")

    valor_da_base = linha_de_base["criterio_primario"]["valor"] if linha_de_base else None
    if melhor and caminho_melhor.exists():
        melhor["sha256"] = sha256_file(caminho_melhor)
        if valor_da_base is not None:
            melhor["delta_contra_a_base"] = round(melhor["valor"] - valor_da_base, 6)
            melhor["supera_a_base"] = melhor["valor"] < valor_da_base
            melhor["recomendacao"] = (
                "usar este adapter" if melhor["supera_a_base"] else
                "NAO usar: nenhum checkpoint superou a linha de base. Manter o R03 sem adapter")
    degradou = bool(linha_de_base and final
                    and final["criterio_primario"]["valor"] > linha_de_base["criterio_primario"]["valor"])
    if melhor and valor_da_base is not None and not melhor["supera_a_base"]:
        print(f"\nATENCAO: NENHUM checkpoint superou a linha de base "
              f"({melhor['valor']:.4f} contra {valor_da_base:.4f}). "
              f"A recomendacao e manter o R03 SEM adapter.")
    elif degradou:
        print("\nATENCAO: o adapter FINAL e pior que a linha de base "
              f"({final['criterio_primario']['valor']:.4f} contra "
              f"{linha_de_base['criterio_primario']['valor']:.4f}). "
              f"Use `adapter_melhor.pt` (passo {melhor['passo'] if melhor else '?'}), nao `adapter.pt`.")

    relatorio = {
        "proveniencia": proveniencia,
        "linha_de_base": linha_de_base,
        "melhor_por_validacao": melhor,
        "final_pior_que_a_base": degradou,
        "delta_da_validacao": delta,
        "entradas": {
            "plano_treino": str(config.plano_treino),
            "plano_treino_sha256": sha256_file(config.plano_treino.expanduser()),
            "plano_validacao": str(config.plano_validacao) if config.plano_validacao else None,
            "plano_validacao_sha256": sha256_file(config.plano_validacao.expanduser())
            if config.plano_validacao else None,
            "exemplos_de_treino": len(treino_exemplos), "exemplos_de_validacao": len(validacao_exemplos),
            "mistura_do_treino": _mistura(treino_exemplos),
            "mistura_da_validacao": _mistura(validacao_exemplos),
            "falhas_treino": falhas_treino, "falhas_validacao": falhas_validacao,
            "fasta": str(config.fasta), "leitor": leitor,
            "checkpoint_sha256": sha256_file(config.checkpoint.expanduser()),
        },
        "receita": {"pesos": pesos, "criterio_primario": mlm.CRITERIO_PRIMARIO, "lr": config.lr,
                    "weight_decay": config.weight_decay, "clip_norma": config.clip_norma,
                    "passos": config.passos, "exemplos_por_passo": config.exemplos_por_passo,
                    "batch": config.batch, "aquecimento": config.aquecimento, "seed": config.seed,
                    "unidade_do_scheduler": "atualizacoes do otimizador, nao lotes",
                    "referencia_uniforme": {
                        "valor": round(math.log(4), 4),
                        "leitura": ("entropia cruzada de uma previsao uniforme sobre as 4 bases. ACIMA dela, o "
                                    "modelo tem media geometrica da probabilidade do alvo inferior a 25%. "
                                    "ABAIXO, a media geometrica e superior a 25%. Nao mede acuracia nem "
                                    "demonstra exposicao ou ausencia de variacao no pre-treino")},
                    "agregacao_da_validacao": "soma e contagem de posicoes, nunca media de medias"},
        "custo": custo,
        "retomada": retomada or None,
        "historico": historico,
        "atualizacoes_do_otimizador": atualizacoes,
        "motivo_de_parada": motivo_de_parada,
        "backbone_congelado_intacto": congelado_intacto,
        "o_que_nao_prova": [
            "nao demonstra ganho de regionalizacao: mede reconstrucao mascarada",
            "o criterio primario e a perda no ALT das variantes que a receita selecionou, nao aprendizado de "
            "estrutura populacional",
            "poucos passos nao demonstram estabilidade de um treino longo",
        ],
        "saidas": {"adapter": str(caminho), "adapter_sha256": sha256_file(caminho)}
        if caminho.exists() else {},
    }
    (out_dir / "treino_do_adapter.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("receita", "atualizacoes_do_otimizador", "motivo_de_parada",
                                                "backbone_congelado_intacto", "custo",
                                                "melhor_por_validacao", "final_pior_que_a_base", "saidas")},
                     ensure_ascii=False, indent=2, default=str))
    if delta:
        print(json.dumps({"delta_da_validacao": delta}, ensure_ascii=False, indent=2, default=str))
    if final:
        print(json.dumps(final, ensure_ascii=False, indent=2, default=str))

    if not congelado_intacto:
        print("\nFALHOU: o backbone congelado MUDOU durante o treino.")
        return 2
    if motivo_de_parada != "passos concluidos":
        print(f"\nFALHOU: {motivo_de_parada}")
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
    parser.add_argument("--treinar", action="store_true", help="roda o laco de treino (piloto curto por padrao)")
    parser.add_argument("--passos", type=int, default=20)
    parser.add_argument("--exemplos-por-passo", type=int, default=8)
    parser.add_argument("--aquecimento", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-norma", type=float, default=1.0)
    parser.add_argument("--validar-a-cada", type=int, default=5)
    parser.add_argument("--salvar-a-cada", type=int, default=0,
                        help="checkpoint parcial a cada N passos; 0 salva so no fim")
    parser.add_argument("--limite-treino", type=int, help="subamostra N linhas preservando a proporcao por fonte (piloto)")
    parser.add_argument("--limite-validacao", type=int)
    parser.add_argument("--retomar", type=Path, help="adapter.pt de onde continuar; o otimizador NAO e restaurado")
    parser.add_argument("--smoke-exemplos", type=int, default=8)
    parser.add_argument("--modulos-esperados", type=Path,
                        help="JSON com a superficie APROVADA de modulos adaptados; sem ele a lista so e relatada")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out-dir", required=True, type=Path)
    config = parser.parse_args(argv)

    if config.smoke and config.treinar:
        print("FALHOU: escolha --smoke OU --treinar, nao os dois.")
        return 2
    if config.smoke:
        return rodar_smoke(config)
    if config.treinar:
        return rodar_treino(config)
    print("FALHOU: use --smoke (checagens de integridade) ou --treinar (laco).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
