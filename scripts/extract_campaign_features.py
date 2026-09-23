#!/usr/bin/env python3
"""G3: extrai as duas leituras candidatas de UM sistema (M0 ou MR) e grava um cache com identidade.

POR QUE CACHE
    Em M0 e em MR o backbone e congelado: dentro DESTE pipeline, a representacao de uma variante nao muda entre
    epocas, e extrair uma vez da o mesmo numero que refazer o forward a cada epoca, a uma fracao do custo. Isso
    NAO faz deste pipeline o mesmo do `eval/clinvar/train.py`: la a montagem do lote, a leitura e a cabeca sao
    outras. A campanha define o seu, e usa o mesmo em M0 e em MR.

O QUE ENTRA
    Os papeis de desenvolvimento declarados em `configs/campanha_r03_desenvolvimento.json`: `train` e
    `validation` do snapshot (o de `nenhum` cobre as tres politicas, que sao aninhadas) e o conjunto de selecao
    comum. O papel `test` (fold 0) e os estudos brasileiros sao RECUSADOS por `eval/campanha/recortes.py`.

O QUE SAI (em --out-dir)
    identidade.json      -- checkpoint, adapter, FASTA, hash de CONTEUDO da tabela, sha256 dos arquivos de codigo
                            que determinam os numeros, ambiente (torch, CUDA, GPU), janela e lote. Retomar com
                            QUALQUER diferenca (menos o commit do git) e recusado: um cache nao mistura dois objetos.
    fragmento_NNNNN.npz  -- variant_id, papel e uma matriz float32 por extracao, validada (formas e finitude) antes
                            de gravar e gravada por troca atomica: uma queda deixa um `.tmp`, que a retomada apaga.
    tabela.parquet       -- a tabela extraida, gravada UMA vez, na criacao; na retomada so e conferida.
    falhas.json          -- variantes cuja janela nao se construiu, com o motivo (se houver).
    manifesto.json       -- contagens, releitura dos fragmentos e completude, escrito no fim.

COMPLETO = TODA variante da tabela no cache. Estes artefatos foram auditados com zero falha de janela; uma falha
agora e problema de dado, e M0 e MR nao podem terminar com tabelas efetivas diferentes. Saida 2 se faltar algo.

UMA EXTRACAO POR CACHE: uma trava com o PID impede duas execucoes no mesmo --out-dir.

MODO --smoke
    Extrai poucas variantes e mede: custo por variante, determinismo (o mesmo lote duas vezes), dependencia da
    posicao no lote e, em MR, que o adapter esta ATIVO. Nao grava cache. As janelas conferidas sao so as da amostra.
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

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import (  # noqa: E402
    estado_do_cache,
    gravar_fragmento,
    identidade_do_codigo,
    ler_fragmentos,
    limpar_temporarios,
    proximo_indice,
    trava,
)
from eval.campanha.recortes import (  # noqa: E402
    PAPEIS_DE_DESENVOLVIMENTO,
    adapter_congelado,
    carregar_campanha,
    hash_da_tabela,
    hash_do_conteudo,
    resumo_da_tabela,
    tabela_de_extracao,
)
from eval.embedding_probe.windows import WindowError, build_window, focal_offset  # noqa: E402
from scripts.audit_variant_windows import abrir_fasta  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

FAMILIA = "lumina-r03"
#: A pesquisa mediu diferenca 0 exata; no smoke de 23/09 a dependencia da vizinhanca no lote foi 1,9e-6 (M0) e
#: 1,4e-6 (MR), e o mesmo lote duas vezes deu 0 exato. A tolerancia so evita reprovar por essa ordem de grandeza;
#: o que protege a comparacao e o protocolo de lote fixo, identico em M0 e MR. O valor medido e sempre impresso.
TOLERANCIA_NUMERICA = 1e-5
VERSAO_DO_EXTRATOR = "campanha_r03_extracao_v2"
PREFIXO_DO_CHECKPOINT = "f2983560"  # R03 best_checkpoint.pt, passo 71.000 (contrato)
#: Todo arquivo que determina um numero do cache. Mudar qualquer um invalida a retomada de um cache existente.
ARQUIVOS_QUE_DETERMINAM_AS_FEATURES = (
    "scripts/extract_campaign_features.py",
    "scripts/audit_variant_windows.py",
    "eval/campanha/recortes.py",
    "eval/campanha/layout.py",
    "eval/campanha/leituras.py",
    "eval/campanha/cache.py",
    "eval/embedding_probe/rich.py",
    "eval/embedding_probe/windows.py",
    "eval/clinvar/lora.py",
    "eval/clinvar/r03_adapter.py",
    "eval/clinvar/adapters.py",
    "eval/adapter/treino.py",
)


# ----------------------------------------------------------------------------------------------- identidade

def identidade(args: argparse.Namespace, *, tabela: pd.DataFrame, checkpoint_sha: str, adapter_sha: str | None,
               fasta_sha: str, revisao: str, codigo: dict[str, Any], ambiente: dict[str, Any]) -> dict[str, Any]:
    """O que torna duas extracoes o MESMO objeto. Tudo que muda um numero do cache esta aqui."""
    from eval.campanha.layout import EXTRACOES, RAIO_DO_CONTEXTO

    return {
        "versao_do_extrator": VERSAO_DO_EXTRATOR,
        "revisao_do_codigo": revisao,
        "codigo": codigo,
        "ambiente": ambiente,
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
                 "ultimo_lote": "completado com copias: o tamanho do forward nunca muda",
                 "ordem": "tabela ordenada por (papel, variant_id); fragmentos de --fragmento variantes"},
        "fragmento": args.fragmento,
        "precisao": {"dtype": "float32", "tf32": False},
        "tolerancia_numerica_do_smoke": TOLERANCIA_NUMERICA,
        "extracoes": {nome: {"blocos": list(blocos), "dims": dims} for nome, (blocos, dims) in EXTRACOES.items()},
        "raio_do_contexto_local": RAIO_DO_CONTEXTO,
        "tabela_sha256_composicao": hash_da_tabela(tabela),
        "tabela_sha256_conteudo": hash_do_conteudo(tabela),
        "papeis": sorted(tabela["papel"].unique().tolist()),
    }


def revisao_do_codigo() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "desconhecida"
    except Exception:  # noqa: BLE001
        return "desconhecida"


def ambiente_de_execucao(device: Any) -> dict[str, Any]:
    """Versoes e GPU: outro kernel da outra numerica, entao retomar noutro ambiente e recusado."""
    import platform

    import torch

    saida: dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__,
                             "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if getattr(device, "type", "") == "cuda":
        saida["gpu"] = torch.cuda.get_device_name(device)
    try:
        import mamba_ssm

        saida["mamba_ssm"] = getattr(mamba_ssm, "__version__", "sem __version__")
    except Exception:  # noqa: BLE001
        saida["mamba_ssm"] = None
    return saida


def codigo_da_extracao() -> dict[str, Any]:
    """sha256 dos arquivos do repositorio e do pacote `lumina` efetivamente importado."""
    import lumina

    return identidade_do_codigo(RAIZ, ARQUIVOS_QUE_DETERMINAM_AS_FEATURES,
                                {"lumina": Path(lumina.__file__).resolve().parent})


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


def janelas_da_tabela(tabela: pd.DataFrame, fetch, window_bp: int) -> tuple[list[tuple[int, Any]], list[dict]]:
    """(posicao na tabela, janela) das variantes que constroem, e cada falha com a variante e o motivo."""
    boas, falhas = [], []
    for posicao, linha in enumerate(tabela.itertuples(index=False)):
        try:
            boas.append((posicao, build_window(fetch, chrom=str(linha.chrom), pos_1based=int(linha.pos_1based),
                                               ref=str(linha.ref), alt=str(linha.alt), window_bp=window_bp)))
        except WindowError as exc:
            falhas.append({"variant_id": str(linha.variant_id), "motivo": exc.reason})
    return boas, falhas


def contagem(falhas: list[dict]) -> dict[str, int]:
    saida: dict[str, int] = {}
    for falha in falhas:
        saida[falha["motivo"]] = saida.get(falha["motivo"], 0) + 1
    return saida


# ----------------------------------------------------------------------------------------------- smoke

def rodar_smoke(args: argparse.Namespace, adapter: Any, tabela: pd.DataFrame, fetch) -> int:
    focal = focal_offset(args.window_bp)
    amostra = tabela.groupby("papel", sort=True).head(args.limite).reset_index(drop=True)
    janelas, falhas = janelas_da_tabela(amostra, fetch, args.window_bp)
    if any(f["motivo"] == "ref_mismatch" for f in falhas):
        print(f"FALHOU: ref_mismatch na amostra -- FASTA ou build errado: {falhas[:3]}")
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
           f"max |diferenca| = {diferenca:.3e} (tolerancia {TOLERANCIA_NUMERICA:.0e})")

    sozinha = extrair_lote(adapter, lote[1:2], variantes_por_lote=args.variantes_por_lote, focal=focal)
    posicao = max(float(np.abs(primeira[n][1] - sozinha[n][0]).max()) for n in primeira)
    checar("dependencia da vizinhanca no lote (mesmo tamanho de forward)", posicao <= TOLERANCIA_NUMERICA,
           f"max |diferenca| = {posicao:.3e} (tolerancia {TOLERANCIA_NUMERICA:.0e})")

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

    inicio = time.perf_counter()
    medidos = [j for _, j in janelas]
    for comeco in range(0, len(medidos), args.variantes_por_lote):
        extrair_lote(adapter, medidos[comeco:comeco + args.variantes_por_lote],
                     variantes_por_lote=args.variantes_por_lote, focal=focal)
    por_variante = (time.perf_counter() - inicio) / max(1, len(medidos))
    total_horas = por_variante * len(tabela) / 3600
    checar("custo medido", True, f"{por_variante:.4f} s/variante em {len(medidos)} variantes -> "
                                 f"~{total_horas:.1f} h para as {len(tabela):,} da tabela")

    relatorio = {"sistema": args.sistema, "variantes_por_lote": args.variantes_por_lote,
                 "janelas_conferidas": {"amostra": int(len(amostra)), "construidas": len(janelas),
                                        "falhas": contagem(falhas),
                                        "alcance": "so a amostra do smoke; a tabela inteira e conferida na extracao"},
                 "tolerancia_numerica": TOLERANCIA_NUMERICA,
                 "segundos_por_variante": por_variante, "horas_estimadas_para_a_tabela": total_horas,
                 "tabela": resumo_da_tabela(tabela), "checagens": checagens,
                 "passou": all(c["ok"] for c in checagens)}
    destino = args.out_dir.expanduser()
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "smoke_da_extracao.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("sistema", "variantes_por_lote", "janelas_conferidas",
                                                "horas_estimadas_para_a_tabela", "tabela", "passou")},
                     ensure_ascii=False, indent=2))
    return 0 if relatorio["passou"] else 2


# ----------------------------------------------------------------------------------------------- extracao

def conferir_identidade(destino: Path, ident: dict[str, Any]) -> str | None:
    """Grava a identidade num cache novo; num cache existente, recusa QUALQUER diferenca menos o commit do git
    (que muda com arquivos que nao determinam numero -- os que determinam estao em `codigo`)."""
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


def conferir_tabela_gravada(destino: Path, tabela: pd.DataFrame, *, novo: bool) -> str | None:
    """Na criacao, grava a tabela UMA vez. Na retomada, so confere: nunca sobrescreve a tabela do cache."""
    arquivo = destino / "tabela.parquet"
    if novo:
        tabela.to_parquet(arquivo, index=False)
        return None
    if not arquivo.exists():
        return f"{arquivo} sumiu de um cache existente"
    if hash_do_conteudo(pd.read_parquet(arquivo)) != hash_do_conteudo(tabela):
        return f"{arquivo} nao confere com a tabela atual (hash de conteudo diferente)"
    return None


def rodar_extracao(args: argparse.Namespace, adapter: Any, tabela: pd.DataFrame, fetch,
                   ident: dict[str, Any]) -> int:
    destino = args.out_dir.expanduser()
    destino.mkdir(parents=True, exist_ok=True)
    with trava(destino):
        removidos = limpar_temporarios(destino)
        if removidos:
            print(f"[retomada] {len(removidos)} gravacoes incompletas removidas: {removidos}")
        novo = not (destino / "identidade.json").exists()
        problema = conferir_identidade(destino, ident) or conferir_tabela_gravada(destino, tabela, novo=novo)
        if problema:
            print(f"FALHOU: {problema}")
            return 2
        feitas, problemas = ler_fragmentos(destino, tabela)
        if problemas:
            print("FALHOU: fragmentos existentes nao passam na validacao; nada foi gravado. Inspecione ou remova:")
            for linha in problemas[:10]:
                print(f"  - {linha}")
            return 2

        pendentes = tabela[~tabela["variant_id"].astype(str).isin(feitas)].reset_index(drop=True)
        print(f"[extracao] {len(tabela):,} na tabela, {len(feitas):,} ja no cache, {len(pendentes):,} pendentes")
        focal = focal_offset(args.window_bp)
        falhas: list[dict] = []
        inicio = time.perf_counter()
        feitas_agora = 0
        for comeco in range(0, len(pendentes), args.fragmento):
            pedaco = pendentes.iloc[comeco:comeco + args.fragmento]
            janelas, falhas_do_pedaco = janelas_da_tabela(pedaco, fetch, args.window_bp)
            falhas += falhas_do_pedaco
            if any(f["motivo"] == "ref_mismatch" for f in falhas_do_pedaco):
                print(f"FALHOU: ref_mismatch -- FASTA ou build errado: {falhas_do_pedaco[:3]}")
                return 2
            if not janelas:
                continue
            blocos: dict[str, list[np.ndarray]] = {}
            for i in range(0, len(janelas), args.variantes_por_lote):
                grupo = janelas[i:i + args.variantes_por_lote]
                for nome, matriz in extrair_lote(adapter, [j for _, j in grupo],
                                                 variantes_por_lote=args.variantes_por_lote, focal=focal).items():
                    blocos.setdefault(nome, []).append(matriz)
            posicoes = [p for p, _ in janelas]
            indice = proximo_indice(destino)
            try:
                gravar_fragmento(destino, indice,
                                 variant_id=pedaco["variant_id"].to_numpy()[posicoes].astype(str),
                                 papel=pedaco["papel"].to_numpy()[posicoes].astype(str),
                                 matrizes={nome: np.concatenate(partes) for nome, partes in blocos.items()})
            except (ValueError, FileExistsError) as exc:
                print(f"FALHOU: {exc}")
                return 2
            feitas_agora += len(janelas)
            taxa = (time.perf_counter() - inicio) / max(1, feitas_agora)
            print(f"  fragmento {indice:05d}: {feitas_agora:,}/{len(pendentes):,}  ({taxa:.4f} s/variante, "
                  f"faltam ~{taxa * (len(pendentes) - comeco - len(pedaco)) / 3600:.1f} h)")

        if falhas:
            (destino / "falhas.json").write_text(json.dumps(falhas, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
        feitas, problemas = ler_fragmentos(destino, tabela)   # releitura completa do que ficou em disco
        estado = estado_do_cache(tabela, feitas)
        completo = estado["completo"] and not problemas
        manifesto = {
            "identidade": ident,
            **estado,
            "completo": completo,
            "falhas_de_janela_desta_execucao": contagem(falhas),
            "problemas_na_releitura": problemas[:20],
            "segundos_desta_execucao": round(time.perf_counter() - inicio, 1),
            "tabela": resumo_da_tabela(tabela),
        }
        (destino / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        print(json.dumps({k: manifesto[k] for k in ("variantes_na_tabela", "variantes_no_cache", "faltando",
                                                     "falhas_de_janela_desta_execucao", "completo")},
                         ensure_ascii=False))
        if not completo:
            print("FALHOU: o cache NAO cobre a tabela inteira. Nada segue com tabela efetiva diferente entre sistemas"
                  + ("; veja falhas.json" if falhas else ""))
            return 2
        return 0


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
                       fasta_sha=sha256_file(args.fasta.expanduser()), revisao=revisao_do_codigo(),
                       codigo=codigo_da_extracao(), ambiente=ambiente_de_execucao(device))
    return rodar_extracao(args, adapter, tabela, fetch, ident)


if __name__ == "__main__":
    sys.exit(main())
