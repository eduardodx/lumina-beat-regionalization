#!/usr/bin/env python3
"""Resumo curto de `treino_do_adapter.json`, para colar na conversa.

O log de uma corrida longa tem uma linha por passo (3.000 na corrida de desenvolvimento) e o relatorio JSON guarda o
historico inteiro: nenhum dos dois cabe numa mensagem. Este script le o relatorio e imprime o que se le de uma
corrida: receita, entradas, custo, a CURVA de validacao, o melhor checkpoint, os deltas e o bootstrap por loco.

Os intervalos do bootstrap saem rotulados como EXPLORATORIOS: sao condicionais ao modelo escolhido na mesma
validacao e nao incluem variacao entre sementes.

Uso:
    python3 scripts/resumir_treino_do_adapter.py ~/artifacts/redesenho/g4_corrida2
    python3 scripts/resumir_treino_do_adapter.py ~/artifacts/redesenho/g4_corrida2/treino_do_adapter.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FOCAL = "focal_alt"
REFERENCIA = "referencia"
FONTES = ("abraom", "global")
ORDEM = "alt_em_primeiro_entre_nao_ref"


def _num(valor: Any, casas: int = 4) -> str:
    if valor is None:
        return "-"
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        return f"{valor:.{casas}f}"
    return str(valor)


def _sinal(valor: Any, casas: int = 4) -> str:
    return f"{valor:+.{casas}f}" if isinstance(valor, (int, float)) and not isinstance(valor, bool) else "-"


def _media(bloco: dict[str, Any] | None, categoria: str) -> float | None:
    return ((bloco or {}).get(categoria) or {}).get("media")


def _linha_da_curva(rotulo: str, validacao: dict[str, Any]) -> str:
    por_fonte = validacao.get("por_fonte") or {}
    diagnostico = validacao.get("diagnostico_do_focal") or {}
    focal = (validacao.get("criterio_primario") or {}).get("valor")
    colunas = [f"{rotulo:>6}", f"{_num(focal)}"]
    colunas += [_num(_media(por_fonte.get(fonte), FOCAL)) for fonte in FONTES]
    colunas.append(_num(_media(validacao.get("por_categoria"), REFERENCIA)))
    ordem = [(diagnostico.get(fonte) or {}).get(ORDEM) for fonte in FONTES]
    colunas.append("/".join(_num(valor, 3) for valor in ordem))
    return "  ".join(colunas)


def _faixa(bloco: dict[str, Any], chave_do_ponto: str = "delta") -> str:
    return (f"{_sinal(bloco.get(chave_do_ponto))} [{_sinal(bloco.get('p2_5'))}; "
            f"{_sinal(bloco.get('p97_5'))}]")


def _bootstrap(rotulo: str, bloco: Any) -> list[str]:
    if bloco is None:
        return [f"{rotulo}: ausente (relatorio anterior ao bootstrap)"]
    if isinstance(bloco, str):
        return [f"{rotulo}: {bloco}"]
    if "indisponivel" in bloco:
        extra = f" -- {bloco['problemas'][:3]}" if bloco.get("problemas") else ""
        return [f"{rotulo}: INDISPONIVEL ({bloco['indisponivel']}){extra}"]
    linhas = [f"{rotulo} -- IC 95% EXPLORATORIO, reamostrando {bloco.get('unidade_de_reamostragem', '?')} "
              f"({bloco.get('replicas', '?')} replicas)",
              f"  locos {bloco.get('locos')} | janelas {bloco.get('janelas')} | janelas por fonte "
              f"{bloco.get('janelas_por_fonte')} | locos por fonte {bloco.get('locos_por_fonte')} | locos com "
              f"mais de uma fonte {bloco.get('locos_com_mais_de_uma_fonte')}"]
    for fonte, campos in (bloco.get("por_fonte") or {}).items():
        linhas.append(f"  {fonte:<7} " + " | ".join(f"{campo} {_faixa(faixa)}" for campo, faixa in campos.items()))
    diferenca = bloco.get("abraom_menos_global")
    if diferenca:
        linhas.append("  abraom-global " + " | ".join(
            f"{campo} {_faixa(faixa, 'diferenca')}" for campo, faixa in diferenca.items()))
    return linhas


def _recorte(entradas: dict[str, Any], lido_do_detalhe: dict[str, Any] | None) -> str:
    registrado = entradas.get("recorte_da_validacao")
    recorte = registrado or lido_do_detalhe
    if not recorte:
        return "recorte nao disponivel (sem registro e sem detalhe_da_validacao.json)"
    origem = "" if registrado else ", lido do detalhe_da_validacao.json"
    confere = " | IDENTICO ao da referencia" if (registrado or {}).get("confere_com") else ""
    return f"recorte {recorte['sha256'][:16]} ({recorte['janelas']} janelas{origem}){confere}"


def resumir(relatorio: dict[str, Any], recorte_lido: dict[str, Any] | None = None) -> list[str]:
    """Linhas do resumo. Funcao pura: recebe o relatorio ja lido (e, para corridas anteriores ao registro do
    recorte, o recorte calculado do detalhe da validacao)."""
    receita = relatorio.get("receita") or {}
    entradas = relatorio.get("entradas") or {}
    custo = relatorio.get("custo") or {}
    semente_da_validacao = receita.get("seed_da_validacao",
                                       "nao registrada (antes da flag era seed + 1)")
    saida = [
        "== receita ==",
        f"  lr {receita.get('lr')} | passos {receita.get('passos')} | exemplos/passo "
        f"{receita.get('exemplos_por_passo')} | batch {receita.get('batch')} | seed {receita.get('seed')} | "
        f"pesos {receita.get('pesos')}",
        "== entradas ==",
        f"  treino {entradas.get('exemplos_de_treino')} {((entradas.get('mistura_do_treino') or {}).get('fracao'))}"
        f" | validacao {entradas.get('exemplos_de_validacao')} "
        f"{((entradas.get('mistura_da_validacao') or {}).get('fracao'))}",
        f"  validacao: semente da subamostra {semente_da_validacao} | {_recorte(entradas, recorte_lido)}",
        f"  falhas treino {entradas.get('falhas_treino')} | falhas validacao {entradas.get('falhas_validacao')}",
        f"  plano_treino {str(entradas.get('plano_treino_sha256'))[:12]} | plano_validacao "
        f"{str(entradas.get('plano_validacao_sha256'))[:12]} | checkpoint {str(entradas.get('checkpoint_sha256'))[:12]}",
        "== custo e parada ==",
        f"  laco {custo.get('segundos_no_laco')} s ({custo.get('segundos_por_atualizacao')} s/atualizacao) | pico "
        f"{custo.get('pico_de_memoria_mb')} MB | atualizacoes {relatorio.get('atualizacoes_do_otimizador')} | "
        f"parada: {relatorio.get('motivo_de_parada')} | backbone intacto: {relatorio.get('backbone_congelado_intacto')}",
        "== curva de validacao (focal_alt: menor = melhor; alt 1o = fracao em que o ALT lidera as nao-ref) ==",
        "  passo   focal  abraom  global     ref  alt 1o abraom/global",
    ]
    base = relatorio.get("linha_de_base")
    if base:
        saida.append("  " + _linha_da_curva("base", base))
    for linha in relatorio.get("historico") or []:
        if "validacao" in linha:
            saida.append("  " + _linha_da_curva(str(linha.get("passo")), linha["validacao"]))

    melhor = relatorio.get("melhor_por_validacao") or {}
    saida += ["== melhor por validacao ==",
              f"  passo {melhor.get('passo')} | {_num(melhor.get('valor'))} | delta contra a base "
              f"{_sinal(melhor.get('delta_contra_a_base'))} | {melhor.get('recomendacao')} | sha "
              f"{str(melhor.get('sha256'))[:12]}"]
    if relatorio.get("final_pior_que_a_base"):
        saida.append("  ATENCAO: o adapter FINAL e pior que a base")

    delta = relatorio.get("delta_da_validacao") or {}
    if delta:
        saida.append("== delta do FINAL contra a base (negativo = melhorou) ==")
        saida.append("  por categoria: " + " | ".join(
            f"{c} {_sinal(v)}" for c, v in (delta.get("por_categoria") or {}).items()))
        for fonte, categorias in (delta.get("por_fonte") or {}).items():
            saida.append(f"  {fonte:<7} " + " | ".join(f"{c} {_sinal(v)}" for c, v in categorias.items()))
        saida.append("  diagnostico do focal (descricao, nao portao):")
        for fonte, metricas in (delta.get("diagnostico_do_focal") or {}).items():
            saida.append(f"    {fonte:<7} " + " | ".join(f"{m} {_sinal(v)}" for m, v in metricas.items()))
        saida += _bootstrap("bootstrap do final", delta.get("bootstrap_por_loco"))
        if "bootstrap_do_melhor" in delta:
            saida += _bootstrap("bootstrap do melhor", delta.get("bootstrap_do_melhor"))
    arquivos = relatorio.get("saidas") or {}
    saida += ["== saidas ==", f"  {arquivos.get('adapter')} sha {str(arquivos.get('adapter_sha256'))[:12]}"]
    return saida


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("caminho", type=Path, help="out-dir da corrida ou o proprio treino_do_adapter.json")
    args = parser.parse_args(argv)
    caminho = args.caminho.expanduser()
    if caminho.is_dir():
        caminho = caminho / "treino_do_adapter.json"
    if not caminho.exists():
        print(f"FALHOU: {caminho} nao existe. Se a corrida caiu antes do fim, o relatorio nao foi escrito: "
              f"veja o log e os adapter_passo*.pt")
        return 2
    relatorio = json.loads(caminho.read_text(encoding="utf-8"))
    recorte_lido = None
    detalhe = caminho.parent / "detalhe_da_validacao.json"
    if not (relatorio.get("entradas") or {}).get("recorte_da_validacao") and detalhe.exists():
        from scripts.train_population_adapter import recorte_de_referencia

        recorte_lido = recorte_de_referencia(detalhe)
    print("\n".join(resumir(relatorio, recorte_lido)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
