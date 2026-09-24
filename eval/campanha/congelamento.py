"""Congelamento de um adapter de semente pela regra DECLARADA: confere a corrida e monta a entrada de
`adapters_congelados`. Sem torch: as regras recebem o relatorio e, quando preciso, a configuracao lida do checkpoint.

A regra (`adapter_do_mr.checkpoint` na declaracao): o `adapter_melhor.pt` da semente, o de menor `focal_alt` na
validacao do adapter; se nenhum checkpoint supera a base, nao ha MR com esta receita para a semente. A campanha
exige tambem que as tres sementes tenham a MESMA receita e o MESMO orcamento, o MESMO recorte de validacao e os
MESMOS planos: e isso que torna as tres execucoes replicas da mesma receita. A mesma conferencia serve ao G6.
"""
from __future__ import annotations

import ast
import math
from typing import Any

CHAVES_DA_RECEITA = ("lr", "passos", "exemplos_por_passo", "batch", "aquecimento", "weight_decay", "clip_norma",
                     "limite_validacao", "seed_da_validacao", "validar_a_cada", "backbone_em_eval")
PELA_REGRA = "adapter_do_mr.checkpoint: menor focal_alt na validacao do adapter; nenhuma metrica clinica consultada"


def _igual(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _valor(texto: Any) -> Any:
    """Os argumentos da linha de comando ficam gravados como texto no checkpoint ("5e-06", "True", "None")."""
    try:
        return ast.literal_eval(str(texto))
    except (ValueError, SyntaxError):
        return texto


def receita_da_configuracao(config: dict[str, Any]) -> dict[str, Any]:
    """A receita no formato da declaracao, a partir dos argumentos gravados no checkpoint. Serve as corridas
    anteriores ao registro completo da receita no relatorio (a a_1). Sem `--seed-da-validacao` gravado, a corrida e
    anterior a flag e a subamostra usou `seed + 1`."""
    c = {chave: _valor(valor) for chave, valor in config.items()}
    receita = {chave: c[chave] for chave in ("lr", "passos", "exemplos_por_passo", "batch", "aquecimento",
                                             "weight_decay", "clip_norma", "limite_validacao", "validar_a_cada",
                                             "backbone_em_eval") if chave in c}
    semente = c.get("seed_da_validacao")
    receita["seed_da_validacao"] = semente if semente is not None else (c["seed"] + 1 if "seed" in c else None)
    receita["pesos"] = {"focal_alt": c.get("peso_focal"), "contexto_da_variante": c.get("peso_contexto"),
                        "referencia": c.get("peso_referencia")}
    receita["lora"] = {"rank": c.get("lora_rank"), "alpha": c.get("lora_alpha"),
                       "rslora": (not c["sem_rslora"]) if "sem_rslora" in c else None}
    return receita


def diferencas_da_receita(declarada: dict[str, Any], da_corrida: dict[str, Any]) -> list[str]:
    """O que difere da receita declarada. Chave declarada e ausente na corrida e diferenca, nao silencio."""
    problemas = []
    for chave in CHAVES_DA_RECEITA:
        if chave not in declarada:
            continue
        if chave not in da_corrida:
            problemas.append(f"receita: {chave} ausente na corrida")
        elif not _igual(declarada[chave], da_corrida[chave]):
            problemas.append(f"receita: {chave} {da_corrida[chave]!r} != {declarada[chave]!r}")
    pesos_d, pesos_c = declarada.get("pesos") or {}, da_corrida.get("pesos") or {}
    if set(pesos_d) != set(pesos_c) or any(not _igual(pesos_d[k], pesos_c[k]) for k in pesos_d):
        problemas.append(f"receita: pesos {pesos_c} != {pesos_d}")
    lora_d, lora_c = declarada.get("lora") or {}, da_corrida.get("lora") or {}
    for chave in ("rank", "alpha", "rslora"):
        if chave in lora_d and not _igual(lora_d[chave], lora_c.get(chave)):
            problemas.append(f"receita: lora.{chave} {lora_c.get(chave)!r} != {lora_d[chave]!r}")
    return problemas


def conferir_corrida(relatorio: dict[str, Any], campanha: dict[str, Any], *, semente: int, sha256_do_arquivo: str,
                     recorte: dict[str, Any], prefixo_do_checkpoint: str,
                     receita_do_checkpoint: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None,
                                                                                list[str]]:
    """(entrada de `adapters_congelados`, problemas). Entrada so sai sem problema algum.

    `recorte` e o do `detalhe_da_validacao.json` da propria corrida; `receita_do_checkpoint` completa o que o
    relatorio nao registrou (corridas anteriores ao registro completo), sem sobrescrever o que ele registrou.
    """
    problemas: list[str] = []
    declarada = campanha["adapter_do_mr"]["receita"]
    if semente not in campanha["sementes"]["adapter"]:
        problemas.append(f"semente {semente} fora das declaradas {campanha['sementes']['adapter']}")
    receita = {**(receita_do_checkpoint or {}), **(relatorio.get("receita") or {})}
    if receita.get("seed") not in (None, semente):
        problemas.append(f"a corrida usou a semente {receita.get('seed')}, nao {semente}")
    problemas += diferencas_da_receita(declarada, receita)

    if relatorio.get("motivo_de_parada") != "passos concluidos":
        problemas.append(f"parada: {relatorio.get('motivo_de_parada')!r}")
    if relatorio.get("backbone_congelado_intacto") is not True:
        problemas.append("backbone congelado NAO ficou intacto")
    if relatorio.get("atualizacoes_do_otimizador") != declarada["passos"]:
        problemas.append(f"{relatorio.get('atualizacoes_do_otimizador')} atualizacoes, a receita declara "
                         f"{declarada['passos']}")

    entradas = relatorio.get("entradas") or {}
    prefixos = campanha["adapter_do_mr"]["planos_sha256_prefixo"]
    for papel, chave in (("treino", "plano_treino_sha256"), ("validacao", "plano_validacao_sha256")):
        if not str(entradas.get(chave) or "").startswith(prefixos[papel]):
            problemas.append(f"{chave} {str(entradas.get(chave))[:12]} nao e o plano declarado ({prefixos[papel]})")
    checkpoint = str(entradas.get("checkpoint_sha256") or "")
    if not checkpoint.startswith(prefixo_do_checkpoint):
        problemas.append(f"checkpoint {checkpoint[:12]} nao e o R03 ({prefixo_do_checkpoint})")
    for chave in ("falhas_treino", "falhas_validacao"):
        if entradas.get(chave):
            problemas.append(f"{chave}: {entradas[chave]}")
    registrado = entradas.get("recorte_da_validacao")
    if registrado and registrado.get("sha256") != recorte["sha256"]:
        problemas.append("o recorte registrado no relatorio nao e o do detalhe da validacao")
    if int(recorte.get("janelas", -1)) != int(declarada["limite_validacao"]):
        problemas.append(f"recorte com {recorte.get('janelas')} janelas, a receita declara "
                         f"{declarada['limite_validacao']}")

    melhor = relatorio.get("melhor_por_validacao") or {}
    if melhor.get("supera_a_base") is not True:
        problemas.append("nenhum checkpoint supera a base: pela regra, nao ha MR com esta receita para a semente")
    if melhor.get("sha256") != sha256_do_arquivo:
        problemas.append("o sha256 de adapter_melhor.pt nao e o registrado no relatorio")
    if problemas:
        return None, problemas

    base = float(relatorio["linha_de_base"]["criterio_primario"]["valor"])
    return {
        "sha256": sha256_do_arquivo,
        "passo": int(melhor["passo"]),
        "focal_alt_validacao": {"base": round(base, 4), "adapter": round(float(melhor["valor"]), 4),
                                "delta": round(float(melhor["valor"]) - base, 4)},
        "supera_a_base": True,
        "pela_regra": PELA_REGRA,
        "recorte_da_validacao_sha256": recorte["sha256"],
        "planos_sha256": {"treino": entradas["plano_treino_sha256"], "validacao": entradas["plano_validacao_sha256"]},
        "checkpoint_sha256": entradas["checkpoint_sha256"],
    }, []
