"""Congelamento de um adapter de semente pela regra DECLARADA: confere a corrida contra a declaracao e monta a
entrada de `adapters_congelados`. Sem torch: as regras recebem o relatorio, a configuracao lida do checkpoint e o que
o script recalculou dos arquivos.

A regra (`adapter_do_mr.checkpoint`): o `adapter_melhor.pt` da semente, o de menor `focal_alt` na validacao do
adapter; se nenhum checkpoint supera a base, nao ha MR com esta receita para a semente. As tres sementes tem de ser
replicas da MESMA receita e orcamento, com o MESMO recorte de validacao, os MESMOS planos e o MESMO R03.

Nada aqui confia no que a corrida declarou sobre si mesma (revisao de 24/09): a melhora e RECALCULADA do historico,
a receita e conferida nas DUAS fontes (relatorio e argumentos gravados no checkpoint), que tambem tem de concordar
entre si, e planos, R03 e recorte sao comparados com uma referencia EXTERNA fixada na declaracao -- o recorte
esperado e re-sorteado do plano declarado, nao lido de outra corrida.
"""
from __future__ import annotations

import ast
import math
from typing import Any, Callable

#: Chaves escalares da receita declarada; `pesos` e `lora` sao conferidos a parte.
CHAVES_DA_RECEITA = ("lr", "passos", "exemplos_por_passo", "batch", "aquecimento", "weight_decay", "clip_norma",
                     "limite_validacao", "seed_da_validacao", "validar_a_cada", "backbone_em_eval", "window_bp",
                     "limite_treino", "retomar", "model_version", "modulos_esperados")
CHAVES_DO_LORA = ("rank", "alpha", "dropout", "rslora")
PELA_REGRA = "adapter_do_mr.checkpoint: menor focal_alt na validacao do adapter; nenhuma metrica clinica consultada"
_AUSENTE = object()


def _igual(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
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
    """A receita no formato da declaracao, a partir dos argumentos gravados no checkpoint. Sem `--seed-da-validacao`
    gravado, a corrida e anterior a flag e a subamostra usou `seed + 1` (caso da a_1)."""
    c = {chave: _valor(valor) for chave, valor in config.items()}
    receita: dict[str, Any] = {}
    for chave in ("lr", "passos", "exemplos_por_passo", "batch", "aquecimento", "weight_decay", "clip_norma",
                  "limite_validacao", "validar_a_cada", "backbone_em_eval", "window_bp", "limite_treino",
                  "retomar", "model_version", "modulos_esperados", "seed"):
        if chave in c:
            receita[chave] = c[chave]
    if "seed_da_validacao" in c and c["seed_da_validacao"] is not None:
        receita["seed_da_validacao"] = c["seed_da_validacao"]
    elif "seed" in c:
        receita["seed_da_validacao"] = c["seed"] + 1
    if {"peso_focal", "peso_contexto", "peso_referencia"} <= set(c):
        receita["pesos"] = {"focal_alt": c["peso_focal"], "contexto_da_variante": c["peso_contexto"],
                            "referencia": c["peso_referencia"]}
    lora = {nome: c[chave] for nome, chave in (("rank", "lora_rank"), ("alpha", "lora_alpha"),
                                               ("dropout", "lora_dropout")) if chave in c}
    if "sem_rslora" in c:
        lora["rslora"] = not c["sem_rslora"]
    if lora:
        receita["lora"] = lora
    return receita


def _achatar(receita: dict[str, Any]) -> dict[str, Any]:
    plana = {chave: receita[chave] for chave in CHAVES_DA_RECEITA + ("seed",) if chave in receita}
    for nome, valor in (receita.get("pesos") or {}).items():
        plana[f"pesos.{nome}"] = valor
    for nome, valor in (receita.get("lora") or {}).items():
        if nome in CHAVES_DO_LORA:
            plana[f"lora.{nome}"] = valor
    return plana


def diferencas_da_receita(declarada: dict[str, Any], fontes: dict[str, dict[str, Any]]) -> list[str]:
    """Cada chave declarada tem de estar em ao menos uma fonte e igual ao declarado em TODAS as que a trazem; duas
    fontes que discordam entre si sao problema mesmo que uma delas bata com a declaracao."""
    problemas = []
    esperado = _achatar(declarada)
    planas = {nome: _achatar(receita) for nome, receita in fontes.items()}
    for chave, valor in esperado.items():
        presentes = {nome: plana[chave] for nome, plana in planas.items() if chave in plana}
        if not presentes:
            problemas.append(f"receita: {chave} ausente no relatorio e no checkpoint")
            continue
        for nome, obtido in presentes.items():
            if not _igual(valor, obtido):
                problemas.append(f"receita: {chave} = {obtido!r} no {nome}, declarado {valor!r}")
    comuns = set.intersection(*(set(p) for p in planas.values())) if len(planas) > 1 else set()
    for chave in sorted(comuns - set(esperado)):
        valores = [plana[chave] for plana in planas.values()]
        if any(not _igual(valores[0], v) for v in valores[1:]):
            problemas.append(f"receita: {chave} diverge entre relatorio e checkpoint ({valores})")
    return problemas


def melhor_do_historico(relatorio: dict[str, Any]) -> tuple[int, float] | None:
    """(passo, valor) do menor `focal_alt` entre TODAS as validacoes do historico, com a regra do runner: empate
    fica com a primeira. None sem validacao."""
    melhor: tuple[int, float] | None = None
    for linha in relatorio.get("historico") or []:
        validacao = linha.get("validacao")
        if not validacao:
            continue
        valor = float(validacao["criterio_primario"]["valor"])
        if melhor is None or valor < melhor[1]:
            melhor = (int(linha["passo"]), valor)
    return melhor


def comparar_estados(a: dict[str, Any], chaves_a: list[str], b: dict[str, Any], chaves_b: list[str],
                     iguais: Callable[[Any, Any], bool]) -> dict[str, Any]:
    """Compara os estados do adapter de dois checkpoints. Estado ausente ou vazio, ou chaves que nao sao as
    declaradas no proprio checkpoint, NUNCA contam como iguais."""
    if not a or not b:
        return {"iguais": False, "motivo": "estado do adapter ausente ou vazio"}
    if sorted(a) != sorted(chaves_a) or sorted(b) != sorted(chaves_b):
        return {"iguais": False, "motivo": "chaves do estado diferentes das declaradas no checkpoint"}
    if set(a) != set(b):
        return {"iguais": False, "motivo": "conjuntos de chaves diferentes"}
    diferentes = sorted(k for k in a if not iguais(a[k], b[k]))
    return {"iguais": not diferentes, "chaves": len(a), "diferentes": diferentes[:5]}


def conferir_corrida(relatorio: dict[str, Any], campanha: dict[str, Any], *, semente: int, sha256_do_arquivo: str,
                     recorte: dict[str, Any], recorte_esperado: dict[str, Any],
                     sha256_dos_planos: dict[str, str],
                     receita_do_checkpoint: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, list[str]]:
    """(entrada de `adapters_congelados`, problemas). Entrada so sai sem problema algum.

    `recorte`: o do `detalhe_da_validacao.json` da propria corrida. `recorte_esperado`: re-sorteado do plano de
    validacao declarado, com a semente e o limite declarados. `sha256_dos_planos`: calculado agora dos arquivos que a
    corrida leu (`treino`, `validacao`).
    """
    problemas: list[str] = []
    adapter = campanha["adapter_do_mr"]
    declarada, referencia = adapter["receita"], adapter["referencia"]
    if semente not in campanha["sementes"]["adapter"]:
        problemas.append(f"semente {semente} fora das declaradas {campanha['sementes']['adapter']}")
    fontes = {"relatorio": relatorio.get("receita") or {}}
    if receita_do_checkpoint:
        fontes["checkpoint"] = receita_do_checkpoint
    for nome, receita in fontes.items():
        if receita.get("seed", semente) != semente:
            problemas.append(f"o {nome} diz semente {receita.get('seed')}, nao {semente}")
    problemas += diferencas_da_receita(declarada, fontes)

    if relatorio.get("motivo_de_parada") != "passos concluidos":
        problemas.append(f"parada: {relatorio.get('motivo_de_parada')!r}")
    if relatorio.get("backbone_congelado_intacto") is not True:
        problemas.append("backbone congelado NAO ficou intacto")
    if relatorio.get("atualizacoes_do_otimizador") != declarada["passos"]:
        problemas.append(f"{relatorio.get('atualizacoes_do_otimizador')} atualizacoes, a receita declara "
                         f"{declarada['passos']}")
    if relatorio.get("retomada", _AUSENTE) is not None:
        problemas.append("a corrida foi retomada (ou nao registrou): outra trajetoria, nao e replica da receita")

    entradas = relatorio.get("entradas") or {}
    for papel in ("treino", "validacao"):
        chave = f"plano_{papel}_sha256"
        declarado = referencia[chave][:12]
        if entradas.get(chave) != referencia[chave]:
            problemas.append(f"{chave} no relatorio {str(entradas.get(chave))[:12]} != declarado {declarado}")
        if sha256_dos_planos.get(papel) != referencia[chave]:
            problemas.append(f"o arquivo do plano de {papel} tem hoje sha256 "
                             f"{str(sha256_dos_planos.get(papel))[:12]}, declarado {declarado}")
    if entradas.get("checkpoint_sha256") != referencia["checkpoint_sha256"]:
        problemas.append(f"checkpoint {str(entradas.get('checkpoint_sha256'))[:12]} nao e o R03 declarado")
    for chave, esperado in (("exemplos_de_treino", referencia["janelas_de_treino"]),
                            ("exemplos_de_validacao", declarada["limite_validacao"])):
        if entradas.get(chave) != esperado:
            problemas.append(f"{chave} = {entradas.get(chave)}, esperado {esperado}")
    for chave in ("falhas_treino", "falhas_validacao"):
        if entradas.get(chave):
            problemas.append(f"{chave}: {entradas[chave]}")

    if recorte.get("sha256") != recorte_esperado.get("sha256") or recorte.get("janelas") != recorte_esperado.get(
            "janelas"):
        problemas.append(f"recorte da corrida {str(recorte.get('sha256'))[:16]} ({recorte.get('janelas')} janelas) "
                         f"!= re-sorteado do plano declarado {str(recorte_esperado.get('sha256'))[:16]} "
                         f"({recorte_esperado.get('janelas')})")
    registrado = entradas.get("recorte_da_validacao")
    if registrado and registrado.get("sha256") != recorte.get("sha256"):
        problemas.append("o recorte registrado no relatorio nao e o do detalhe da validacao")

    base = float((relatorio.get("linha_de_base") or {}).get("criterio_primario", {}).get("valor", math.nan))
    melhor = relatorio.get("melhor_por_validacao") or {}
    valor = float(melhor.get("valor", math.nan))
    do_historico = melhor_do_historico(relatorio)
    supera = math.isfinite(base) and math.isfinite(valor) and valor < base
    if not (math.isfinite(base) and math.isfinite(valor)):
        problemas.append(f"base ({base}) ou melhor ({valor}) nao finitos")
    elif not supera:
        problemas.append(f"o melhor ({valor:.4f}) nao e menor que a base ({base:.4f}): pela regra, nao ha MR com "
                         f"esta receita para a semente")
    if bool(melhor.get("supera_a_base")) != supera:
        problemas.append(f"o relatorio diz supera_a_base={melhor.get('supera_a_base')}, recalculado {supera}")
    if do_historico is None:
        problemas.append("o historico nao tem validacao: o melhor nao pode ser conferido")
    elif int(melhor.get("passo", -1)) != do_historico[0] or not _igual(valor, do_historico[1]):
        problemas.append(f"o melhor declarado (passo {melhor.get('passo')}, {valor}) nao e o menor do historico "
                         f"(passo {do_historico[0]}, {do_historico[1]})")
    if melhor.get("sha256") != sha256_do_arquivo:
        problemas.append("o sha256 de adapter_melhor.pt nao e o registrado no relatorio")
    if problemas:
        return None, problemas

    return {
        "sha256": sha256_do_arquivo,
        "passo": int(melhor["passo"]),
        "focal_alt_validacao": {"base": round(base, 4), "adapter": round(valor, 4), "delta": round(valor - base, 4)},
        "supera_a_base": supera,
        "pela_regra": PELA_REGRA,
        "recorte_da_validacao_sha256": recorte["sha256"],
        "conferido": ("receita e orcamento no relatorio e no checkpoint; planos, R03 e recorte contra a referencia "
                      "declarada; melhor recalculado do historico"),
    }, []
