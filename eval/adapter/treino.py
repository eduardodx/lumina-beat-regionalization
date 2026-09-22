"""Nucleo em torch do treino do adapter populacional: loss diferenciavel, invariantes e checkpoint.

Precisa de torch. A REGRA (quem e alvo, de que categoria, como as perdas se combinam) mora em
`eval/adapter/mlm.py`, que roda sem torch; aqui so entra o que exige tensores e o modelo.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.1.

POR QUE A LOSS E REESCRITA AQUI
-------------------------------
`mlm.perda_ponderada` converte para `float`: serve de REFERENCIA NUMERICA e de relatorio, e romperia o caminho
do gradiente se virasse a loss de treino. Esta versao calcula a MESMA formula com tensores --
``L = sum_i w_cat(i) * CE_i / sum_i w_cat(i)`` -- e so destaca os valores depois, para registrar. O teste que
liga os dois mundos confere a igualdade numerica; se as duas formulas divergirem, ele quebra.

O QUE FICA TREINAVEL, E POR QUE
-------------------------------
Ordem obrigatoria: carregar os pesos base, CONGELAR TUDO, e so entao inserir o rsLoRA -- assim os modulos novos
nascem treinaveis num modelo ja congelado. Conferido no codigo (`eval/clinvar/lora.py`):

  * os parametros do adapter se chamam `lora_a` e `lora_b`;
  * `lora_b` NASCE EM ZEROS. Logo, no primeiro passo o gradiente de `lora_a` e zero por construcao -- exigir
    gradiente nao nulo em TODO tensor do adapter reprovaria um treino correto. O que se exige e que o adapter
    receba sinal e MUDE, enquanto o que esta congelado permanece identico;
  * `mlm_head` esta em `_EXCLUDE_PATTERNS`, entao o `apply_lora` NAO a embrulha. Ela fica congelada e mesmo assim
    participa do grafo: e por ela que o erro chega ao adapter. Nada de `no_grad()` neste caminho.

`eval()` e decisao SEPARADA de congelar peso: congelar zera o gradiente, nao desliga dropout. O modo e escolhido
por quem chama e vai declarado no manifesto.

O QUE NAO PROVA
---------------
- Nao demonstra aprendizado de estrutura populacional nem ganho clinico: mede reconstrucao mascarada.
- Nao decide pesos, taxa de aprendizado nem duracao.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eval.adapter import mlm

#: Formato proprio. NAO reaproveitar `abraom_frequency_adapter_v1`: aquele e de outro objetivo (regressao de AF)
#: e guarda outro conjunto de parametros.
FORMATO = "lumina_population_adapter_mlm_v1"

#: Sufixos dos parametros que o `apply_lora` cria. Qualquer outro treinavel e defeito, nao variacao.
SUFIXOS_DO_ADAPTER = ("lora_a", "lora_b")


def e_do_adapter(nome: str) -> bool:
    return nome.rsplit(".", 1)[-1] in SUFIXOS_DO_ADAPTER


def congelar_tudo(modulo: nn.Module) -> int:
    """`requires_grad=False` em todo parametro. Roda ANTES do `apply_lora`."""
    quantos = 0
    for parametro in modulo.parameters():
        parametro.requires_grad = False
        quantos += 1
    return quantos


def parametros_do_adapter(modulo: nn.Module) -> dict[str, Tensor]:
    """Os parametros treinaveis, que tem de ser exatamente os do adapter."""
    return {nome: p for nome, p in modulo.named_parameters() if p.requires_grad}


def assert_so_o_adapter_treina(modulo: nn.Module) -> list[str]:
    """Falha se algo fora do adapter estiver treinavel, ou se o adapter nao estiver.

    Nao basta contar: um LayerNorm descongelado por engano, ou uma cabeca nativa, passaria numa contagem e
    mudaria o backbone que a campanha promete manter fixo.
    """
    treinaveis = parametros_do_adapter(modulo)
    intrusos = sorted(nome for nome in treinaveis if not e_do_adapter(nome))
    if intrusos:
        raise RuntimeError(f"parametros treinaveis fora do adapter: {intrusos[:10]}"
                           f"{f' (+{len(intrusos) - 10})' if len(intrusos) > 10 else ''}")
    if not treinaveis:
        raise RuntimeError("nenhum parametro treinavel: o apply_lora rodou antes do congelamento?")
    return sorted(treinaveis)


def impressao_dos_congelados(modulo: nn.Module) -> str:
    """Hash dos parametros NAO treinaveis: comparado antes e depois do passo, prova que o backbone nao mudou."""
    digest = hashlib.sha256()
    for nome, parametro in sorted(modulo.named_parameters(), key=lambda item: item[0]):
        if parametro.requires_grad:
            continue
        digest.update(nome.encode("utf-8"))
        digest.update(parametro.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def logits_mlm(adapter: Any, input_ids: Tensor) -> Tensor:
    """[B, L, 4] -- as quatro classes da cabeca MLM (A, C, G, T).

    Caminho deliberadamente curto: `encode` e `mlm_head`, sem passar pelas outras cabecas do R03. E SEM
    `no_grad()`: a `mlm_head` e congelada, mas o gradiente precisa atravessa-la para chegar ao adapter.
    """
    hidden = adapter.forward_hidden_states({"input_ids": input_ids})
    logits = adapter.backbone.mlm_head(hidden)
    if logits.shape[-1] != len(mlm.SNV_BASES):
        raise RuntimeError(f"mlm_head devolveu {logits.shape[-1]} classes, esperado {len(mlm.SNV_BASES)}")
    return logits


@dataclass(frozen=True)
class Lote:
    """Um lote ja achatado: cada posicao mascarada vira uma linha."""

    input_ids: Tensor      # [B, L], espaco DNA_VOCAB
    indice_no_lote: Tensor  # [N] long
    posicao: Tensor         # [N] long
    alvo: Tensor            # [N] long, 0..3
    categoria: Tensor       # [N] long, indice em mlm.CATEGORIAS
    fontes: tuple[str, ...] = ()


def montar_lote(exemplos: Sequence[mlm.Exemplo], *, device: torch.device | str = "cpu") -> Lote:
    """Achata os exemplos. O `Exemplo` ja garantiu categorias e espaco dos alvos."""
    if not exemplos:
        raise ValueError("lote vazio")
    largura = len(exemplos[0].input_ids)
    if any(len(e.input_ids) != largura for e in exemplos):
        raise ValueError("todos os exemplos do lote precisam ter a mesma janela")
    indice_da_categoria = {c: i for i, c in enumerate(mlm.CATEGORIAS)}

    entradas, no_lote, posicoes, alvos, categorias = [], [], [], [], []
    for indice, exemplo in enumerate(exemplos):
        entradas.append(list(exemplo.input_ids))
        for posicao, alvo, categoria in zip(exemplo.posicoes, exemplo.alvos, exemplo.categorias):
            no_lote.append(indice)
            posicoes.append(int(posicao))
            alvos.append(int(alvo))
            categorias.append(indice_da_categoria[categoria])

    longo = dict(dtype=torch.long, device=device)
    return Lote(
        input_ids=torch.tensor(entradas, **longo),
        indice_no_lote=torch.tensor(no_lote, **longo),
        posicao=torch.tensor(posicoes, **longo),
        alvo=torch.tensor(alvos, **longo),
        categoria=torch.tensor(categorias, **longo),
        fontes=tuple(e.fonte for e in exemplos),
    )


def perda_do_lote(
    logits: Tensor, lote: Lote, pesos: Mapping[str, float]
) -> tuple[Tensor, dict[str, dict[str, float]]]:
    """Loss DIFERENCIAVEL e a decomposicao destacada, nesta ordem.

    Mesma formula de `mlm.perda_ponderada`, em tensores. A decomposicao sai com `.detach()`: relatorio nao pode
    segurar o grafo, e a loss de treino nao pode nascer de `float`.
    """
    limpos = mlm.validar_pesos(pesos)
    selecionados = logits[lote.indice_no_lote, lote.posicao]           # [N, 4]
    ce = F.cross_entropy(selecionados, lote.alvo, reduction="none")    # [N]

    tabela = torch.tensor([limpos[c] for c in mlm.CATEGORIAS], dtype=ce.dtype, device=ce.device)
    peso = tabela[lote.categoria]                                      # [N]
    denominador = peso.sum()
    if float(denominador) <= 0:
        raise ValueError("nenhuma posicao mascarada com peso positivo neste lote")
    perda = (peso * ce).sum() / denominador

    nomes = [mlm.CATEGORIAS[i] for i in lote.categoria.detach().cpu().tolist()]
    decomposicao = mlm.decompor_perdas(ce.detach().float().cpu().tolist(), nomes)
    return perda, decomposicao


def decomposicao_por_fonte(
    logits: Tensor, lote: Lote
) -> dict[str, dict[str, dict[str, float]]]:
    """A mesma decomposicao, separada por fonte -- a comparacao que o numero agregado esconde."""
    with torch.no_grad():
        selecionados = logits[lote.indice_no_lote, lote.posicao]
        ce = F.cross_entropy(selecionados, lote.alvo, reduction="none").float().cpu().tolist()
    nomes = [mlm.CATEGORIAS[i] for i in lote.categoria.detach().cpu().tolist()]
    fontes = [lote.fontes[i] for i in lote.indice_no_lote.detach().cpu().tolist()]

    por_fonte: dict[str, tuple[list[float], list[str]]] = {}
    for perda, categoria, fonte in zip(ce, nomes, fontes):
        perdas, categorias = por_fonte.setdefault(fonte, ([], []))
        perdas.append(perda)
        categorias.append(categoria)
    return {fonte: mlm.decompor_perdas(p, c) for fonte, (p, c) in sorted(por_fonte.items())}


def receita_do_adapter(modulo: nn.Module) -> dict[str, Any]:
    """Le do MODELO a receita do rsLoRA, para comparar com a que o checkpoint declara.

    Conferir chaves e formas nao basta: duas configuracoes com as MESMAS formas e escalas diferentes -- `alpha`
    ou `use_rslora` trocados -- seriam aceitas uma pela outra, e o adapter aplicaria um delta com magnitude
    errada sem nada reclamar. A escala e recuperavel: `scaling = alpha / (sqrt(r) se rslora senao r)`.
    """
    modulos: list[str] = []
    ranks: set[int] = set()
    rsloras: set[bool] = set()
    alphas: set[float] = set()
    for nome, filho in modulo.named_modules():
        if not hasattr(filho, "lora_a") or not hasattr(filho, "scaling"):
            continue
        modulos.append(nome)
        rank = int(filho.lora_a.shape[0])
        usa_rslora = bool(getattr(filho, "use_rslora", False))
        ranks.add(rank)
        rsloras.add(usa_rslora)
        alphas.add(round(float(filho.scaling) * ((rank ** 0.5) if usa_rslora else rank), 6))
    return {"modulos": sorted(modulos), "rank": sorted(ranks), "use_rslora": sorted(rsloras),
            "alpha": sorted(alphas)}


def gradientes_do_adapter(modulo: nn.Module) -> dict[str, Any]:
    """Estado dos gradientes ANTES do passo. `optimizer.step()` muda peso mesmo com gradiente zero.

    O AdamW tem `weight_decay` positivo por padrao: um parametro com gradiente nulo continua encolhendo. Logo
    "o adapter mudou" NAO e evidencia de que ele aprendeu -- so a de que o otimizador rodou.
    """
    do_adapter = parametros_do_adapter(modulo)
    sem_gradiente = sorted(nome for nome, p in do_adapter.items() if p.grad is None)
    nao_finitos = sorted(nome for nome, p in do_adapter.items()
                         if p.grad is not None and not bool(torch.isfinite(p.grad).all()))
    normas = {nome: float(p.grad.abs().sum()) for nome, p in do_adapter.items() if p.grad is not None}
    congelados_com_gradiente = sorted(
        nome for nome, p in modulo.named_parameters()
        if not p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0)
    com_gradiente_zero = sorted(nome for nome, p in do_adapter.items()
                                if p.grad is not None and float(p.grad.abs().sum()) == 0)
    return {
        "tensores_do_adapter": len(do_adapter),
        "sem_gradiente": sem_gradiente,
        "com_gradiente_zero": com_gradiente_zero,
        "diferenca": ("`sem_gradiente` = o modulo nem entrou no grafo (embrulho inerte, caminho nao percorrido); "
                      "`com_gradiente_zero` = entrou e nao recebeu sinal (lora_a no primeiro passo, ou caminho "
                      "tocado com magnitude zero). Sao diagnosticos diferentes"),
        "nao_finitos": nao_finitos,
        "com_gradiente_nao_nulo": sorted(n for n, v in normas.items() if v > 0),
        "congelados_com_gradiente": congelados_com_gradiente,
        "nota": ("lora_b nasce em zeros, entao no primeiro passo o gradiente de lora_a e zero por construcao: "
                 "exigir nao nulo em TODOS reprovaria um treino correto"),
    }


# --------------------------------------------------------------------------- checkpoint

def salvar_adapter(
    caminho: Path, *, backbone: nn.Module, resumo_lora: Any, config: Mapping[str, Any],
    identidades: Mapping[str, str], metricas: Mapping[str, Any], passo: int,
) -> dict[str, Any]:
    """Salva SO os parametros do adapter, com a identidade do que o produziu.

    Recusa salvar se algo fora do adapter estiver treinavel: um checkpoint "do adapter" que carrega LayerNorm
    descongelado ou cabeca extra e outro objeto, e a diferenca reapareceria como resultado inexplicavel.
    """
    estado = {nome: p.detach().cpu() for nome, p in backbone.named_parameters() if p.requires_grad}
    intrusos = sorted(nome for nome in estado if not e_do_adapter(nome))
    if intrusos:
        raise RuntimeError(f"nao salvo: treinaveis fora do adapter: {intrusos[:10]}")
    if not estado:
        raise RuntimeError("nao salvo: nenhum parametro do adapter")

    carga = {
        "formato": FORMATO,
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
        "passo": int(passo),
        "config": dict(config),
        "identidades": dict(identidades),
        "lora": asdict(resumo_lora) if hasattr(resumo_lora, "__dataclass_fields__") else resumo_lora,
        "chaves": sorted(estado),
        "receita_lida_do_modelo": receita_do_adapter(backbone),
        "formas": {nome: list(t.shape) for nome, t in sorted(estado.items())},
        "metricas": dict(metricas),
        "estado_do_adapter": estado,
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save(carga, caminho)
    return {chave: carga[chave] for chave in ("formato", "passo", "chaves", "identidades")}


def carregar_adapter(caminho: Path, backbone: nn.Module) -> dict[str, Any]:
    """Carrega exigindo o CONJUNTO EXATO de chaves do adapter -- nem faltando, nem sobrando.

    O carregador antigo usava `strict=False` e so recusava algumas chaves INESPERADAS; chave AUSENTE passava em
    silencio e deixava parte do adapter na inicializacao. Aqui os dois lados sao conferidos, mais as formas.
    """
    carga = torch.load(caminho, map_location="cpu", weights_only=False)
    if carga.get("formato") != FORMATO:
        raise RuntimeError(f"{caminho} nao e {FORMATO} (formato={carga.get('formato')!r})")
    estado = carga.get("estado_do_adapter")
    if not isinstance(estado, dict):
        raise RuntimeError(f"{caminho} nao tem estado_do_adapter")

    atuais = parametros_do_adapter(backbone)
    faltando = sorted(set(atuais) - set(estado))
    sobrando = sorted(set(estado) - set(atuais))
    if faltando or sobrando:
        raise RuntimeError(f"conjunto de chaves nao bate: faltando={faltando[:5]} sobrando={sobrando[:5]}")
    incompativeis = [nome for nome in atuais if tuple(estado[nome].shape) != tuple(atuais[nome].shape)]
    if incompativeis:
        raise RuntimeError(f"formas incompativeis em {incompativeis[:5]}")

    # Formas iguais com ESCALA diferente passariam aqui sem isto: `alpha` ou `use_rslora` trocados aplicam um
    # delta de magnitude errada, e o resultado sairia diferente sem nada reclamar.
    declarada = carga.get("lora") or {}
    atual = receita_do_adapter(backbone)
    for campo, esperado in (("use_rslora", declarada.get("use_rslora")),
                            ("rank", declarada.get("rank")),
                            ("alpha", declarada.get("alpha"))):
        if esperado is None:
            continue
        obtidos = atual[campo]
        if len(obtidos) != 1 or obtidos[0] != (round(float(esperado), 6) if campo == "alpha" else esperado):
            raise RuntimeError(f"receita do rsLoRA nao bate em {campo}: checkpoint={esperado} modelo={obtidos}")

    with torch.no_grad():
        for nome, tensor in estado.items():
            atuais[nome].copy_(tensor.to(atuais[nome].device, atuais[nome].dtype))
    return {chave: carga.get(chave) for chave in ("formato", "passo", "config", "identidades", "lora",
                                                  "metricas")}


def identidade_do_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with open(caminho, "rb") as handle:
        for pedaco in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(pedaco)
    return digest.hexdigest()


def resumo_json(objeto: Any) -> str:
    return json.dumps(objeto, ensure_ascii=False, indent=2, sort_keys=True, default=str)
