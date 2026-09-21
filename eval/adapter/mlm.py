"""Nucleo do MLM do adapter populacional: janela, mascaras, alvos e DECOMPOSICAO da loss.

Stdlib puro. SEM torch, SEM FASTA, SEM rede -- roda no Windows e e onde a
regra pode ser provada. A parte que toca o modelo fica no treinador, fina de proposito.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.1.

O QUE O ADAPTER E, E O QUE ELE NAO E
------------------------------------
E um adapter POPULACIONAL treinado com MLM sobre janelas em que variantes amostradas por frequencia foram
aplicadas. **Ele nao preve AF.** A frequencia entra na AMOSTRAGEM (quais variantes compoem o plano, em que
proporcao por bin), nunca como entrada nem como alvo. Trocar isso por regressao de frequencia seria outro
experimento, com outra interpretacao -- e o desenho aprovado nao e esse.

AS TRES CATEGORIAS DE POSICAO MASCARADA
---------------------------------------
Esta e a distincao que o relatorio e a loss precisam carregar, e que se perde se alguem falar em "loss do span
da variante":

  `focal_alt`              a UNICA posicao onde o alelo alternativo foi aplicado. O alvo e o ALT. E a pergunta da
                           campanha: dado o contexto, o modelo preve o alelo que aquela populacao carrega?
  `contexto_da_variante`   as demais posicoes DENTRO do span que cobre o focal. O span tem de 3 a 10 bp, entao ele
                           quase sempre cobre bases de REFERENCIA alem do focal. Os alvos delas sao de referencia,
                           nao o alelo variante. Juntar estas com a de cima diluiria a medida que interessa.
  `referencia`             posicoes dos spans que nao tocam o focal. Sao a guarda contra o atalho
                           "mascarado = variante": sem elas, mascara viraria sinonimo de alelo alternativo.

ONDE ESTA A ARMADILHA DA VARREDURA DE PESOS
-------------------------------------------
Comparar configuracoes pela propria loss ponderada de cada uma nao compara nada: cada configuracao otimiza um
objetivo diferente e a "melhor" seria a de pesos mais frouxos. O criterio de validacao tem de ser COMUM e
independente dos pesos de treino -- aqui, a entropia cruzada media por posicao, NAO ponderada, publicada por
categoria, com uma delas declarada primaria antes de rodar (`CRITERIO_PRIMARIO`).

O QUE NAO PROVA
---------------
- Nao valida o modelo: aqui nao ha modelo. Valida a construcao do exemplo e a contabilidade da loss.
- Nao decide os pesos: eles sao declarados por quem chama.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Vocabulario do R03, REDECLARADO em vez de importado: `lumina/__init__.py` importa torch no topo, e este
#: modulo existe justamente para ser provado sem torch. `lumina/constants.py` em si nao depende de nada, entao
#: `tests/test_adapter_mlm.py` carrega o arquivo por caminho e trava a igualdade -- se o vocabulario do modelo
#: mudar, o teste quebra em vez de o treinador tokenizar errado em silencio.
DNA_VOCAB: dict[str, int] = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
MASK_ID = 6
VOCAB_SIZE = 8

CATEGORIA_FOCAL = "focal_alt"
CATEGORIA_CONTEXTO = "contexto_da_variante"
CATEGORIA_REFERENCIA = "referencia"
CATEGORIAS = (CATEGORIA_FOCAL, CATEGORIA_CONTEXTO, CATEGORIA_REFERENCIA)

TIPO_VARIANTE = "variante"
TIPO_REFERENCIA = "referencia"

#: Declarado ANTES de qualquer varredura: e por ele que duas configuracoes de peso sao comparadas. Nao ponderado,
#: entao nao favorece quem treinou com peso frouxo.
CRITERIO_PRIMARIO = CATEGORIA_FOCAL

#: Receita inicial dos pesos de treino. E ponto de partida declarado, nao resultado de busca.
PESOS_INICIAIS: dict[str, float] = {
    CATEGORIA_FOCAL: 1.0,
    CATEGORIA_CONTEXTO: 0.5,
    CATEGORIA_REFERENCIA: 0.5,
}


@dataclass(frozen=True)
class Exemplo:
    """Uma janela pronta para o MLM, com a proveniencia de cada posicao mascarada."""

    variant_id: str
    fonte: str
    input_ids: tuple[int, ...]
    posicoes: tuple[int, ...]
    alvos: tuple[int, ...]
    categorias: tuple[str, ...]
    focal_index: int
    metadados: dict[str, Any] = field(default_factory=dict)

    def por_categoria(self) -> dict[str, tuple[int, ...]]:
        """Indices de `posicoes` agrupados por categoria -- a chave da decomposicao da loss."""
        out: dict[str, list[int]] = {c: [] for c in CATEGORIAS}
        for indice, categoria in enumerate(self.categorias):
            out[categoria].append(indice)
        return {c: tuple(v) for c, v in out.items()}


def codificar(sequencia: str) -> tuple[int, ...]:
    """DNA para ids do R03. Base fora de ACGTN e erro: a auditoria de janelas ja garantiu ACGT."""
    ids = []
    for base in sequencia.upper():
        token = DNA_VOCAB.get(base)
        if token is None:
            raise ValueError(f"base {base!r} fora do vocabulario {sorted(DNA_VOCAB)}")
        ids.append(token)
    return tuple(ids)


def spans_do_plano(bruto: Any) -> list[tuple[int, int, str]]:
    """Le a coluna `spans` do plano, que o gerador grava como JSON."""
    if isinstance(bruto, str):
        bruto = json.loads(bruto)
    return [(int(inicio), int(fim), str(tipo)) for inicio, fim, tipo in bruto]


def categoria_da_posicao(posicao: int, *, focal_index: int, tipo_do_span: str) -> str:
    """A regra em uma linha: so a posicao focal do span de variante e `focal_alt`."""
    if tipo_do_span == TIPO_VARIANTE and posicao == focal_index:
        return CATEGORIA_FOCAL
    if tipo_do_span == TIPO_VARIANTE:
        return CATEGORIA_CONTEXTO
    return CATEGORIA_REFERENCIA


def montar_exemplo(
    alt_seq: str,
    spans: Sequence[tuple[int, int, str]],
    *,
    variant_id: str,
    fonte: str,
    focal_index: int,
    mask_id: int = MASK_ID,
) -> Exemplo:
    """Monta o exemplo a partir da janela JA com o ALT aplicado.

    `alt_seq` vem de `eval/embedding_probe/windows.py::build_window`, que confere o REF contra o FASTA e aplica o
    ALT no indice focal. Aqui so se mascara e se rotula -- a geometria ja foi provada la e na auditoria de janelas.

    O alvo de cada posicao e a base de `alt_seq`, entao o alvo do focal e o ALT e os demais sao de referencia. E
    exatamente por isso que as categorias existem: sem elas, "loss do span da variante" misturaria as duas coisas.
    """
    if not 0 <= focal_index < len(alt_seq):
        raise ValueError(f"focal_index {focal_index} fora de [0,{len(alt_seq)})")

    ids = list(codificar(alt_seq))
    alvos_por_posicao: dict[int, int] = {}
    categorias_por_posicao: dict[int, str] = {}
    cobrindo_o_focal = 0

    for inicio, fim, tipo in spans:
        if not 0 <= inicio < fim <= len(alt_seq):
            raise ValueError(f"span [{inicio},{fim}) fora de [0,{len(alt_seq)})")
        if tipo == TIPO_VARIANTE and inicio <= focal_index < fim:
            cobrindo_o_focal += 1
        for posicao in range(inicio, fim):
            if posicao in alvos_por_posicao:
                raise ValueError(f"posicao {posicao} coberta por mais de um span")
            alvos_por_posicao[posicao] = ids[posicao]
            categorias_por_posicao[posicao] = categoria_da_posicao(
                posicao, focal_index=focal_index, tipo_do_span=tipo)

    if cobrindo_o_focal != 1:
        raise ValueError(f"{cobrindo_o_focal} spans de variante cobrem o focal {focal_index}, esperado 1")

    for posicao in alvos_por_posicao:
        ids[posicao] = mask_id

    ordenadas = tuple(sorted(alvos_por_posicao))
    return Exemplo(
        variant_id=variant_id, fonte=fonte, input_ids=tuple(ids), posicoes=ordenadas,
        alvos=tuple(alvos_por_posicao[p] for p in ordenadas),
        categorias=tuple(categorias_por_posicao[p] for p in ordenadas),
        focal_index=focal_index,
        metadados={"window_bp": len(alt_seq), "mascaradas": len(ordenadas)},
    )


def decompor_perdas(
    perdas: Sequence[float], categorias: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Media NAO ponderada da perda em cada categoria, mais a contagem de posicoes.

    Nao ponderada de proposito: e este numero que compara configuracoes treinadas com pesos diferentes.
    """
    if len(perdas) != len(categorias):
        raise ValueError(f"{len(perdas)} perdas para {len(categorias)} categorias")
    soma: dict[str, float] = {c: 0.0 for c in CATEGORIAS}
    quantas: dict[str, int] = {c: 0 for c in CATEGORIAS}
    for perda, categoria in zip(perdas, categorias):
        if categoria not in soma:
            raise ValueError(f"categoria {categoria!r} fora de {CATEGORIAS}")
        soma[categoria] += float(perda)
        quantas[categoria] += 1
    return {c: {"media": (soma[c] / quantas[c]) if quantas[c] else float("nan"),
                "posicoes": quantas[c]} for c in CATEGORIAS}


def perda_ponderada(por_categoria: Mapping[str, Mapping[str, float]], pesos: Mapping[str, float]) -> float:
    """Loss de TREINO: media das posicoes, com cada categoria pesada como declarado.

    Pesa por POSICAO e nao por categoria: com peso igual, uma janela com 1 focal e 12 de referencia nao deve
    valer metade focal e metade referencia. O peso expressa prioridade do objetivo, nao normalizacao.
    """
    faltando = [c for c in CATEGORIAS if c not in pesos]
    if faltando:
        raise ValueError(f"pesos faltando para {faltando}")
    numerador = 0.0
    denominador = 0.0
    for categoria in CATEGORIAS:
        dados = por_categoria.get(categoria) or {}
        quantas = float(dados.get("posicoes", 0) or 0)
        if not quantas:
            continue
        media = float(dados["media"])
        numerador += pesos[categoria] * media * quantas
        denominador += pesos[categoria] * quantas
    if denominador == 0:
        raise ValueError("nenhuma posicao mascarada com peso positivo")
    return numerador / denominador


def agregar(relatorios: Iterable[Mapping[str, Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    """Junta decomposicoes de varios exemplos, ponderando pela contagem de posicoes de cada um."""
    soma: dict[str, float] = {c: 0.0 for c in CATEGORIAS}
    quantas: dict[str, int] = {c: 0 for c in CATEGORIAS}
    for relatorio in relatorios:
        for categoria in CATEGORIAS:
            dados = relatorio.get(categoria) or {}
            n = int(dados.get("posicoes", 0) or 0)
            if not n:
                continue
            soma[categoria] += float(dados["media"]) * n
            quantas[categoria] += n
    return {c: {"media": (soma[c] / quantas[c]) if quantas[c] else float("nan"),
                "posicoes": quantas[c]} for c in CATEGORIAS}
