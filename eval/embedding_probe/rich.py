"""Extracao RICA: todos os pontos de tomada do R03 numa unica passada de GPU.

A ideia do R1: o forward e o caro; guardar mais coisa dele e quase gratis. Entao extraimos de uma
vez tudo que as ablacoes do R2 precisam -- 27 profundidades do mid-stack, os register tokens, a
representacao regional, o trunk pre e pos norma em varias escalas, as cabecas supervisionadas e a
media reverse-complement.

COMO OS HOOKS FUNCIONAM AQUI (e por que nao sao forward hooks)
--------------------------------------------------------------
``LuminaMidStack.forward`` NAO chama as camadas de atencao local como modulos::

    if kind == "mamba":   x = layer(x)
    elif kind == "local":  x = x + attn(norm(x), None)     # o ModuleDict nunca e invocado
    elif kind == "sparse": x = layer(x, edit_mid_mask=...)

Um ``register_forward_hook`` no ModuleDict das camadas locais jamais dispararia. Usamos entao
**forward PRE-hooks**, que capturam a ENTRADA de cada camada -- ou seja, a saida da anterior:

    pre-hook na camada i (mamba/sparse)      -> estado antes de i
    pre-hook no ``norm`` da camada i (local) -> estado antes de i  (o ModuleDict nao serve)
    forward hook na ULTIMA camada            -> estado final

Com isso cobrimos as 27 fronteiras (26 "antes da camada i" + a saida final), sem buraco.

REGISTERS
---------
Os 16 register tokens sao prependados no inicio do mid-stack e removidos so no fim
(``x = x[:, n_reg:]``). Logo, dentro de qualquer camada hookada o tensor e ``[B, 16 + L/4, d]``:
os registers estao em ``[:, :16]`` e a posicao focal em ``[:, 16 + focal//4]``. Ja o
``mid_hidden_state`` devolvido pelo ``encode()`` vem SEM registers.

CABECAS
-------
As cabecas leem o trunk POS-RMSNorm. Para as 7 lineares vale ``cabeca(alt) - cabeca(ref) = W.Delta``
com o Delta POS-norma -- exato e sem forward extra. As 3 MLP nao sao lineares, entao rodamos a
cabeca nas duas torres e subtraimos.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

# Cabecas lineares sobre o trunk: para elas W.Delta e exato.
LINEAR_HEADS = (
    "mlm_head", "conservation_scalar_head", "conservation_bin_head", "region_head",
    "counterfactual_snv_head", "population_af_head", "population_observed_head",
)
# Cabecas MLP: precisam rodar nas duas torres.
MLP_HEADS = ("splice_class_head", "splice_distance_head", "missense_severity_head")

# Layout de ``heads_lin`` no R03, verificado contra lumina/models/model.py e config/lumina_r03_base.json.
# As fatias em configs/*.json enderecam COLUNAS -- 'heads_lin[60:68]' so e o gnomAD se este layout
# valer. E ele depende do CONFIG, nao so do codigo: NUM_COUNTERFACTUAL_EFFECT_CLASSES tem default 12
# em lumina/constants.py e o R03 sobrescreve para 8. Rodar com o default daria 84 dims em vez de 68 e
# TODAS as fatias apontariam para colunas erradas, sem erro nenhum -- as ablacoes de circularidade e
# de conservacao mediriam outra coisa e o numero sairia normal.
R03_HEAD_LAYOUT = (
    ("mlm_head", 4),
    ("conservation_scalar_head", 3),
    ("conservation_bin_head", 16),
    ("region_head", 5),
    ("counterfactual_snv_head", 32),   # len(SNV_BASES)=4 x num_counterfactual_effect_classes=8
    ("population_af_head", 4),
    ("population_observed_head", 4),
)

BASES = ("A", "C", "G", "T")
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
SUBSTITUTIONS = tuple(f"{r}>{a}" for r in BASES for a in BASES if r != a)  # 12, ordem estavel


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq))


class MidStackTaps:
    """Captura o estado do mid-stack em todas as fronteiras de camada.

    Uso::

        taps = MidStackTaps(model)
        ...                       # um forward
        states = taps.states      # lista [B, 16 + L/4, d], da entrada ate a saida
        taps.close()
    """

    def __init__(self, model: Any) -> None:
        stack = model.mid_stack
        self.kinds: list[str] = list(stack.layer_kinds)
        self.n_registers = int(stack.n_register_tokens)
        self._handles: list[Any] = []
        self._buf: dict[int, Tensor] = {}
        self.n_taps = len(self.kinds) + 1  # 26 "antes de i" + a saida final

        for i, (kind, layer) in enumerate(zip(self.kinds, stack.layers)):
            target = layer["norm"] if kind == "local" else layer
            self._handles.append(target.register_forward_pre_hook(self._make_pre(i)))
        self._handles.append(stack.layers[-1].register_forward_hook(self._make_post(len(self.kinds))))

    def _make_pre(self, index: int):
        def hook(_module: Any, args: tuple) -> None:
            self._buf[index] = args[0]
        return hook

    def _make_post(self, index: int):
        def hook(_module: Any, _args: tuple, output: Any) -> None:
            self._buf[index] = output if isinstance(output, torch.Tensor) else output[0]
        return hook

    def reset(self) -> None:
        self._buf.clear()

    @property
    def states(self) -> list[Tensor]:
        missing = [i for i in range(self.n_taps) if i not in self._buf]
        if missing:
            raise RuntimeError(f"taps do mid-stack nao dispararam: {missing}")
        return [self._buf[i] for i in range(self.n_taps)]

    def describe(self) -> list[str]:
        """Rotulo de cada tomada: o que veio ANTES dela."""
        out = ["entrada"]
        for i, kind in enumerate(self.kinds):
            out.append(f"apos_{i:02d}_{kind}")
        return out

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def mid_index(focal_index: int, downsample: int = 4) -> int:
    """Posicao no mid correspondente ao focal full-res. Verificada empiricamente (checagem 1)."""
    return focal_index // downsample


@torch.inference_mode()
def head_readouts(model: Any, post_ref: Tensor, post_alt: Tensor) -> dict[str, Tensor]:
    """Leituras supervisionadas no sitio. ``post_*`` sao ``[B, d_full]`` (trunk POS-norma).

    ``linear`` usa a identidade W.Delta (exata, sem custo). ``mlp`` roda as tres cabecas nao-lineares
    nas duas torres. ``ref`` guarda os valores absolutos na referencia, que sao o contexto sobre o
    qual o delta acontece.

    Guardamos os QUATRO alelos das cabecas por-alelo (counterfactual 4x8, missense 4) em vez de so
    o alelo real: nao se perde informacao, e o one-hot de substituicao diz ao modelo downstream qual
    posicao e a relevante. Selecionar so o alelo real seria mais denso mas jogaria fora "o que os
    outros alelos fariam aqui", que e contexto legitimo.
    """
    delta = post_alt - post_ref
    linear, ref_vals = [], []
    for name in LINEAR_HEADS:
        head = getattr(model, name)
        linear.append(delta @ head.weight.T)          # bias cancela na diferenca
        ref_vals.append(head(post_ref))
    mlp = []
    for name in MLP_HEADS:
        head = getattr(model, name, None)
        if head is None:
            continue
        out_ref, out_alt = head(post_ref), head(post_alt)
        if out_ref.ndim == 1:                          # splice_distance devolve [B]
            out_ref, out_alt = out_ref.unsqueeze(-1), out_alt.unsqueeze(-1)
        mlp.append(out_alt - out_ref)
        ref_vals.append(out_ref)
    return {
        "linear": torch.cat(linear, dim=-1),
        "mlp": torch.cat(mlp, dim=-1) if mlp else delta[:, :0],
        "ref": torch.cat(ref_vals, dim=-1),
    }


def pooled(delta: Tensor, focal: int, radius: int, *, reduce: str = "mean") -> Tensor:
    """Agrega ``delta`` [B, L, d] numa janela simetrica de raio ``radius`` em torno do focal.

    O raio nao e arbitrario: 25 e o horizonte da via convolucional pura e 512 e o raio da atencao
    local (fixo, nao cresce com L) -- os dois medidos/derivados na sonda. Os limites sao cortados
    na janela, entao um raio maior que o disponivel simplesmente usa o que ha.
    """
    lo, hi = max(0, focal - radius), min(delta.shape[1], focal + radius + 1)
    window = delta[:, lo:hi, :]
    return window.mean(dim=1) if reduce == "mean" else window.abs().amax(dim=1)


def substitution_onehot(ref: str, alt: str) -> list[float]:
    """4 (base de referencia) + 12 (substituicao ordenada) = 16 dims.

    Substitui as 64 dims de ``h_pure``, que codificam exatamente esta informacao: num offset focal
    fixo, ``h_pure(ref)`` assume 4 valores distintos no dataset inteiro e ``h_pure(Delta)``, 12.
    """
    vec = [0.0] * 16
    vec[BASES.index(ref)] = 1.0
    vec[4 + SUBSTITUTIONS.index(f"{ref}>{alt}")] = 1.0
    return vec


def head_layout(model: Any) -> list[tuple[str, int, int]]:
    """(nome, inicio, fim) de cada cabeca linear dentro do bloco ``heads_lin``."""
    out, off = [], 0
    for name in LINEAR_HEADS:
        n = int(getattr(model, name).out_features)
        out.append((name, off, off + n))
        off += n
    return out


def assert_r03_head_layout(model: Any) -> list[tuple[str, int, int]]:
    """Falha alto se o layout das cabecas nao for o do R03. Ver R03_HEAD_LAYOUT."""
    layout = head_layout(model)
    got = [(name, end - start) for name, start, end in layout]
    if got != list(R03_HEAD_LAYOUT):
        linhas = "\n".join(
            f"    {name:<28} esperado {exp:>3}  obtido {obt:>3}"
            f"{'   <-- DIFERE' if exp != obt else ''}"
            for (name, exp), (_, obt) in zip(R03_HEAD_LAYOUT, got)
        )
        raise SystemExit(
            "layout das cabecas lineares nao e o do R03 -- as fatias dos configs "
            "('heads_lin[60:68]' = gnomAD, 'heads_lin[4:23]' = conservacao) apontariam para "
            f"colunas erradas SEM dar erro:\n{linhas}\n"
            "  confira num_counterfactual_effect_classes no config do checkpoint (R03 usa 8; "
            "o default de lumina/constants.py e 12)."
        )
    return layout
