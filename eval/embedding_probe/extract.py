"""Extracao da sonda: hidden states PRE e POS a RMSNorm final, metricas e perfil espacial.

Por que nao reusar ``eval/clinvar/adapters.py::_extract_paired_variant_features``: ela colapsa o
par em tres vetores POS-norm de 448 dims. Isso e insuficiente aqui por uma razao arquitetural:

    hidden = torch.cat([h_up, h_pure], dim=-1)   # model.py:369  (384 + 64 = 448)
    hidden = self.trunk_final_norm(hidden)       # model.py:374  nn.RMSNorm(448)

e ``h_pure = self.purity(x)`` com ``purity = nn.Linear(384, 64)`` sobre ``x = token_emb + pos_emb``
(backbone.py:64) -- **pointwise, sem contexto nenhum**. Logo:

    Delta h_pure[focal] = W_purity . (E[alt] - E[ref])

e um vetor CONSTANTE por par ordenado (ref,alt): igual em qualquer locus, qualquer contexto e
qualquer tamanho de janela; e EXATAMENTE zero em toda posicao != focal. Ou seja, 64 das 448
dimensoes codificam a identidade da base por construcao, e "o embedding mudou?" da positivo em
100% das variantes mesmo para um modelo que fosse uma lookup table.

A pergunta real vive em ``h_up`` (as 384 dims que atravessam o mid-stack Mamba). Para separar os
dois blocos sem a mistura que a RMSNorm de 448 introduz, capturamos o tensor PRE-norm com um
forward pre-hook em ``trunk_final_norm``. Continuamos reportando tambem o bloco POS-norm de 448,
que e o que a infra existente produz, para os numeros seguirem comparaveis.

A constancia de ``Delta h_pure`` e uma previsao analitica exata -- serve como teste de unidade
gratuito do pipeline (``check_hpure_invariant``): se ela falhar, a extracao esta errada.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from eval.embedding_probe.profile import ProfileBin, profile_bins

BASES: tuple[str, ...] = ("A", "C", "G", "T")

# INVARIANTE QUE CARREGA PESO: todo forward da sonda usa batch de exatamente 4 (as 4 bases do
# sitio). Medido em scripts/probe_batch_diagnostic.py: nao ha cross-talk entre linhas do batch
# (invariancia de conteudo = 0.0 exato), mas cuBLAS/cuDNN/Mamba escolhem algoritmo em funcao da
# dimensao de batch, e como soma em float nao e associativa o resultado muda ~2e-3 (pre-norm)
# entre B=1/2/4/8. Isso e inofensivo AQUI porque ref e alt vao no MESMO forward e o vies cancela
# na diferenca (Delta estavel a 0.4%; sinal/ruido = 426x). Deixa de ser inofensivo se alguem
# agrupar varios sitios num batch maior: as comparacoes ENTRE janelas (E4) passariam a carregar
# esse ruido. Por isso o tamanho e travado, nao apenas documentado.
PROBE_BATCH_SIZE = 4


@dataclass(frozen=True)
class PairMetrics:
    """Metricas de um par (ref, alt) num bloco de dimensoes."""

    cosine_distance: float  # 1 - cos(ref, alt); escala-invariante
    l2: float  # ||alt - ref||
    relative_l2: float  # ||alt - ref|| / ||ref||
    delta_sum: float  # sum(alt - ref) -- a "soma" que o Eduardo sugeriu
    ref_norm: float
    alt_norm: float

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_{k}": v for k, v in asdict(self).items()}


def pair_metrics(ref_vec: Tensor, alt_vec: Tensor) -> PairMetrics:
    """Metricas entre dois vetores 1-D. Em float64 -- a sonda mede diferencas pequenas."""
    ref = ref_vec.detach().to(torch.float64)
    alt = alt_vec.detach().to(torch.float64)
    delta = alt - ref
    ref_norm = float(ref.norm())
    alt_norm = float(alt.norm())
    denom = ref_norm * alt_norm
    cos = float(torch.dot(ref, alt)) / denom if denom > 0 else float("nan")
    return PairMetrics(
        cosine_distance=1.0 - cos,
        l2=float(delta.norm()),
        relative_l2=float(delta.norm()) / ref_norm if ref_norm > 0 else float("nan"),
        delta_sum=float(delta.sum()),
        ref_norm=ref_norm,
        alt_norm=alt_norm,
    )


class ProbeModel:
    """Backbone R03 com captura do trunk PRE-RMSNorm.

    O tensor capturado e valido ate a proxima chamada de ``encode``; consumimos na hora (nao
    clonamos: em 32k um clone de um batch de 4 sao ~230 MB por chamada, sem ganho).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device,
        *,
        dtype: torch.dtype = torch.float32,
        allow_tf32: bool = False,
    ) -> None:
        from eval.clinvar.r03_adapter import install_tilelang_fallback_shim

        install_tilelang_fallback_shim()  # antes do primeiro import de mamba_ssm (via lumina)
        from lumina import batch_encode_dna, load_model_from_checkpoint

        # TF32 tem ~10 bits de mantissa contra os 23 do fp32. As convolucoes do stem/down/up
        # passam por cuDNN, cujo TF32 vem LIGADO por default -- e a sonda compara dois forwards
        # quase identicos, entao o erro de TF32 pode ser da ordem do sinal medido. Desligado por
        # default; o teste de determinismo mostra se sobra ruido.
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        self._encode_dna = batch_encode_dna
        self.device = torch.device(device)
        self.dtype = dtype
        self.model: Any = load_model_from_checkpoint(
            checkpoint_path, device=self.device, dtype=dtype, strict=True
        )
        if self.model.trunk_final_norm is None:
            raise RuntimeError(
                "trunk_final_norm ausente (trunk_final_norm_enabled=False). A decomposicao "
                "h_up/h_pure PRE-norm depende dele; o R03 o tem habilitado."
            )
        self.d_model = int(self.model.cfg.d_model)
        self.d_pure = int(self.model.cfg.d_pure)
        self.d_full = int(self.model.full_hidden_dim)
        self.l_max = int(self.model.cfg.l_max)
        if self.d_model + self.d_pure != self.d_full:
            raise RuntimeError(f"d_model+d_pure != d_full: {self.d_model}+{self.d_pure} != {self.d_full}")

        self._captured: Tensor | None = None
        self._handle = self.model.trunk_final_norm.register_forward_pre_hook(self._capture)

    def _capture(self, _module: Any, args: tuple) -> None:
        self._captured = args[0]

    def close(self) -> None:
        self._handle.remove()

    @torch.inference_mode()
    def encode(self, sequences: list[str]) -> tuple[Tensor, Tensor]:
        """Retorna ``(pre, post)``, ambos ``[B, L, d_full]``.

        ``pre`` e ``cat(h_up, h_pure)`` antes da RMSNorm: ``pre[..., :d_model]`` = h_up
        (contextual) e ``pre[..., d_model:]`` = h_pure (pointwise, o canal trivial).
        """
        length = len(sequences[0])
        if any(len(s) != length for s in sequences):
            raise ValueError("todas as sequencias do batch precisam ter o mesmo comprimento")
        if length > self.l_max:
            raise ValueError(f"L={length} excede l_max={self.l_max}")

        self._captured = None
        batch = self._encode_dna(sequences, pad_to=length, device=self.device)
        # variant_edit_mask fica None de proposito: ele ligaria o boost `gamma*h_stem` na posicao
        # editada (model.py:365) E o caminho de ancoras da atencao global (backbone.py:230), o que
        # entregaria o sinal de graca em vez de medir a representacao aprendida.
        out = self.model.encode(batch["input_ids"])
        post = out["last_hidden_state"]
        pre = self._captured
        if pre is None:
            raise RuntimeError("o forward pre-hook em trunk_final_norm nao disparou")
        if pre.shape != post.shape:
            raise RuntimeError(f"shape pre {tuple(pre.shape)} != pos {tuple(post.shape)}")
        return pre, post

    def verify_hook(self, sequences: list[str], *, rtol: float = 1e-4, atol: float = 1e-5) -> dict:
        """PROVA que o hook capturou o tensor certo: RMSNorm(pre) tem que reproduzir o `post`.

        Sem isso, "capturamos o pre-norm" seria suposicao.
        """
        pre, post = self.encode(sequences)
        with torch.inference_mode():
            recomputed = self.model.trunk_final_norm(pre)
        max_abs = float((recomputed - post).abs().max())
        ok = bool(torch.allclose(recomputed, post, rtol=rtol, atol=atol))
        return {"ok": ok, "max_abs_diff": max_abs, "shape": tuple(pre.shape)}

    def split(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """``hidden[..., :d_model]`` (h_up, contextual) e ``hidden[..., d_model:]`` (h_pure, trivial)."""
        return hidden[..., : self.d_model], hidden[..., self.d_model :]


class BinReducer:
    """Reduz ``||Delta(q)||`` aos bins do perfil com dois kernels, nao um slice por bin."""

    def __init__(self, bins: list[ProfileBin], window_bp: int, device: torch.device) -> None:
        self.bins = bins
        index = torch.empty(window_bp, dtype=torch.long)
        for i, b in enumerate(bins):
            index[b.start : b.end] = i
        self.index = index.to(device)
        self.n_bins = len(bins)
        self.counts = torch.zeros(self.n_bins, dtype=torch.float64, device=device).index_add_(
            0, self.index, torch.ones(window_bp, dtype=torch.float64, device=device)
        )

    def reduce(self, norms: Tensor) -> tuple[Tensor, Tensor]:
        """``norms`` [L] -> (media, maximo) por bin, ambos [n_bins], em float64."""
        values = norms.detach().to(torch.float64)
        sums = torch.zeros(self.n_bins, dtype=torch.float64, device=values.device).index_add_(
            0, self.index, values
        )
        maxes = torch.full(
            (self.n_bins,), -math.inf, dtype=torch.float64, device=values.device
        ).scatter_reduce_(0, self.index, values, reduce="amax", include_self=True)
        return sums / self.counts, maxes


def reducer_for(window_bp: int, focal_index: int, device: torch.device) -> BinReducer:
    return BinReducer(profile_bins(focal_index=focal_index, window_bp=window_bp), window_bp, device)


@torch.inference_mode()
def extract_site(
    probe: ProbeModel,
    *,
    ref_seq: str,
    focal_index: int,
    reducer: BinReducer,
) -> dict[str, Any]:
    """Extrai as QUATRO bases na posicao focal de um sitio, num unico batch.

    As quatro compartilham o forward de referencia conceitualmente e custam ~1.5x o par
    (ref,alt) sozinho, entregando de graca: o controle de "troca qualquer" (o modelo reage a
    qualquer substituicao ou as reais?) e o contraste alelo-vs-alelo em TODOS os sitios, nao so
    nos multialelicos.
    """
    ref_base = ref_seq[focal_index].upper()
    if ref_base not in BASES:
        raise ValueError(f"base focal {ref_base!r} nao e ACGT")
    seqs = [ref_seq[:focal_index] + b + ref_seq[focal_index + 1 :] for b in BASES]
    if len(seqs) != PROBE_BATCH_SIZE:  # ver PROBE_BATCH_SIZE: invariante, nao conveniencia
        raise RuntimeError(f"batch da sonda tem que ser {PROBE_BATCH_SIZE}, veio {len(seqs)}")

    pre, post = probe.encode(seqs)  # [4, L, 448]
    ref_row = BASES.index(ref_base)
    pre_focal = pre[:, focal_index, :]  # [4, 448]
    post_focal = post[:, focal_index, :]
    hup_pre, hpure_pre = probe.split(pre)

    metrics: dict[str, dict[str, float]] = {}
    profiles: dict[str, dict[str, Tensor]] = {}
    for i, base in enumerate(BASES):
        if i == ref_row:
            continue
        blocks = {
            "post448": (post_focal[ref_row], post_focal[i]),
            "pre448": (pre_focal[ref_row], pre_focal[i]),
            "hup": (hup_pre[ref_row, focal_index], hup_pre[i, focal_index]),
            "hpure": (hpure_pre[ref_row, focal_index], hpure_pre[i, focal_index]),
        }
        row: dict[str, float] = {}
        for name, (a, b) in blocks.items():
            row.update(pair_metrics(a, b).as_dict(name))
        metrics[base] = row

        hup_mean, hup_max = reducer.reduce((hup_pre[i] - hup_pre[ref_row]).norm(dim=-1))
        hpure_mean, hpure_max = reducer.reduce((hpure_pre[i] - hpure_pre[ref_row]).norm(dim=-1))
        post_mean, post_max = reducer.reduce((post[i] - post[ref_row]).norm(dim=-1))
        profiles[base] = {
            "hup_mean": hup_mean, "hup_max": hup_max,
            "hpure_mean": hpure_mean, "hpure_max": hpure_max,
            "post_mean": post_mean, "post_max": post_max,
        }

    def to_np(tensor: Tensor):
        return tensor.detach().to(torch.float32).cpu().numpy().copy()

    return {
        "ref_base": ref_base,
        "ref_row": ref_row,
        "bases": list(BASES),
        "bin_labels": [b.label for b in reducer.bins],
        "bin_directions": [b.direction for b in reducer.bins],
        "pre_focal": to_np(pre_focal),
        "post_focal": to_np(post_focal),
        "metrics": metrics,
        "profiles": {
            base: {k: v.detach().cpu().numpy().copy() for k, v in prof.items()}
            for base, prof in profiles.items()
        },
        "hpure_delta_focal": {
            base: to_np(hpure_pre[BASES.index(base), focal_index] - hpure_pre[ref_row, focal_index])
            for base in BASES
            if base != ref_base
        },
    }


def check_hpure_invariant(results: list[dict[str, Any]], *, atol: float = 1e-5) -> dict[str, Any]:
    """Verifica a previsao analitica de ``Delta h_pure``, que valida o pipeline inteiro.

    (a) Fora da posicao focal, ``Delta h_pure`` e EXATAMENTE zero (``purity`` e pointwise e so a
        base focal mudou).
    (b) No focal, ``Delta h_pure`` depende SO do par ordenado (ref,alt) -- mesmo vetor em
        qualquer locus, contexto ou janela.

    Desvio aqui = bug de extracao (janela desalinhada, hook errado, base trocada no lugar errado),
    nao um achado sobre o modelo.
    """
    import numpy as np

    by_substitution: dict[str, list] = {}
    off_focal_max = 0.0
    for res in results:
        keep = [i for i, d in enumerate(res["bin_directions"]) if d != "focal"]
        for base, prof in res["profiles"].items():
            if keep:
                off_focal_max = max(off_focal_max, float(np.max(prof["hpure_max"][keep])))
            by_substitution.setdefault(f"{res['ref_base']}>{base}", []).append(
                res["hpure_delta_focal"][base]
            )

    spreads = {}
    for key, vectors in by_substitution.items():
        if len(vectors) < 2:
            continue
        stack = np.stack(vectors)
        spreads[key] = float(np.abs(stack - stack[0]).max())
    worst = max(spreads.values(), default=0.0)
    return {
        "off_focal_max_abs": off_focal_max,
        "off_focal_ok": off_focal_max <= atol,
        "substitution_spread": spreads,
        "substitution_spread_max": worst,
        "substitution_ok": worst <= atol,
        "n_substitutions": len(spreads),
    }
