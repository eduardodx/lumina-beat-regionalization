"""Testes da extracao rica. Precisa de torch (ha no notebook e em qualquer venv com torch)::

    PYTHONPATH=. "$PY" tests/test_embedding_probe_rich.py

O TESTE QUE IMPORTA
-------------------
``MidStackTaps`` usa forward PRE-hooks porque as camadas de atencao local do R03 nao sao chamadas
como modulos (``x = x + attn(norm(x), None)``) -- um forward hook nelas jamais dispararia, e o sweep
de camada ficaria com buracos silenciosos. O teste monta um mid-stack falso com a MESMA estrutura
(mamba / local-ModuleDict / sparse, registers prependados) e verifica que cada tomada captura
exatamente o estado intermediario esperado, comparando contra o valor calculado a mao.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    print("SKIP: precisa de torch")
    sys.exit(0)

from eval.embedding_probe.rich import (  # noqa: E402
    BASES,
    LINEAR_HEADS,
    MLP_HEADS,
    SUBSTITUTIONS,
    MidStackTaps,
    head_readouts,
    mid_index,
    pooled,
    reverse_complement,
    substitution_onehot,
)


@contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


class FakeMidStack(nn.Module):
    """Mesma estrutura do LuminaMidStack: registers prependados e os tres tipos de camada,
    com a local NAO sendo chamada como modulo -- que e a armadilha que os taps precisam vencer."""

    def __init__(self, d=8, n_reg=3, kinds=("mamba", "mamba", "local", "sparse", "mamba")):
        super().__init__()
        self.n_register_tokens = n_reg
        self.register_tokens = nn.Parameter(torch.randn(n_reg, d) * 0.02)
        self.layer_kinds = list(kinds)
        self.layers = nn.ModuleList()
        for kind in kinds:
            if kind == "local":
                self.layers.append(nn.ModuleDict({"norm": nn.LayerNorm(d), "attn": nn.Linear(d, d)}))
            else:
                self.layers.append(nn.Linear(d, d))

    def forward(self, x, edit_mid_mask=None):
        n_reg = self.n_register_tokens
        reg = self.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([reg, x], dim=1)
        self.trace = [x]                       # gabarito: o estado antes de cada camada
        for kind, layer in zip(self.layer_kinds, self.layers):
            if kind == "mamba":
                x = layer(x)
            elif kind == "local":
                x = x + layer["attn"](layer["norm"](x))
            else:
                x = layer(x)
            self.trace.append(x)
        return x[:, n_reg:]


class FakeModel(nn.Module):
    def __init__(self, d=8, **kw):
        super().__init__()
        self.mid_stack = FakeMidStack(d=d, **kw)


# --------------------------------------------------------------------------------------------
# taps do mid-stack
# --------------------------------------------------------------------------------------------


def test_taps_fire_on_every_layer_boundary():
    model = FakeModel()
    taps = MidStackTaps(model)
    try:
        out = model.mid_stack(torch.randn(2, 12, 8))
        states = taps.states
    finally:
        taps.close()
    assert len(states) == taps.n_taps == len(model.mid_stack.layer_kinds) + 1
    assert out.shape == (2, 12, 8), "o mid_hidden_state sai SEM os registers"


def test_taps_capture_the_exact_intermediate_states():
    """Cada tomada tem que ser bit-identica ao gabarito calculado pelo proprio forward."""
    model = FakeModel()
    taps = MidStackTaps(model)
    try:
        model.mid_stack(torch.randn(3, 16, 8))
        states, trace = taps.states, model.mid_stack.trace
    finally:
        taps.close()
    assert len(states) == len(trace)
    for i, (got, want) in enumerate(zip(states, trace)):
        assert torch.equal(got, want), f"tomada {i} nao bate com o estado real do forward"


def test_taps_capture_the_state_after_a_local_layer():
    """O caso critico: a camada local nao e chamada como modulo. Se o desenho fosse com forward
    hooks, o estado depois dela sumiria e o sweep teria um buraco invisivel."""
    kinds = ("mamba", "local", "mamba")
    model = FakeModel(kinds=kinds)
    taps = MidStackTaps(model)
    try:
        model.mid_stack(torch.randn(2, 8, 8))
        states, trace = taps.states, model.mid_stack.trace
    finally:
        taps.close()
    local_pos = kinds.index("local")
    assert torch.equal(states[local_pos + 1], trace[local_pos + 1]), \
        "o estado APOS a atencao local nao foi capturado"


def test_taps_include_registers_at_the_front():
    n_reg, seq = 3, 12
    model = FakeModel(n_reg=n_reg)
    taps = MidStackTaps(model)
    try:
        model.mid_stack(torch.randn(2, seq, 8))
        states = taps.states
    finally:
        taps.close()
    for s in states:
        assert s.shape[1] == n_reg + seq, "os registers tem que estar presentes dentro do stack"
    assert taps.n_registers == n_reg


def test_taps_raise_if_forward_did_not_run():
    model = FakeModel()
    taps = MidStackTaps(model)
    try:
        with assert_raises(RuntimeError):
            _ = taps.states
    finally:
        taps.close()


def test_close_removes_every_hook():
    model = FakeModel()
    taps = MidStackTaps(model)
    model.mid_stack(torch.randn(1, 8, 8))
    taps.close()
    taps.reset()
    model.mid_stack(torch.randn(1, 8, 8))
    with assert_raises(RuntimeError):
        _ = taps.states


def test_describe_labels_every_tap():
    model = FakeModel()
    taps = MidStackTaps(model)
    labels = taps.describe()
    taps.close()
    assert len(labels) == taps.n_taps
    assert labels[0] == "entrada"
    assert "local" in labels[3], labels


# --------------------------------------------------------------------------------------------
# cabecas
# --------------------------------------------------------------------------------------------


class HeadModel(nn.Module):
    """So as cabecas, com as mesmas formas do R03."""

    def __init__(self, d_full=448):
        super().__init__()
        self.mlm_head = nn.Linear(d_full, 4)
        self.conservation_scalar_head = nn.Linear(d_full, 3)
        self.conservation_bin_head = nn.Linear(d_full, 16)
        self.region_head = nn.Linear(d_full, 5)
        self.counterfactual_snv_head = nn.Linear(d_full, 32)
        self.population_af_head = nn.Linear(d_full, 4)
        self.population_observed_head = nn.Linear(d_full, 4)
        self.splice_class_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, 5))
        self.splice_distance_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, 1))
        self.missense_severity_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, 4))


def test_linear_head_identity_is_exact():
    """A base da proposta: cabeca(alt) - cabeca(ref) == W.Delta, com o bias cancelando."""
    torch.manual_seed(0)
    m = HeadModel()
    ref, alt = torch.randn(5, 448), torch.randn(5, 448)
    got = head_readouts(m, ref, alt)
    offset = 0
    for name in LINEAR_HEADS:
        head = getattr(m, name)
        want = head(alt) - head(ref)
        n = want.shape[-1]
        chunk = got["linear"][:, offset:offset + n]
        assert torch.allclose(chunk, want, atol=1e-4), f"{name}: W.Delta nao reproduz a diferenca"
        offset += n
    assert offset == got["linear"].shape[-1] == 68


def test_mlp_heads_are_not_in_the_linear_span():
    """Contraprova: para as MLP a identidade NAO vale -- por isso rodamos as duas torres."""
    torch.manual_seed(1)
    m = HeadModel()
    ref, alt = torch.randn(4, 448), torch.randn(4, 448)
    delta = alt - ref
    head = m.splice_class_head
    linear_guess = delta @ head[0].weight.T @ head[2].weight.T  # o que a identidade daria
    real = head(alt) - head(ref)
    assert not torch.allclose(linear_guess, real, atol=1e-2), \
        "se a identidade valesse para a MLP, nao precisariamos das duas torres"


def test_mlp_block_has_ten_dims():
    m = HeadModel()
    got = head_readouts(m, torch.randn(3, 448), torch.randn(3, 448))
    assert got["mlp"].shape[-1] == 5 + 1 + 4 == 10
    assert got["ref"].shape[-1] == 68 + 10


def test_head_readouts_survive_a_missing_optional_head():
    m = HeadModel()
    m.missense_severity_head = None
    got = head_readouts(m, torch.randn(2, 448), torch.randn(2, 448))
    assert got["mlp"].shape[-1] == 6


# --------------------------------------------------------------------------------------------
# helpers puros
# --------------------------------------------------------------------------------------------


def test_reverse_complement_is_an_involution():
    seq = "ACGTTGCAAGGCT"
    assert reverse_complement(reverse_complement(seq)) == seq
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AAAA") == "TTTT"


def test_reverse_complement_maps_position_i_to_L_minus_1_minus_i():
    """A relacao que a media RC depende: emb_rc[L-1-i] corresponde a emb[i]."""
    seq = "ACGTACGTAC"
    rc = reverse_complement(seq)
    from eval.embedding_probe.rich import COMPLEMENT
    for i, base in enumerate(seq):
        assert rc[len(seq) - 1 - i] == COMPLEMENT[base]


def test_mid_index_maps_focal_to_quarter_resolution():
    assert mid_index(2047) == 511
    assert mid_index(8191) == 2047
    assert mid_index(0) == 0


def test_substitution_onehot_is_exactly_the_hpure_information():
    vec = substitution_onehot("A", "G")
    assert len(vec) == 16 and sum(vec) == 2.0
    assert vec[BASES.index("A")] == 1.0
    assert vec[4 + SUBSTITUTIONS.index("A>G")] == 1.0
    assert len(SUBSTITUTIONS) == 12
    assert len(set(SUBSTITUTIONS)) == 12
    # o mesmo par sempre no mesmo lugar, em qualquer locus -- e isso que h_pure codifica
    assert substitution_onehot("A", "G") == substitution_onehot("A", "G")
    assert substitution_onehot("A", "G") != substitution_onehot("A", "C")


def test_pooled_respects_the_radius_and_the_window_edges():
    delta = torch.arange(20, dtype=torch.float32).view(1, 20, 1)
    assert torch.allclose(pooled(delta, 10, 0), torch.tensor([[10.0]]))
    assert torch.allclose(pooled(delta, 10, 2), torch.tensor([[10.0]]))   # media de 8..12
    assert torch.allclose(pooled(delta, 1, 5), torch.tensor([[3.0]]))     # cortado em 0..6
    assert torch.allclose(pooled(delta, 10, 2, reduce="max"), torch.tensor([[12.0]]))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passaram")
    sys.exit(1 if failed else 0)
