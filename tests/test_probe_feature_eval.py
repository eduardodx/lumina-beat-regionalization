"""Validacao ponta-a-ponta do harness de avaliacao, com dados sinteticos de resposta conhecida.

Precisa de numpy + pandas (roda no notebook com ``$PY``, ou em qualquer venv cientifico):
    PYTHONPATH=. python tests/test_probe_feature_eval.py

O TESTE QUE IMPORTA
-------------------
Construimos tres blocos de features com comportamento conhecido:

  ``signal``  correlacionado com o rotulo        -> tem que dar AUROC bem acima de 0.5
  ``noise``   aleatorio puro                     -> ~0.5
  ``locus``   ONE-HOT da unidade de bloqueio       -> **~0.5 sob o protocolo**

O ``locus`` e o teste critico. E um one-hot da unidade -- a feature de memorizacao canonica: um
modelo LINEAR sobre ela decora exatamente o vies de cada locus do treino. (Um vetor aleatorio por
unidade NAO serviria: um probe linear nao consegue decorar vieses arbitrarios a partir de 16 dims
sem relacao linear com eles -- descobrimos isso quando a contraprova falhou.) Como o protocolo garante que nenhuma unidade cruza papeis,
as unidades do teste sao ineditas e a memorizacao nao transfere -- AUROC tem que desabar para o
acaso. Se este teste passar com AUROC alto, o harness esta vazando e todas as ablacoes que ele
produzir sao invalidas.

Para provar que o colapso vem do BLOQUEIO e nao de a feature ser inutil, o mesmo bloco e avaliado
tambem com folds embaralhados (unidades cruzando papeis): ali ele TEM que subir.
"""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    print("SKIP: precisa de numpy + pandas")
    sys.exit(0)

from scripts.probe_feature_eval import main, ridge_scores  # noqa: E402

PANELS = ("missense", "splice", "noncoding")


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS.

    Ja fomos enganados por isto uma vez: um teste que retornava cedo aparecia como PASS e a
    ausencia de cobertura passou despercebida. O runner conta os skips separado, e o codigo de
    saida so ignora skips -- nunca os soma aos aprovados.
    """


def build(tmp: Path, *, n_units=150, per_unit=12, seed=7, shuffle_folds=False):
    """Release + features sinteticos. Rotulo depende do sitio (tenta a memorizacao) e de um sinal."""
    rng = np.random.default_rng(seed)
    rows, feats_signal, feats_noise, feats_locus = [], [], [], []
    unit_bias = {u: rng.normal() * 2.2 for u in range(n_units)}

    for u in range(n_units):
        fold = u % 5  # a UNIDADE define o fold -> nenhuma unidade cruza papeis
        for j in range(per_unit):
            latent = rng.normal()
            logit = 1.6 * latent + unit_bias[u]
            label = int(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append({
                "variant_id": f"var:{u:04d}:{j:02d}",
                "binary_label": label,
                # tier e painel PRECISAM ser independentes: se coincidirem, um painel inteiro
                # pode cair fora do gold e a macro fica indefinida em toda execucao
                "label_tier": "consensus" if j >= per_unit - 3 else "gold",
                "primary_panel": PANELS[j % 3],
                "core_fold": fold,
                "overlap_cluster_id": f"ovl:{u:04d}",
            })
            feats_signal.append(np.concatenate([[latent], rng.normal(size=7) * 0.3]))
            feats_noise.append(rng.normal(size=8))
            onehot = np.zeros(n_units, dtype=np.float32)
            onehot[u] = 1.0
            feats_locus.append(onehot)

    df = pd.DataFrame(rows)
    if shuffle_folds:
        # "sem bloqueio": fold aleatorio POR VARIANTE e unidade unica por variante. A assercao de
        # vazamento continua satisfeita (unidades de tamanho 1 nao cruzam papeis), mas o mesmo
        # locus aparece nos dois lados -- que e exatamente o cenario que o bloqueio real impede.
        df["core_fold"] = rng.integers(0, 5, len(df))
        df["overlap_cluster_id"] = df["variant_id"]

    df[["variant_id", "binary_label", "label_tier"]].to_parquet(tmp / "pb_examples.parquet", index=False)
    df[["variant_id", "primary_panel"]].assign(panel_role="discrimination").to_parquet(
        tmp / "pb_panels.parquet", index=False)
    df[["variant_id", "core_fold", "overlap_cluster_id"]].assign(
        gene_transfer_fold=df["core_fold"], gene_transfer_group_id=df["overlap_cluster_id"],
    ).to_parquet(tmp / "pb_partitions.parquet", index=False)

    np.savez(tmp / "probe_features.npz",
             variant_id=np.array(df["variant_id"].tolist(), dtype="U40"),
             blk_signal=np.array(feats_signal, dtype=np.float32),
             blk_noise=np.array(feats_noise, dtype=np.float32),
             blk_locus=np.array(feats_locus, dtype=np.float32))
    return df


def build_nonlinear(tmp: Path, *, n_units=200, per_unit=12, seed=11):
    """Rotulo = XOR do sinal das duas primeiras features. Ridge NAO consegue; MLP consegue."""
    rng = np.random.default_rng(seed)
    rows, feats = [], []
    for u in range(n_units):
        for j in range(per_unit):
            a, b = rng.normal(), rng.normal()
            label = int((a * b) > 0)
            rows.append({
                "variant_id": f"var:{u:04d}:{j:02d}", "binary_label": label,
                "label_tier": "consensus" if j >= per_unit - 3 else "gold",
                "primary_panel": PANELS[j % 3], "core_fold": u % 5,
                "overlap_cluster_id": f"ovl:{u:04d}",
            })
            feats.append([a, b, rng.normal() * 0.2, rng.normal() * 0.2])
    df = pd.DataFrame(rows)
    df[["variant_id", "binary_label", "label_tier"]].to_parquet(tmp / "pb_examples.parquet", index=False)
    df[["variant_id", "primary_panel"]].assign(panel_role="discrimination").to_parquet(
        tmp / "pb_panels.parquet", index=False)
    df[["variant_id", "core_fold", "overlap_cluster_id"]].assign(
        gene_transfer_fold=df["core_fold"], gene_transfer_group_id=df["overlap_cluster_id"],
    ).to_parquet(tmp / "pb_partitions.parquet", index=False)
    np.savez(tmp / "probe_features.npz",
             variant_id=np.array(df["variant_id"].tolist(), dtype="U40"),
             blk_xor=np.array(feats, dtype=np.float32))


def run(tmp: Path, **kw) -> dict:
    out = tmp / "eval.json"
    feats = kw.get("features") or [tmp / "probe_features.npz"]
    argv = ["--features", *[str(f) for f in feats], "--release-root", str(tmp),
            "--out", str(out), "--train-tiers", kw.get("train_tiers", "gold")]
    if kw.get("configs"):
        cfg = tmp / "cfg.json"
        cfg.write_text(json.dumps(kw["configs"]), encoding="utf-8")
        argv += ["--configs", str(cfg)]
    if kw.get("mlp"):
        argv.append("--mlp")
    rc = main(argv)
    assert rc == 0, f"main retornou {rc}"
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------


def test_signal_beats_noise():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        r = run(tmp)["configs"]
        sig, noi = r["signal"]["macro"], r["noise"]["macro"]
        assert sig > 0.65, f"o bloco com sinal deveria ser facilmente detectado, deu {sig:.3f}"
        assert 0.40 < noi < 0.60, f"ruido puro deveria ficar no acaso, deu {noi:.3f}"
        assert sig > noi + 0.10, f"sinal ({sig:.3f}) mal separa de ruido ({noi:.3f})"


def test_blocking_is_what_kills_locus_memorisation():
    """O TESTE CRITICO, e a sua contraprova, no mesmo lugar.

    A MESMA feature de memorizacao e avaliada com e sem bloqueio. Sob o protocolo ela tem que cair
    para o acaso; sem bloqueio ela tem que subir. Sao as duas metades da afirmacao: se so a primeira
    passasse, poderia ser que a feature fosse simplesmente inutil.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        blocked = run(tmp)["configs"]["locus"]["macro"]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp, shuffle_folds=True)
        unblocked = run(tmp)["configs"]["locus"]["macro"]

    assert 0.40 < blocked < 0.60, (
        f"one-hot de locus deu macro {blocked:.3f} sob folds bloqueados -- o harness esta VAZANDO; "
        "toda ablacao feita com ele seria invalida"
    )
    assert unblocked > blocked + 0.08, (
        f"sem bloqueio a memorizacao deveria subir claramente: {unblocked:.3f} vs {blocked:.3f}. "
        "Se nao sobe, o teste acima nao prova que foi o BLOQUEIO que matou a feature."
    )
    print(f"        (bloqueado {blocked:.3f} · sem bloqueio {unblocked:.3f})")


def test_protocol_metadata_is_recorded():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        rep = run(tmp)
        assert rep["track"] == "core_locus"
        assert rep["train_tiers"] == "gold"
        assert rep["probe"] == "ridge"
        cfg = rep["configs"]["signal"]
        assert cfg["n_runs"] == 5, "as cinco execucoes do protocolo tem que rodar"
        for r in cfg["per_run"]:
            assert r["sizes"]["train"] > 0 and r["sizes"]["test"] > 0
            assert r["val_macro"] is not None, "lambda tem que ser escolhido na validation"


def test_test_set_is_gold_only():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        ex = pd.read_parquet(tmp / "pb_examples.parquet").set_index("variant_id")
        pt = pd.read_parquet(tmp / "pb_partitions.parquet").set_index("variant_id")
        with np.load(tmp / "probe_features.npz") as z:
            vid = [str(v) for v in z["variant_id"]]
        from eval.embedding_probe.protocol import cross_fitted_split
        folds = [int(pt.loc[v, "core_fold"]) for v in vid]
        tiers = [ex.loc[v, "label_tier"] for v in vid]
        for run_id in range(5):
            s = cross_fitted_split(folds, tiers, run_id=run_id, train_tiers=("gold",))
            assert all(tiers[i] == "gold" for i in s.test)
            assert all(tiers[i] == "gold" for i in s.validation)


def test_standardisation_is_fit_on_train_only():
    """Uma dimensao constante no TREINO mas variavel no teste nao pode explodir a escala."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); df = build(tmp)
        with np.load(tmp / "probe_features.npz") as f:
            z = dict(f)
        const = np.zeros((len(df), 4), dtype=np.float32)
        const[df["core_fold"].to_numpy() == 0] = 5.0  # so varia no fold 0
        z["blk_const"] = const
        np.savez(tmp / "probe_features.npz", **z)
        r = run(tmp)["configs"]["const"]
        assert r["macro"] is not None and np.isfinite(r["macro"]), "escala degenerada quebrou o probe"


def test_ridge_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        a, b = run(tmp)["configs"]["signal"]["macro"], run(tmp)["configs"]["signal"]["macro"]
        assert a == b, f"ridge deveria ser deterministico: {a} != {b}"


def test_high_dim_uses_dual_path():
    """d > n: a rotina tem que cair no Gram dual sem quebrar."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp, n_units=40, per_unit=12)
        with np.load(tmp / "probe_features.npz") as f:
            z = dict(f)
        rng = np.random.default_rng(3)
        z["blk_wide"] = rng.normal(size=(len(z["variant_id"]), 400)).astype(np.float32)
        np.savez(tmp / "probe_features.npz", **z)
        r = run(tmp)["configs"]["wide"]
        assert r["n_dims"] == 400 and r["macro"] is not None


def test_ridge_ignores_directions_without_support_in_training():
    """Regressao: colunas sem suporte no treino nao podem influenciar o score de teste.

    Bug real encontrado na revisao: com features de posto deficiente o Gram tem autovalores ~0, e
    ``proj/(s + lambda)`` amplificava ruido de ponto flutuante no espaco nulo por 1/lambda. Como
    cada linha de teste ativa uma direcao nula diferente, os scores deixavam de ser constantes e o
    resultado passava a depender da implementacao de LAPACK (medimos 0.511 e 0.580 no mesmo dado,
    em maquinas diferentes). A correcao zera essas direcoes.
    """
    rng = np.random.default_rng(5)
    n, d = 300, 20
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + rng.normal(size=n) * 0.5 > 0).astype(np.float64)
    Xe = rng.normal(size=(40, d))
    lams = [n * 1e-5, n * 1.0]

    base = ridge_scores(X, y, [Xe], lams)
    # acrescenta 15 colunas que sao ZERO no treino mas variam no teste -- exatamente o caso
    # degenerado (categorias ausentes do treino, colunas constantes, colineares)
    Xz = np.concatenate([X, np.zeros((n, 15))], axis=1)
    Xez = np.concatenate([Xe, rng.normal(size=(40, 15))], axis=1)
    ext = ridge_scores(Xz, y, [Xez], lams)

    for lam in lams:
        diff = np.abs(base[lam][0] - ext[lam][0]).max()
        assert diff < 1e-9, (
            f"lambda={lam}: colunas sem suporte no treino mudaram o score em {diff:.2e} -- "
            "o espaco nulo esta voltando a ser amplificado"
        )


def test_mlp_probe_learns_what_ridge_cannot():
    """Se o MLP nao superar o ridge num sinal puramente nao-linear, ele esta subtreinado -- e a
    pergunta 'as cabecas lineares ajudam um modelo nao-linear?' ficaria sem instrumento."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise Skip("sem torch") from None
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build_nonlinear(tmp)
        ridge = run(tmp)["configs"]["xor"]["macro"]
        mlp = run(tmp, mlp=True)["configs"]["xor"]["macro"]
    assert 0.42 < ridge < 0.58, f"ridge deveria ficar no acaso num XOR, deu {ridge:.3f}"
    assert mlp > ridge + 0.15, (
        f"o probe MLP deu {mlp:.3f} contra {ridge:.3f} do ridge -- nao esta aprendendo o "
        "nao-linear, entao nao serve para a pergunta das cabecas"
    )
    print(f"        (ridge {ridge:.3f} · mlp {mlp:.3f})")


def _split_features(tmp: Path, *, shuffle_second=False, rename=None):
    """Quebra o npz sintetico em dois arquivos, para exercitar a uniao."""
    with np.load(tmp / "probe_features.npz") as z:
        d = {k: z[k] for k in z.files}
    vid = d["variant_id"]
    np.savez(tmp / "a.npz", variant_id=vid, blk_signal=d["blk_signal"])
    order = np.arange(len(vid))
    if shuffle_second:
        order = np.random.default_rng(3).permutation(order)
    name = rename or "blk_noise"
    np.savez(tmp / "b.npz", variant_id=vid[order], **{name: d["blk_noise"][order]})
    return [tmp / "a.npz", tmp / "b.npz"]


def test_merging_feature_files_matches_a_single_file():
    """Unir dois npz tem de dar exatamente o mesmo que um npz com os dois blocos."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        cfg = {"juntos": ["signal", "noise"]}
        single = run(tmp, configs=cfg)["configs"]["juntos"]["macro"]
        merged = run(tmp, features=_split_features(tmp), configs=cfg)["configs"]["juntos"]["macro"]
        assert single == merged, f"uniao deu {merged:.6f}, arquivo unico deu {single:.6f}"


def test_merge_aligns_by_variant_id_not_by_row_order():
    """O segundo arquivo vem embaralhado: se a uniao alinhasse por posicao, o sinal se perderia."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        cfg = {"juntos": ["signal", "noise"]}
        ref = run(tmp, features=_split_features(tmp), configs=cfg)["configs"]["juntos"]["macro"]
        shuf = run(tmp, features=_split_features(tmp, shuffle_second=True),
                   configs=cfg)["configs"]["juntos"]["macro"]
        assert ref == shuf, (
            f"embaralhar as linhas do segundo arquivo mudou o resultado ({shuf:.6f} vs {ref:.6f}) "
            "-- a uniao esta alinhando por posicao, nao por variant_id"
        )


def test_merge_refuses_duplicate_block_names():
    """Dois arquivos com o mesmo nome de bloco: sobrescrever em silencio trocaria o experimento."""
    from scripts.probe_feature_eval import load_features
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        paths = _split_features(tmp, rename="blk_signal")
        try:
            load_features(paths)
        except SystemExit as exc:
            assert "mais de um arquivo" in str(exc), str(exc)
        else:
            raise AssertionError("nome de bloco repetido deveria ser erro")


def test_merge_keeps_only_the_intersection_of_ids():
    from scripts.probe_feature_eval import load_features
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        a, b = _split_features(tmp)
        with np.load(b) as z:
            keep = slice(0, 40)
            np.savez(tmp / "b.npz", variant_id=z["variant_id"][keep], blk_noise=z["blk_noise"][keep])
        ids, blocks = load_features([a, tmp / "b.npz"])
        assert len(ids) == 40, f"esperava a interseccao (40), veio {len(ids)}"
        assert all(v.shape[0] == 40 for v in blocks.values())


def test_lambda_grid_reaches_far_enough_to_regularise():
    """Uma feature COM sinal nao pode selecionar o teto do grid.

    O teto tem duas leituras. Num bloco de puro ruido, regularizar ate a media e a escolha certa e
    o teto sera sempre selecionado -- nenhum grid finito muda isso, entao testar ruido nao mede
    nada. Ja num bloco informativo, colar no teto significa que o grid acabou antes do otimo e o
    numero reportado e artefato do grid. Esse e o caso que importa.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        res = run(tmp, configs={"signal": ["signal"]})["configs"]["signal"]
        chosen = [c for c in res["lambdas_chosen"] if c is not None]
        assert chosen, "nenhum lambda registrado"
        tops = [r["sizes"]["train"] * (10 ** 6) for r in res["per_run"]]
        at_top = sum(1 for c, t in zip(chosen, tops) if c >= t * 0.99)
        assert at_top == 0, (
            f"bloco com sinal selecionou o teto do grid em {at_top}/{len(chosen)} execucoes -- "
            "o grid acabou antes do otimo e os numeros sao artefato dele"
        )
        assert res["macro"] > 0.65, f"o bloco com sinal deveria pontuar, deu {res['macro']:.3f}"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed, skipped = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    ran = len(tests) - failed - len(skipped)
    tail = f"  |  {len(skipped)} PULADO(S), sem cobertura: {', '.join(skipped)}" if skipped else ""
    print(f"\n{ran}/{len(tests) - len(skipped)} passaram{tail}")
    sys.exit(1 if failed else 0)
