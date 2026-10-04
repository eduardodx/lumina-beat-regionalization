"""Fase 1, passo 4 (leituras) sobre o cenario sintetico do passo 3, com membership e criticas sinteticos.

No Windows nao ha o pacote `mosaic`: as coortes oficiais, o limiar (`mcc_max_threshold`, a mesma regra de
`calibrate_threshold`) e os contrastes sao copias minimas. No .venv do Mosaic os testes usam as funcoes reais.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha import metricas  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from tests import test_fase1_bracos as passo3  # noqa: E402

try:
    from mosaic.comparator_eval import contrasts as _contrasts
    from mosaic.comparator_eval import primary_cohorts as _primarias
    from mosaic.comparator_eval import stats as _stats
    TEM_MOSAIC = True
except ImportError:
    TEM_MOSAIC = False
try:
    from scipy.stats import beta as _beta
    TEM_SCIPY = True
except ImportError:
    TEM_SCIPY = False


# ------------------------------------------------------------------------------------- copias minimas do Mosaic

def limiar_copiado(labels, scores, *, validation_ids, test_ids):
    """`stats.mcc_max_threshold`: MCC maximo, depois especificidade, depois o maior limiar."""
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if s.size == 0 or len(set(y.tolist())) < 2:
        return SimpleNamespace(threshold=None, status="sem duas classes")
    distintos, inverso = np.unique(s, return_inverse=True)
    candidatos = np.concatenate(([np.nextafter(distintos[0], -np.inf)], distintos, [np.nextafter(distintos[-1], np.inf)]))
    pos = np.bincount(inverso, weights=(y == 1), minlength=distintos.size).astype(np.int64)
    neg = np.bincount(inverso, weights=(y == 0), minlength=distintos.size).astype(np.int64)
    tp = np.concatenate(([pos.sum()], np.cumsum(pos[::-1])[::-1], [0]))
    fp = np.concatenate(([neg.sum()], np.cumsum(neg[::-1])[::-1], [0]))
    fn, tn = pos.sum() - tp, neg.sum() - fp
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = np.where(den == 0, 0.0, (tp * tn - fp * fn) / np.sqrt(den.astype(float)))
    return SimpleNamespace(threshold=float(candidatos[np.lexsort((candidatos, tn / neg.sum(), mcc))[-1]]), status="ok")


def core_cohort_copiado(release, window):
    view = pd.read_parquet(release / bracos.VISTA, columns=["variant_id", "sequence_eligible", "overlap_cluster_id",
                                                             "core_fold", "core_purged_runs"])
    ex = pd.read_parquet(release / "clinical-variants.parquet", columns=["variant_id", "binary_label", "label_tier"])
    panels = pd.read_parquet(release / "evaluation-panels.parquet", columns=["variant_id", "primary_panel"])
    gold = ex.merge(panels, on="variant_id").rename(columns={"binary_label": "y", "primary_panel": "panel"})
    gold = gold.merge(view, on="variant_id")
    gold = gold[gold["label_tier"] == "gold"]
    testes, validacoes = [], []
    for run in range(5):
        purgada = np.array([run in (r if r is not None else ()) for r in gold["core_purged_runs"]])
        t = gold[(gold["core_fold"] == run) & gold["sequence_eligible"]]
        v = gold[(gold["core_fold"] == (run + 1) % 5) & ~purgada]
        testes.append(t.assign(run=run, role="test", unit=t["overlap_cluster_id"]))
        validacoes.append(v.assign(run=run, role="validation", unit=v["overlap_cluster_id"]))
    colunas = ["variant_id", "run", "role", "panel", "y", "unit"]
    return pd.concat(testes)[colunas], pd.concat(validacoes)[colunas]


def regional_copiado(release, window=4096):
    teste, validacao = core_cohort_copiado(release, window)
    ann = pd.read_parquet(release / "variant-annotations.parquet", columns=["variant_id", "present_abraom"])
    presentes = set(ann.loc[ann["present_abraom"].fillna(False).astype(bool), "variant_id"])
    return teste[teste["variant_id"].isin(presentes)], validacao


def contraste_copiado(test, validation, *, a, b, aggregate, n_replicates, seed):
    linhas = []
    primarios = test[test["panel"].isin(metricas.PAINEIS_DE_DISCRIMINACAO)]
    valores = {}
    for s in (a, b):
        por_painel = [metricas.auprc(g[s].to_numpy(), g["y"].to_numpy()) for _, g in primarios.groupby("panel")]
        valores[s] = None if any(v is None for v in por_painel) else float(np.mean(por_painel))
    for parte, estimativa in (("a", valores[a]), ("b", valores[b]),
                              ("delta", None if None in valores.values() else valores[a] - valores[b])):
        linhas.append({"endpoint": "macro_auprc", "part": parte, "estimate": estimativa, "ci95_low": None,
                       "ci95_high": None})
        linhas.append({"endpoint": "mcc", "part": parte, "estimate": None, "ci95_low": None, "ci95_high": None})
    return linhas


def contraste_de_chamadas_copiado(test, *, a, b, n_replicates, seed):
    pa = leituras.endpoints_de_chamada(test["y"].to_numpy(), test[a].to_numpy())
    pb = leituras.endpoints_de_chamada(test["y"].to_numpy(), test[b].to_numpy())
    return [{"endpoint": e, "part": parte, "estimate": v, "ci95_low": None, "ci95_high": None}
            for e in pa for parte, v in (("a", pa[e]), ("b", pb[e]), ("delta", pa[e] - pb[e]))]


def _ferramentas(referencia):
    def verificar(raiz, config_dir=None):
        return {"release_identity_hash": referencia["release_identity_hash"], "protocol_hash": referencia["protocol_hash"]}
    if TEM_MOSAIC:
        return {"verificar_identidade": verificar, "calibrate_threshold": _stats.calibrate_threshold,
                "core_cohort": _primarias.core_cohort, "regional_clinical_cohort": _primarias.regional_clinical_cohort,
                "paired_contrast": _contrasts.paired_contrast, "paired_call_contrast": _contrasts.paired_call_contrast}
    return {"verificar_identidade": verificar, "calibrate_threshold": limiar_copiado, "core_cohort": core_cohort_copiado,
            "regional_clinical_cohort": regional_copiado, "paired_contrast": contraste_copiado,
            "paired_call_contrast": contraste_de_chamadas_copiado}


# -------------------------------------------------------------------------------------------- dados sinteticos

def _membership(df: pd.DataFrame) -> pd.DataFrame:
    """Os dois estudos com pareamento 1:1 exato em rotulo x painel, no esquema do Mosaic."""
    linhas = []

    def linha(r, estudo, papel, par):
        return {"variant_id": r["variant_id"], "study_id": estudo, "member_role": papel, "matched_variant_id": par,
                "stratum": f"{r['binary_label']}|{r['primary_panel']}|bin", "label_tier": r["label_tier"],
                "binary_label": int(r["binary_label"]), "primary_panel": r["primary_panel"], "gnomad_af_bin": "bin",
                "overlap_cluster_id": f"c32_{r['variant_id']}", "core_fold": int(r["core_fold"]),
                "br_lab_any": False, "present_abraom": bool(r["present_abraom"])}

    def parear(estudo, casos, livres):
        for _, caso in casos.iterrows():
            opcoes = livres[(livres["binary_label"] == caso["binary_label"])
                            & (livres["primary_panel"] == caso["primary_panel"])]
            if opcoes.empty:
                linhas.append(linha(caso, estudo, "unmatched_case", None))
                continue
            controle = opcoes.iloc[0]
            livres = livres.drop(controle.name)
            linhas.extend([linha(caso, estudo, "case", controle["variant_id"]),
                           linha(controle, estudo, "control", caso["variant_id"])])

    gold = df[df["label_tier"] == "gold"]
    parear("br_population_observed", gold[gold["present_abraom"]], gold[~gold["present_abraom"]])
    consenso = df[df["label_tier"] == "consensus"]
    marcados = np.arange(len(consenso)) % 2 == 0   # o papel de br_lab_any: metade dos consensus vira caso
    parear("br_clinical_evidence", consenso[marcados], consenso[~marcados])
    return pd.DataFrame(linhas)


def _criticas(df: pd.DataFrame) -> list[dict]:
    alvo = df[(df["binary_label"] == 1) & df["present_abraom"] & df["sequence_eligible"]].iloc[0]
    return [{"gene": "SINT", "hgvs": "critica no release",
             "grch38": {"chrom": alvo["chrom"], "pos": int(alvo["pos_1based"]), "ref": alvo["ref"], "alt": alvo["alt"]}},
            {"gene": "TP53", "hgvs": "R337H", "grch38": {"chrom": "chr17", "pos": 7670699, "ref": "C", "alt": "T"}}]


class Cenario:
    """O passo 3 rodado sobre o release sintetico, mais o membership e as criticas."""

    def __init__(self, pasta: Path):
        self.pasta = pasta
        self.c3 = passo3.Cenario(pasta)
        membership = self.c3.entrega / bracos.RELEASE / leituras.MEMBERSHIP
        membership.parent.mkdir(parents=True)
        _membership(self.c3.df).to_parquet(membership, index=False)
        self.criticas = _criticas(self.c3.df)
        self.bracos = pasta / "bracos"
        if self.c3.rodar(self.bracos) != 0:
            raise RuntimeError("o passo 3 sintetico falhou")

    def rodar(self, destino: Path, *extra: str) -> int:
        argv = ["--entrega", str(self.c3.entrega), "--bracos", str(self.bracos), "--out-dir", str(destino),
                "--replicas", "20", *extra]
        with patch.object(leituras, "ferramentas_do_mosaic", return_value=_ferramentas(self.c3.referencia)), \
                patch.object(bracos, "REFERENCIA_DO_BENCHMARK", self.c3.referencia), \
                patch.object(leituras, "ler_criticas", return_value=self.criticas):
            return leituras.main(argv)


# -------------------------------------------------------------------------------------------------------- testes

class UnidadesTests(unittest.TestCase):
    def test_limite_superior_de_clopper_pearson(self):
        self.assertAlmostEqual(leituras.limite_superior_cp(0, 580), 1 - 0.05 ** (1 / 580), places=12)
        self.assertEqual(leituras.limite_superior_cp(580, 580), 1.0)
        valores = [leituras.limite_superior_cp(x, 580) for x in range(6)]
        self.assertTrue(all(a < b for a, b in zip(valores, valores[1:])))
        if TEM_SCIPY:
            for x in (1, 3, 10):
                self.assertAlmostEqual(leituras.limite_superior_cp(x, 580), float(_beta.ppf(0.95, x + 1, 580 - x)),
                                       places=10)

    def test_selecao_revisada(self):
        def registro(c, grade, internos=()):
            return {"run": 0, "C": c, "grade": grade, "s_interno": list(internos)}
        grade = [{"C": 0.1, "macro_validacao": 0.9, "convergiu": True, "avisos": []},
                 {"C": 1.0, "macro_validacao": 0.95, "convergiu": False, "avisos": ["ConvergenceWarning: x"]}]
        ok = leituras.conferir_selecao({"bracos": {"E": [registro(0.1, grade)]}})
        self.assertEqual(ok["nao_convergidos_fora_da_escolha"], 1)
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "escolheria C=0.1"):
            leituras.conferir_selecao({"bracos": {"E": [registro(1.0, grade)]}})
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "S interno"):
            leituras.conferir_selecao({"bracos": {"S+F": [registro(0.1, grade[:1],
                                                                   [{"fold_interno": 2, "convergiu": False}])]}})

    def test_p_br_conta_como_o_evaluate_safety(self):
        release = pd.DataFrame({
            "variant_id": ["a", "b", "c", "d", "e"], "binary_label": [1, 1, 1, 1, 0],
            "present_abraom": [True, True, True, False, True], "gene_transfer_group_id": ["g1", "g1", "g2", "g3", "g4"],
            "primary_panel": ["missense"] * 5, "label_tier": ["gold", "consensus", "gold", "gold", "gold"],
            "chrom": ["chr1"] * 5, "pos_1based": [1, 2, 3, 4, 5], "ref": ["A"] * 5, "alt": ["G"] * 5})

        def chamadas(positivas):
            ids = ["a", "b", "c", "d", "e"]
            return pd.DataFrame({"run": 0, "score": 0.0, "limiar": 0.0,
                                 "chamada": ["positive" if v in positivas else "negative" for v in ids]},
                                index=pd.Index(ids, name="variant_id"))
        base = chamadas({"a", "b"})
        novo = chamadas({"b", "c"})
        todos = {b: base for b in bracos.BRACOS} | {"F+BR": novo}
        r = leituras.p_br(release, todos, [{"gene": "X", "grch38": {"chrom": "chr1", "pos": 1, "ref": "A", "alt": "G"}}])
        self.assertEqual((r["n_p_br"], r["n_grupos_de_gene"]), (3, 2))
        par = r["pares"]["F -> F+BR"]
        self.assertEqual((par["reconhecidas_pela_base"], par["perdidas"], par["ganhas"]), (2, 1, 1))
        self.assertEqual((par["ids_perdidas"], par["criticas_perdidas"], par["seguranca_declarada"]), (["a"], 1, False))
        self.assertAlmostEqual(par["limite_superior_95"], leituras.limite_superior_cp(1, 2))


class PontaAPontaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = Cenario(Path(cls.temp.name))
        cls.destino = Path(cls.temp.name) / "leituras"
        cls.codigo = cls.c.rodar(cls.destino)
        cls.r = json.loads((cls.destino / "leituras.json").read_text(encoding="utf-8")) if cls.codigo == 0 else {}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_rodou_e_tem_todas_as_leituras(self):
        self.assertEqual(self.codigo, 0)
        self.assertTrue((self.destino / "resumo.md").exists())
        self.assertFalse(self.destino.with_name("leituras.tmp").exists())
        self.assertEqual(set(self.r["nucleo"]["pares"]), {*(leituras.nome_do_par(b, n) for b, n in leituras.PARES),
                                                          "E+F -> S+F"})
        self.assertEqual(set(self.r["proxies"]), {leituras.nome_do_par(b, n) for b, n in leituras.PARES})
        for par in self.r["proxies"].values():
            self.assertEqual(set(par) - {"base", "novo"}, {"br_clinical_evidence", "br_population_observed"})
        self.assertEqual(set(self.r["beneficio"]["por_braco"]), set(bracos.BRACOS))
        self.assertEqual(self.c.rodar(self.destino), 2, "nunca grava por cima")

    def test_limiares_e_chamadas(self):
        for braco, por_run in self.r["limiares"].items():
            self.assertEqual(set(por_run), {"0", "1", "2", "3", "4"})
            self.assertTrue(all(math.isfinite(v) for v in por_run.values()))
        preds = leituras.carregar_predicoes(self.c.bracos)
        df = self.c.c3.df
        pbr = df[(df["binary_label"] == 1) & df["present_abraom"]]
        self.assertEqual(self.r["p_br"]["n_p_br"], len(pbr))
        for braco in bracos.BRACOS:
            limiar = {int(k): v for k, v in self.r["limiares"][braco].items()}
            chamadas = leituras.chamadas_de_teste(preds[braco], limiar)
            positivas = int((chamadas["chamada"].reindex(pbr["variant_id"]) == "positive").sum())
            self.assertEqual(self.r["p_br"]["por_braco"][braco]["positivas"], positivas)

    def test_criticas_uma_a_uma(self):
        dentro, fora = self.r["criticas"]
        self.assertTrue(dentro["no_release"])
        self.assertEqual(set(dentro["bracos"]), set(bracos.BRACOS))
        self.assertTrue(all(v["chamada"] in ("positive", "negative") for v in dentro["bracos"].values()))
        self.assertEqual((fora["no_release"], fora["bracos"]), (False, None))

    def test_avaliador_oficial_e_limiares_conferidos(self):
        avaliacao = Path(self.temp.name) / "avaliacao"
        for braco, cfg in bracos.BRACOS.items():
            pasta = avaliacao / cfg["id"] / "v1-sintetic" / "4kb"
            pasta.mkdir(parents=True)
            metricas_ = [{"evaluation_id": "core_locus", "track": "core_locus", "run": None, "panel": "missense",
                          "comparator_id": cid, "metric": "auroc", "estimate": 0.8, "ci95_low": 0.7, "ci95_high": 0.9,
                          "n_scored": 10, "coverage": 1.0} for cid in (cfg["id"], "revel")]
            pd.DataFrame(metricas_).astype({"run": "Int64"}).to_parquet(pasta / "specialist-metrics.parquet")
            pd.DataFrame({"comparator_id": cfg["id"], "track": "core_locus", "run": list(range(5)),
                          "threshold": [self.r["limiares"][braco][str(r)] for r in range(5)]}).to_parquet(
                pasta / "validation-thresholds.parquet")
        destino = Path(self.temp.name) / "leituras_com_avaliador"
        self.assertEqual(self.c.rodar(destino, "--avaliacao", str(avaliacao)), 0)
        r = json.loads((destino / "leituras.json").read_text(encoding="utf-8"))
        sistemas = {m["sistema"] for m in r["avaliador_oficial"]["metricas"]}
        self.assertEqual(sistemas, {*bracos.BRACOS, "revel"})
        pasta = avaliacao / bracos.BRACOS["E"]["id"] / "v1-sintetic" / "4kb"
        gravados = pd.read_parquet(pasta / "validation-thresholds.parquet")
        gravados.loc[0, "threshold"] += 1e-6
        gravados.to_parquet(pasta / "validation-thresholds.parquet")
        self.assertEqual(self.c.rodar(Path(self.temp.name) / "leituras_limiar_errado", "--avaliacao", str(avaliacao)), 2)

    def test_predicao_faltando_reprova(self):
        preds = leituras.carregar_predicoes(self.c.bracos)
        df = self.c.c3.df.assign(sequence_eligible=self.c.c3.df["sequence_eligible"].astype(bool))
        leituras.conferir_linhas(preds, df)
        preds["E"] = preds["E"].iloc[1:]
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "linhas esperadas"):
            leituras.conferir_linhas(preds, df)


if __name__ == "__main__":
    unittest.main()
