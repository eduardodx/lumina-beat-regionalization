"""Fase 1, BR v2: conferencia do ABraOM, bloco BR2, layout de cabecas_172 e ponta a ponta (treino e leituras).

Sem scipy e sem o pacote `mosaic` (Windows), o limite inferior de Clopper-Pearson vem de uma copia por bissecao e as
checagens do Mosaic sao as copias dos testes dos passos 3 e 4. No .venv do Mosaic, as funcoes reais.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import fase1_br2_ler as ler  # noqa: E402
from scripts import fase1_br2_treinar as treinar  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from tests import test_fase1_bracos as passo3  # noqa: E402
from tests import test_fase1_leituras as passo4  # noqa: E402

try:
    from mosaic.regional_truth import af_lower_bound as _limite_do_mosaic
    TEM_MOSAIC = True
except ImportError:
    TEM_MOSAIC = False
try:
    from scipy.stats import beta as _beta
    TEM_SCIPY = True
except ImportError:
    TEM_SCIPY = False


def limite_inferior_copiado(ac, an):
    """`regional_truth.af_lower_bound`: o p com P(X >= ac | an, p) = 0,05 (0 quando ac = 0), por bissecao."""
    saida = []
    for x, n in zip(np.asarray(ac, dtype=float), np.asarray(an, dtype=float)):
        x, n = int(x), int(n)
        if x <= 0:
            saida.append(0.0)
            continue

        def cauda(p):   # P(X >= x) = 1 - P(X <= x - 1), somando so os x termos de baixo
            termos = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p)
                      + (n - k) * math.log1p(-p) for k in range(x)]
            maior = max(termos)
            return 1.0 - math.exp(maior) * sum(math.exp(t - maior) for t in termos)

        baixo, alto = 1e-15, 1.0 - 1e-15
        for _ in range(100):
            meio = (baixo + alto) / 2
            if cauda(meio) < 0.05:
                baixo = meio
            else:
                alto = meio
        saida.append((baixo + alto) / 2)
    return np.array(saida)


LIMITE = _limite_do_mosaic if TEM_MOSAIC else limite_inferior_copiado


def _linhas() -> pd.DataFrame:
    return pd.DataFrame({
        "abraom_status": ["present", "present", "present", "ac0", "no_call", "not_found", None],
        "abraom_ac": [1, 50, 50, 0, np.nan, np.nan, np.nan],
        "abraom_an": [2342, 2342, 1000, 2000, 0, np.nan, np.nan],
        "abraom_af": [1 / 2342, 50 / 2342, 50 / 1000, 0.0, np.nan, np.nan, np.nan],
        "abraom_filter": ["PASS", "PASS", "PASS", "PASS", ".", None, None],
        "gnomad_status": ["present", "present", "present", "present", "not_found", "present", "ac0"],
        "gnomad_v4_af": [1e-5, 0.002, 0.0001, 0.001, np.nan, 0.01, 0.0]})


class ContagensTests(unittest.TestCase):
    def test_coerente_passa_e_relata(self):
        r = treinar.conferir_contagens(_linhas())
        self.assertEqual(r["faixas_de_ac_das_presentes"], {"ac1": 1, "ac2": 0, "ac3a9": 0, "ac10mais": 2})
        self.assertEqual(r["por_estado"]["<nulo>"], 1)
        self.assertEqual(r["af_publicada_contra_ac_an"]["acima_de_1pct"], 0)

    def test_contradicoes_interrompem(self):
        for coluna, linha, valor, padrao in (("abraom_ac", 0, 0, "present sem AC"),
                                             ("abraom_an", 5, 2342, "not_found com"),
                                             ("abraom_status", 0, "outro", "estados"),
                                             ("abraom_af", 3, 0.1, "ac0 com AF")):
            with self.subTest(padrao=padrao):
                linhas = _linhas()
                linhas.loc[linha, coluna] = valor
                with self.assertRaisesRegex(bracos.FalhaDaFase1, padrao):
                    treinar.conferir_contagens(linhas)


class BlocoTests(unittest.TestCase):
    def test_faixas_limite_e_excesso_sustentado(self):
        b = treinar.bloco_br2(_linhas(), LIMITE)
        self.assertEqual(len(b.columns), 13)
        self.assertEqual(list(b["br2_ac1"]), [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(b["br2_ac10mais"]), [0, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(b[["br2_not_found", "br2_no_call", "br2_ac0"]].sum(axis=1)), [0, 0, 0, 1, 1, 1, 0])
        self.assertAlmostEqual(b.loc[0, "br2_log_af"], math.log10(1 / 2342 + 1e-6))
        self.assertLess(b.loc[0, "br2_log_af_inferior"], b.loc[0, "br2_log_af"] - 1, "uma copia: limite bem abaixo")
        self.assertTrue(np.allclose(b.loc[3:, "br2_log_af_inferior"], -6.0))
        self.assertGreater(b.loc[1, "br2_excesso_sustentado"], 0, "50 copias contra 0,2% global: excesso sustentado")
        self.assertEqual(b.loc[2, "br2_excesso_sustentado"], 0.0, "AN 1000 < 80% de 2342: nao sustenta")
        self.assertEqual(list(b["br2_an_suficiente"]), [1, 1, 0, 1, 0, 0, 0])
        if TEM_SCIPY:
            np.testing.assert_allclose(limite_inferior_copiado([1, 50], [2342, 2342]),
                                       _beta.ppf(0.05, [1, 50], [2342, 2293]), rtol=1e-6)


class LayoutTests(unittest.TestCase):
    def _cabecas(self, ref, alt):
        n = len(ref)
        cab = np.zeros((n, 172), dtype=np.float32)
        for i, (r, a) in enumerate(zip(ref, alt)):
            cab[i, ler.SUBST_INICIO + ler.BASES.index(r)] = 1.0
            cab[i, ler.REF_INICIO + ler.BASES.index(r)] = 5.0                       # MLM aponta a REF
            cab[i, ler.REF_INICIO + ler._inicio("population_af_head") + ler.BASES.index(a)] = -2.0 - i
            cab[i, ler.REF_INICIO + ler._inicio("population_observed_head") + ler.BASES.index(a)] = 1.0 - i
        return cab

    def test_offsets_e_conferencia(self):
        self.assertEqual((ler.LIN_TOTAL, ler.REF_INICIO, ler.SUBST_INICIO), (68, 78, 156))
        self.assertEqual(ler.REF_INICIO + ler._inicio("population_af_head"), 138)
        ref, alt = ["A", "C", "G", "T"], ["G", "T", "A", "C"]
        cab = self._cabecas(ref, alt)
        self.assertEqual(ler.conferir_layout(cab, np.array(ref))["mlm_na_ref_aponta_a_base_ref"], 1.0)
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "one-hot"):
            ler.conferir_layout(cab, np.array(["C", "C", "G", "T"]))
        elegiveis = pd.DataFrame({"ref": ref, "alt": alt, "gnomad_status": ["present", "present", "present", "not_found"],
                                  "gnomad_v4_af": [1e-2, 1e-3, 1e-4, np.nan],
                                  "abraom_status": ["present", "present", "not_found", "not_found"]})
        r = ler.cabecas_nativas(cab, elegiveis)
        self.assertEqual(r["af_pred"]["n_achadas_no_gnomad"], 3)
        self.assertAlmostEqual(r["af_pred"]["spearman_com_log10_af_gnomad"], 1.0)
        self.assertEqual(r["observed_pred"]["auroc_achada_no_gnomad"], 1.0)


class PontaAPontaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        pasta = Path(cls.temp.name)
        cls.c = passo4.Cenario(pasta)
        cls.ref = cls.c.c3.referencia
        cls.leituras = pasta / "leituras"
        assert cls.c.rodar(cls.leituras) == 0
        cls.br2 = pasta / "br2"
        ferramentas = {**passo3._ferramentas(cls.ref), "af_lower_bound": LIMITE}
        ajustador = bracos.ajustar_sklearn if passo3.TEM_SKLEARN else passo3.ajustador_ridge
        with patch.object(treinar, "ferramentas_do_mosaic", return_value=ferramentas), \
                patch.object(bracos, "REFERENCIA_DO_BENCHMARK", cls.ref), \
                patch.object(bracos, "CONTAGENS_DO_GUIA_RUN0", None), patch.object(bracos, "AJUSTADOR", ajustador):
            cls.codigo_treino = treinar.main(["--entrega", str(cls.c.c3.entrega), "--bracos", str(cls.c.bracos),
                                              "--out-dir", str(cls.br2),
                                              *sum((["--cache", str(p)] for p in cls.c.c3.caches), [])])
        cls.destino = pasta / "leituras_br2"
        cls.codigo_leitura = cls.ler(cls.destino)
        cls.r = (json.loads((cls.destino / "br2.json").read_text(encoding="utf-8"))
                 if cls.codigo_leitura == 0 else {})

    @classmethod
    def ler(cls, destino, *extra):
        with patch.object(leituras, "ferramentas_do_mosaic", return_value=passo4._ferramentas(cls.ref)), \
                patch.object(bracos, "REFERENCIA_DO_BENCHMARK", cls.ref), \
                patch.object(ler, "ler_criticas", return_value=cls.c.criticas):
            return ler.main(["--entrega", str(cls.c.c3.entrega), "--bracos", str(cls.c.bracos), "--br2", str(cls.br2),
                             "--leituras", str(cls.leituras), "--out-dir", str(destino), "--replicas", "20",
                             *sum((["--cache", str(p)] for p in cls.c.c3.caches), []), *extra])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_treino_no_contrato_com_as_mesmas_linhas(self):
        self.assertEqual(self.codigo_treino, 0)
        novo = pd.read_parquet(self.br2 / treinar.ID / "predictions.parquet")
        antigo = pd.read_parquet(self.c.bracos / "fase1-e-f-br" / "predictions.parquet")
        chave = ["variant_id", "run", "role"]
        self.assertTrue(novo[chave].sort_values(chave).reset_index(drop=True).equals(
            antigo[chave].sort_values(chave).reset_index(drop=True)))
        sistema = json.loads((self.br2 / treinar.ID / "system.yaml").read_text(encoding="utf-8"))
        self.assertEqual((sistema["id"], sistema["fase1"]["blocos"]), (treinar.ID, ["e", "f", "br2"]))
        contagens = json.loads((self.br2 / "contagens.json").read_text(encoding="utf-8"))
        self.assertEqual(len(contagens["colunas_br2"]), 13)
        self.assertEqual(len(list((self.br2 / "modelos").glob("*.npz"))), 5)

    def test_leituras_reproduzem_o_passo_4_e_medem_a_recuperacao(self):
        self.assertEqual(self.codigo_leitura, 0)
        self.assertEqual(set(self.r["nucleo"]["pares"]), {leituras.nome_do_par(b, n) for b, n in ler.PARES_BR2})
        lidas4 = json.loads((self.leituras / "leituras.json").read_text(encoding="utf-8"))
        self.assertEqual(self.r["p_br"]["pares"][ler.ORIGINAL]["perdidas"], lidas4["p_br"]["pares"][ler.ORIGINAL]["perdidas"])
        for leitura in ("limiar_mcc_original", "especificidade_equivalente"):
            rec = self.r["recuperacao"][leitura]
            self.assertEqual(rec["recuperadas_pelo_br2"] + rec["mantidas"], rec["perdidas_pelo_br_original"])
            self.assertEqual(rec["mantidas"] + rec["novas_no_br2"], rec["perdidas_pelo_br2"])
        self.assertEqual(set(self.r["coeficientes"]), {"E+F+BR", "E+F+BR2"})
        self.assertTrue(all(c.startswith("br2_") for c in self.r["coeficientes"]["E+F+BR2"]["media"]))
        self.assertIn("falhou", self.r["cabecas_nativas"], "caches sinteticos tem cabecas zeradas: o layout reprova")
        self.assertTrue((self.destino / "br2.md").exists())
        self.assertEqual(self.ler(self.destino), 2, "nunca grava por cima")

    def test_braco_novo_sem_especificacao_nao_roda(self):
        with patch.object(treinar, "ESPECIFICACAO", Path(self.temp.name) / "nao_existe.md"):
            self.assertEqual(treinar.main(["--entrega", "x", "--bracos", "y", "--out-dir",
                                           str(Path(self.temp.name) / "sem_especificacao")]), 2)


if __name__ == "__main__":
    unittest.main()
