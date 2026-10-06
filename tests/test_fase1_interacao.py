"""Fase 1, cabeca com interacao: matriz H, veredito, S reaproveitado e ponta a ponta (treino e leituras).

Sem sklearn, scipy e o pacote `mosaic` (Windows), o ajuste e as checagens do Mosaic sao as copias dos testes dos
passos 3 e 4 e do BR v2. No .venv do Mosaic, as funcoes reais.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import fase1_br2_ler as ler_br2  # noqa: E402
from scripts import fase1_br2_treinar as treinar_br2  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_interacao_ler as ler_h  # noqa: E402
from scripts import fase1_interacao_treinar as treinar_h  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from tests import test_fase1_br2 as teste_br2  # noqa: E402
from tests import test_fase1_bracos as passo3  # noqa: E402
from tests import test_fase1_leituras as passo4  # noqa: E402

AJUSTADOR = bracos.ajustar_sklearn if passo3.TEM_SKLEARN else passo3.ajustador_ridge


class MatrizTests(unittest.TestCase):
    def test_produtos_padronizados_no_treino(self):
        dados = bracos.Dados(ids=np.array(["a", "b", "c", "d"]), y=np.array([0, 1, 0, 1]),
                             paineis=np.array(["missense"] * 4), cromossomos=np.array(["1"] * 4),
                             posicoes=np.arange(4), folds=np.zeros(4, dtype=np.int64),
                             blocos={"e": np.arange(8, dtype=float).reshape(4, 2),
                                     "f": np.array([[1.0], [2.0], [3.0], [6.0]])},
                             colunas={"e": ["e_0000", "e_0001"], "f": ["f_x"], "s": ["s_e"]}, papeis=[])
        cfg = {"blocos": ("e", "f"), "interacoes": ("f",)}
        treino, s = np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])
        z = treinar_h.padronizacao_das_interacoes(dados, cfg, treino, s)
        X = treinar_h.montar_h(dados, cfg, treino, s, z)
        self.assertEqual(treinar_h.colunas_h(dados, cfg), ["e_0000", "e_0001", "f_x", "s_e", "s_x_f_x"])
        np.testing.assert_allclose(X[:, :3], np.hstack([dados.blocos["e"][treino], dados.blocos["f"][treino]]))
        np.testing.assert_allclose(X[:, 3], s, err_msg="S entra sozinho")
        f = dados.blocos["f"][treino, 0]
        np.testing.assert_allclose(X[:, 4], (s - s.mean()) / s.std() * (f - f.mean()) / f.std())
        fora = treinar_h.montar_h(dados, cfg, np.array([3]), np.array([5.0]), z)
        self.assertAlmostEqual(fora[0, 4], (5.0 - s.mean()) / s.std() * (6.0 - f.mean()) / f.std(),
                               msg="fora do treino, a padronizacao e a do treino")
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "um score por linha"):
            treinar_h.montar_h(dados, cfg, treino, s[:2], z)


class VereditoTests(unittest.TestCase):
    @staticmethod
    def entradas(mcc=(3, 4), especificidade=(2, 3), protegidas=8, regional=(0.002, 0.004), fp=(0.010, 0.009)):
        perdas = {ler_h.REGIONAL: {"mcc": mcc[0], "especificidade": especificidade[0]},
                  ler_h.SEGURANCA: {"mcc": mcc[1], "especificidade": especificidade[1]}}
        delta = {"regional": {"clinico": regional[0], "beneficio": regional[1]},
                 "referencia": {"clinico": 0.0015, "beneficio": 0.0038}}
        return perdas, protegidas, delta, {ler_h.H_EF: fp[0], ler_h.H_EFBR2: fp[1]}

    def test_regras_da_especificacao(self):
        def v(**kw):
            return ler_h.veredito(*self.entradas(**kw))["veredito"]
        self.assertEqual(v(), "apoio")
        self.assertEqual(v(mcc=(6, 6), especificidade=(5, 5), protegidas=6, regional=(0.0015, 0.0038),
                           fp=(0.01, 0.01)), "apoio", "os limites declarados ainda apoiam")
        self.assertEqual(v(mcc=(3, 7)), "inconclusivo", "a seguranca contra o E+F original tambem conta")
        self.assertEqual(v(especificidade=(2, 6)), "inconclusivo")
        self.assertEqual(v(protegidas=5), "inconclusivo")
        self.assertEqual(v(regional=(0.0014, 0.004)), "inconclusivo")
        self.assertEqual(v(fp=(0.009, 0.010)), "inconclusivo")
        self.assertEqual(v(mcc=(3, 10)), "contra")
        self.assertEqual(v(mcc=(10, 3)), "contra")
        self.assertEqual(v(protegidas=3), "contra")


class PontaAPontaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        pasta = Path(cls.temp.name)
        cls.c = passo4.Cenario(pasta)
        cls.ref = cls.c.c3.referencia
        cls.leituras = pasta / "leituras"
        assert cls.c.rodar(cls.leituras) == 0
        cls.ferramentas = {**passo3._ferramentas(cls.ref), "af_lower_bound": teste_br2.LIMITE}
        caches = sum((["--cache", str(p)] for p in cls.c.c3.caches), [])
        cls.br2, cls.leituras_br2, cls.h = pasta / "br2", pasta / "leituras_br2", pasta / "h"
        with cls.treino():
            assert treinar_br2.main(["--entrega", str(cls.c.c3.entrega), "--bracos", str(cls.c.bracos),
                                     "--out-dir", str(cls.br2), *caches]) == 0
            cls.codigo_treino = treinar_h.main(["--entrega", str(cls.c.c3.entrega), "--bracos", str(cls.c.bracos),
                                                "--out-dir", str(cls.h), *caches])
            cls.preparado = treinar_h.preparar(cls.c.c3.entrega, cls.c.bracos, list(cls.c.c3.caches), cls.ferramentas)
        with patch.object(leituras, "ferramentas_do_mosaic", return_value=passo4._ferramentas(cls.ref)), \
                patch.object(bracos, "REFERENCIA_DO_BENCHMARK", cls.ref), \
                patch.object(ler_br2, "ler_criticas", return_value=cls.c.criticas):
            assert ler_br2.main(["--entrega", str(cls.c.c3.entrega), "--bracos", str(cls.c.bracos), "--br2",
                                 str(cls.br2), "--leituras", str(cls.leituras), "--out-dir", str(cls.leituras_br2),
                                 "--replicas", "20", *caches]) == 0
        cls.destino = pasta / "leituras_h"
        cls.codigo_leitura = cls.ler(cls.destino)
        cls.r = (json.loads((cls.destino / "interacao.json").read_text(encoding="utf-8"))
                 if cls.codigo_leitura == 0 else {})

    @classmethod
    def treino(cls):
        class Patches:
            def __enter__(self_):
                self_.pilha = [patch.object(treinar_br2, "ferramentas_do_mosaic", return_value=cls.ferramentas),
                               patch.object(treinar_h, "ferramentas_do_mosaic", return_value=cls.ferramentas),
                               patch.object(bracos, "REFERENCIA_DO_BENCHMARK", cls.ref),
                               patch.object(bracos, "CONTAGENS_DO_GUIA_RUN0", None),
                               patch.object(bracos, "AJUSTADOR", AJUSTADOR)]
                for p in self_.pilha:
                    p.__enter__()

            def __exit__(self_, *exc):
                for p in reversed(self_.pilha):
                    p.__exit__(*exc)
        return Patches()

    @classmethod
    def ler(cls, destino, *extra):
        with patch.object(leituras, "ferramentas_do_mosaic", return_value=passo4._ferramentas(cls.ref)), \
                patch.object(bracos, "REFERENCIA_DO_BENCHMARK", cls.ref), \
                patch.object(ler_h, "ler_criticas", return_value=cls.c.criticas):
            return ler_h.main(["--entrega", str(cls.c.c3.entrega), "--bracos", str(cls.c.bracos), "--leituras",
                               str(cls.leituras), "--br2", str(cls.br2), "--leituras-br2", str(cls.leituras_br2),
                               "--h", str(cls.h), "--out-dir", str(destino), "--replicas", "20", *extra])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_treino_no_contrato_com_as_mesmas_linhas(self):
        self.assertEqual(self.codigo_treino, 0)
        chave = ["variant_id", "run", "role"]
        antigo = pd.read_parquet(self.c.bracos / "fase1-e-f" / "predictions.parquet")[chave]
        for braco, cfg in treinar_h.BRACOS_H.items():
            novo = pd.read_parquet(self.h / cfg["id"] / "predictions.parquet")[chave]
            self.assertTrue(novo.sort_values(chave).reset_index(drop=True).equals(
                antigo.sort_values(chave).reset_index(drop=True)))
            sistema = json.loads((self.h / cfg["id"] / "system.yaml").read_text(encoding="utf-8"))
            self.assertEqual((sistema["id"], sistema["fase1"]["braco"]), (cfg["id"], braco))
        self.assertEqual(len(list((self.h / "modelos").glob("*.npz"))), 10)
        contagens = json.loads((self.h / "contagens.json").read_text(encoding="utf-8"))
        self.assertEqual(contagens["colunas"]["H(E+F)"][0], "s_e")
        self.assertEqual(sum(c.startswith("s_x_br2_") for c in contagens["colunas"]["H(E+F+BR2)"]), 13)
        self.assertFalse(any(c.startswith("s_x_br2_") for c in contagens["colunas"]["H(E+F)"]))
        self.assertEqual(len(contagens["s"]["execucoes"]), 5)

    def _adulterado(self, nome: str) -> Path:
        pasta = Path(self.temp.name) / nome
        (pasta / "diagnosticos").mkdir(parents=True)
        (pasta / "fase1-e").mkdir()
        for relativo in ("selecao.json", "diagnosticos/s_fora_da_amostra.parquet", "fase1-e/predictions.parquet"):
            shutil.copy(self.c.bracos / relativo, pasta / relativo)
        return pasta

    def test_s_reaproveitado_e_conferido(self):
        dados = self.preparado["dados"]
        s, relatorio = treinar_h.carregar_s(self.c.bracos, dados)
        self.assertEqual(len(s), 5)
        self.assertEqual(relatorio["execucoes"][0]["n"]["treino"], len(dados.papeis[0]["treino"]))

        pasta = self._adulterado("s_fora_de_ordem")
        t = pd.read_parquet(pasta / "diagnosticos" / "s_fora_da_amostra.parquet")
        pd.concat([t.iloc[[1, 0]], t.iloc[2:]], ignore_index=True).to_parquet(
            pasta / "diagnosticos" / "s_fora_da_amostra.parquet", index=False)
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "na ordem"):
            treinar_h.carregar_s(pasta, dados)

        pasta = self._adulterado("s_alterado")
        t = pd.read_parquet(pasta / "diagnosticos" / "s_fora_da_amostra.parquet")
        t.loc[0, "score_e"] += 1.0
        t.to_parquet(pasta / "diagnosticos" / "s_fora_da_amostra.parquet", index=False)
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "s_resumo"):
            treinar_h.carregar_s(pasta, dados)

        for nome, mudar, padrao in (
                ("purga_alterada", lambda sel: sel["bracos"]["S+F"][0]["s_interno"][0].update(
                    purgadas_internas=sel["bracos"]["S+F"][0]["s_interno"][0]["purgadas_internas"] + 1), "purgas"),
                ("c_alterado", lambda sel: sel["bracos"]["E"][0].update(C=999.0), "outro C")):
            with self.subTest(nome=nome):
                pasta = self._adulterado(nome)
                selecao = json.loads((pasta / "selecao.json").read_text(encoding="utf-8"))
                mudar(selecao)
                (pasta / "selecao.json").write_text(json.dumps(selecao), encoding="utf-8")
                with self.assertRaisesRegex(bracos.FalhaDaFase1, padrao):
                    treinar_h.carregar_s(pasta, dados)

    def test_leituras_reproduzem_o_br2_e_calculam_o_veredito(self):
        self.assertEqual(self.codigo_leitura, 0)
        self.assertEqual(set(self.r["nucleo"]["pares"]), {leituras.nome_do_par(b, n) for b, n in ler_h.PARES_H})
        self.assertEqual(self.r["reproducao"]["p_br"], "igual ao br2.json")
        self.assertIn(self.r["veredito"]["veredito"], {"apoio", "contra", "inconclusivo"})
        self.assertEqual(set(self.r["veredito"]["entradas"]["perdas"]), {ler_h.REGIONAL, ler_h.SEGURANCA})
        br2 = json.loads((self.leituras_br2 / "br2.json").read_text(encoding="utf-8"))
        self.assertEqual(self.r["perdas_do_br2"]["n"], len(br2["p_br"]["pares"][ler_br2.NOVO]["ids_perdidas"]))
        self.assertEqual(set(self.r["coeficientes"]), set(treinar_h.BRACOS_H))
        self.assertTrue(all(c.startswith("s_") for c in self.r["coeficientes"]["H(E+F+BR2)"]["media"]))
        self.assertTrue((self.destino / "interacao.md").exists())
        self.assertEqual(self.ler(self.destino), 2, "nunca grava por cima")

    def test_especificacao_alterada_depois_do_treino_reprova(self):
        outra = Path(self.temp.name) / "outra_especificacao.md"
        outra.write_text("outra regra, escrita depois de treinar\n", encoding="utf-8")
        with patch.object(treinar_h, "ESPECIFICACAO", outra):
            self.assertEqual(self.ler(Path(self.temp.name) / "leituras_h_outra"), 2)


if __name__ == "__main__":
    unittest.main()
