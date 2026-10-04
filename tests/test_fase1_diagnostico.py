"""Diagnostico posterior da Fase 1 (perdas e ganhos de P-BR) sobre o cenario sintetico dos passos 3 e 4."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_diagnostico as diag  # noqa: E402
from tests import test_fase1_leituras as passo4  # noqa: E402


class QuantidadesTests(unittest.TestCase):
    def test_fpr_exigido(self):
        benignas = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(diag.fpr_exigido(2.0, benignas), 0.5)    # 2 e 3 ficariam positivas
        self.assertEqual(diag.fpr_exigido(3.5, benignas), 0.0)
        self.assertEqual(diag.fpr_exigido(-1.0, benignas), 1.0)

    def test_limiar_com_especificidade_respeita_empates(self):
        b = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
        self.assertTrue(0.0 < diag.limiar_com_especificidade(b, 0.2) < 1.0)    # logo acima de 0: 1 de 5 abaixo
        self.assertTrue(1.0 < diag.limiar_com_especificidade(b, 0.4) < 2.0)    # o empate em 1 leva 4 de 5 abaixo
        self.assertLess(diag.limiar_com_especificidade(b, 0.0), 0.0)
        self.assertGreater(diag.limiar_com_especificidade(b, 1.0), 2.0)
        for alvo in (0.2, 0.4, 0.8, 1.0):
            t = diag.limiar_com_especificidade(b, alvo)
            self.assertGreaterEqual(1 - diag.fpr_exigido(t, b), alvo - 1e-12)

    def test_mecanismo_de_ponto_de_operacao_e_de_ordenacao(self):
        pbr = pd.DataFrame({"variant_id": ["a", "b"], "run": [0, 0], "abraom_af": [0.01, 0.02],
                            "gnomad_v4_af": [0.001, 0.0], "label_tier": ["gold", "consensus"],
                            "primary_panel": ["missense", "missense"]}).set_index("variant_id", drop=False)
        benignas = {"F": {0: np.linspace(0, 1, 101)}, "F+BR": {0: np.linspace(0, 1, 101)}}

        def chamadas(scores, limiar):
            return pd.DataFrame({"run": 0, "score": scores, "limiar": limiar,
                                 "chamada": np.where(np.array(scores) >= limiar, "positive", "negative")},
                                index=pd.Index(["a", "b"], name="variant_id"))
        # base: limiar 0,9 (fpr 0,10); as duas positivas. novo: limiar 0,97 (fpr 0,03).
        # a: 0,95 no novo, exige fpr 0,05 <= 0,10 -> ponto de operacao; b: 0,5 no novo, exige 0,5 -> ordenacao.
        todas = {"F": chamadas([0.92, 0.93], 0.9), "F+BR": chamadas([0.95, 0.5], 0.97)}
        linhas = diag.linhas_do_par(pbr, todas, benignas, "F", "F+BR")
        self.assertEqual(list(linhas["tipo"]), ["perdida", "perdida"])
        self.assertEqual(list(linhas["mecanismo"]), ["ponto_de_operacao", "ordenacao"])
        self.assertEqual(list(linhas["razao_abraom_gnomad"]), [">=5x", "gnomAD 0 ou ausente"])
        eq = diag.especificidade_equivalente(pbr, todas, benignas, "F", "F+BR")
        self.assertEqual((eq["perdidas"], eq["ganhas"]), (1, 0), "no fpr da base, so a recupera")


class PontaAPontaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = passo4.Cenario(Path(cls.temp.name))
        cls.leituras = Path(cls.temp.name) / "leituras"
        assert cls.c.rodar(cls.leituras) == 0

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def rodar(self, destino: Path, leituras: Path | None = None) -> int:
        argv = ["--entrega", str(self.c.c3.entrega), "--bracos", str(self.c.bracos),
                "--leituras", str(leituras or self.leituras), "--out-dir", str(destino)]
        with patch.object(bracos, "REFERENCIA_DO_BENCHMARK", self.c.c3.referencia), \
                patch.object(diag, "ler_criticas", return_value=self.c.criticas):
            return diag.main(argv)

    def test_reproduz_o_passo_4_e_grava_tudo(self):
        destino = Path(self.temp.name) / "diagnostico"
        self.assertEqual(self.rodar(destino), 0)
        d = json.loads((destino / "diagnostico.json").read_text(encoding="utf-8"))
        lidas = json.loads((self.leituras / "leituras.json").read_text(encoding="utf-8"))
        for nome, par in d["pares"].items():
            gravado = lidas["p_br"]["pares"][nome]
            self.assertEqual(par["perdidas"]["n"], gravado["perdidas"])
            self.assertEqual(par["ganhas"]["n"], gravado["ganhas"])
            self.assertEqual(sum(par["perdidas"]["mecanismo"].values()), gravado["perdidas"])
            # Toda perda de ordenacao continua perdida no falso-positivo da base; as de ponto de operacao, nao.
            self.assertGreaterEqual(par["especificidade_equivalente"]["perdidas"],
                                    par["perdidas"]["mecanismo"].get("ordenacao", 0))
            self.assertIn("natureza", par["especificidade_equivalente"])
        self.assertEqual(len(d["criticas"]), 2)
        tabela = pd.read_csv(destino / "perdas_e_ganhos.csv")
        self.assertTrue(set(tabela["tipo"]) <= {"perdida", "ganha"})
        self.assertTrue((destino / "diagnostico.md").exists())
        self.assertEqual(self.rodar(destino), 2, "nunca grava por cima")

    def test_perdas_que_nao_batem_com_o_passo_4_reprovam(self):
        adulterado = Path(self.temp.name) / "leituras_adulteradas"
        adulterado.mkdir()
        lidas = json.loads((self.leituras / "leituras.json").read_text(encoding="utf-8"))
        par = next(iter(lidas["p_br"]["pares"].values()))
        par["ids_perdidas"] = par["ids_perdidas"] + ["v9999"]
        (adulterado / "leituras.json").write_text(json.dumps(lidas), encoding="utf-8")
        self.assertEqual(self.rodar(Path(self.temp.name) / "diagnostico_errado", adulterado), 2)


if __name__ == "__main__":
    unittest.main()
