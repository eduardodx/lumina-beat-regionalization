"""Conferencia de reproducao do cache, so as regras puras (sem torch nem GPU)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.conferir_reproducao_do_cache import comparar_leituras, problemas_do_alinhamento  # noqa: E402


class AlinhamentoTests(unittest.TestCase):
    def setUp(self):
        self.tabela = np.array([f"v{i:03d}" for i in range(100)])

    def test_primeiras_iguais_em_lotes_completos_passa(self):
        self.assertEqual(problemas_do_alinhamento(self.tabela, self.tabela[:4096], 64, 8), [])

    def test_lote_incompleto_ou_fragmento_curto_recusa(self):
        self.assertTrue(problemas_do_alinhamento(self.tabela, self.tabela, 60, 8))
        self.assertTrue(problemas_do_alinhamento(self.tabela, self.tabela[:32], 64, 8))

    def test_fragmento_de_retomada_nao_e_o_comeco_da_tabela(self):
        # Um cache retomado pode ter o fragmento 0 fora da ordem da tabela: os lotes nao seriam os mesmos.
        self.assertTrue(problemas_do_alinhamento(self.tabela, self.tabela[8:], 64, 8))


class ComparacaoTests(unittest.TestCase):
    def test_identica_dentro_e_fora_da_tolerancia(self):
        a = {"x": np.ones((4, 3), dtype=np.float32), "y": np.zeros((4, 2), dtype=np.float32)}
        r = comparar_leituras({k: v.copy() for k, v in a.items()}, a, 1e-5)
        self.assertTrue(r["passou"])
        self.assertTrue(all(v["identica"] for v in r["por_leitura"].values()))
        quase = {"x": a["x"] + 2e-6, "y": a["y"]}
        r = comparar_leituras(quase, a, 1e-5)
        self.assertTrue(r["passou"])
        self.assertFalse(r["por_leitura"]["x"]["identica"])
        longe = {"x": a["x"] + 1e-3, "y": a["y"]}
        self.assertFalse(comparar_leituras(longe, a, 1e-5)["passou"])

    def test_leitura_ausente_ou_forma_diferente_reprova(self):
        a = {"x": np.ones((4, 3), dtype=np.float32)}
        self.assertFalse(comparar_leituras({"x": np.ones((4, 2))}, a, 1e-5)["passou"])
        self.assertFalse(comparar_leituras({}, a, 1e-5)["passou"])


if __name__ == "__main__":
    unittest.main()
