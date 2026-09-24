"""A composicao final declarada para o G6 nao pode divergir do pareamento de sementes (revisao de 23/09).

Os tres comparadores (a_1, a_2, a_3) produzem nove cabecas MR; o sistema MR e so a_i + h_i. Este teste impede que
uma edicao da declaracao troque, misture ou reordene os componentes em silencio.
"""
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha.recortes import carregar_campanha  # noqa: E402


class DeclaracaoG6Tests(unittest.TestCase):
    def setUp(self):
        self.campanha = carregar_campanha(RAIZ / "configs" / "campanha_r03_desenvolvimento.json")
        self.composicao = self.campanha["g6"]["composicao_final"]
        self.adapters = list(self.campanha["sementes"]["adapter"])

    def test_mr_e_exatamente_o_pareamento_declarado(self):
        pares = [(c["adapter"], c["cabeca"]) for c in self.campanha["sementes"]["combinacoes"]]
        self.assertEqual([(c["adapter"], c["cabeca"]) for c in self.composicao["MR"]], pares)

    def test_m0_usa_as_mesmas_cabecas_sem_adapter(self):
        self.assertEqual([c["cabeca"] for c in self.composicao["M0"]], list(self.campanha["sementes"]["cabeca"]))
        self.assertTrue(all("adapter" not in c for c in self.composicao["M0"]))

    def test_cada_arquivo_e_o_do_comparador_do_proprio_adapter(self):
        for componente in self.composicao["MR"]:
            indice = self.adapters.index(componente["adapter"]) + 1
            self.assertEqual(componente["arquivo"], f"comparacao_dev_a{indice}/cabeca_MR_h{componente['cabeca']}.pt")
        for componente in self.composicao["M0"]:
            self.assertEqual(componente["arquivo"], f"comparacao_dev_a1/cabeca_M0_h{componente['cabeca']}.pt")

    def test_tres_componentes_distintos_por_sistema(self):
        for sistema in ("M0", "MR"):
            arquivos = [c["arquivo"] for c in self.composicao[sistema]]
            self.assertEqual(len(arquivos), 3)
            self.assertEqual(len(set(arquivos)), 3)


if __name__ == "__main__":
    unittest.main()
