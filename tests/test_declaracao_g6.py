"""A composicao final declarada para o G6 nao pode divergir do pareamento de sementes (revisao de 23/09).

Os tres comparadores (a_1, a_2, a_3) produzem nove cabecas MR; o sistema MR e so a_i + h_i. Este teste impede que
uma edicao da declaracao troque, misture ou reordene os componentes em silencio.
"""
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha import g6  # noqa: E402
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

    def test_criterios_da_equipe_de_25_09(self):
        # O que a equipe declarou antes de qualquer score dos estudos. Mudar isto depois do --congelar exige um G6
        # novo; depois do G7 real, e post hoc.
        g = self.campanha["g6"]
        m = g["margens"]
        self.assertEqual(g6.problemas_das_margens(m), [])
        self.assertEqual(g6.problemas_do_bootstrap(g["bootstrap_da_interacao"]), [])
        self.assertEqual(g6.problemas_das_pendencias(g["pendencias_antes_do_congelamento"]), [])
        self.assertEqual(m["papel_dos_estudos"], {"br_clinical_evidence": "exigido",
                                                  "br_population_observed": "descritivo"})
        self.assertEqual(m["limiar_de_relevancia"]["valor"], 0.01)
        c1 = m["melhoria_minima_no_coorte_br"]
        self.assertEqual((c1["delta"], c1["metrica"]), ("delta_br_full", "auroc"))
        self.assertEqual(c1["condicoes"], [{"estatistica": "estimativa", "comparacao": ">=", "limite": 0.01},
                                           {"estatistica": "p2_5", "comparacao": ">", "limite": 0.0}])
        self.assertEqual(m["regressao_maxima_no_controle"]["condicoes"],
                         [{"estatistica": "p2_5", "comparacao": ">=", "limite": -0.01}])
        self.assertEqual(m["paineis_com_regressao_inaceitavel"]["paineis"], ["missense"])
        self.assertEqual(m["beneficio_nao_explicado_por_um_painel"]["suporte_minimo_por_painel"], 20)
        self.assertIs(m["interacao"]["criterio_proprio"], False)
        self.assertIn("NAO aprovacao do Eduardo", m["estado"])
        self.assertEqual(g["bootstrap_da_interacao"]["unidade_principal"], "cluster_conjunto")

    def test_tres_componentes_distintos_por_sistema(self):
        for sistema in ("M0", "MR"):
            arquivos = [c["arquivo"] for c in self.composicao[sistema]]
            self.assertEqual(len(arquivos), 3)
            self.assertEqual(len(set(arquivos)), 3)


if __name__ == "__main__":
    unittest.main()
