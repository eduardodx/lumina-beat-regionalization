"""Metricas da comparacao de desenvolvimento, G5 e o comparador sobre probabilidades prontas. Sem torch."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha import g5, metricas  # noqa: E402
from scripts.comparar_m0_mr_desenvolvimento import comparar  # noqa: E402


class AurocTests(unittest.TestCase):
    def test_ordenamento_perfeito_inverso_e_constante(self):
        self.assertEqual(metricas.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(metricas.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)
        self.assertEqual(metricas.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5, "empate vale meio ponto")

    def test_ruido_de_ponto_flutuante_nao_vira_ordenamento(self):
        # O caso que a pesquisa corrigiu: scores constantes a menos de 1e-17 davam 0,511 e 0,580.
        ruido = [0.3 + i * 1e-17 for i in range(6)]
        self.assertEqual(metricas.auroc(ruido, [1, 0, 1, 0, 1, 0]), 0.5)

    def test_uma_classe_so_e_indefinida(self):
        self.assertIsNone(metricas.auroc([0.1, 0.2], [1, 1]))

    def test_auprc_perfeito_e_exemplo_a_mao(self):
        self.assertEqual(metricas.auprc([0.9, 0.8, 0.1], [1, 1, 0]), 1.0)
        # ordem 0.9(P) 0.8(N) 0.7(P): precisao 1 em R=0,5; 2/3 em R=1 -> 0,5*1 + 0,5*2/3
        self.assertAlmostEqual(metricas.auprc([0.9, 0.8, 0.7], [1, 0, 1]), 0.5 + 0.5 * 2 / 3)

    def test_macro_indefinida_se_um_painel_de_discriminacao_nao_tem_as_duas_classes(self):
        paineis = ["missense", "missense", "splice", "splice", "noncoding", "noncoding"]
        completo = metricas.por_painel([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], [0, 1, 0, 1, 0, 1], paineis)
        self.assertEqual(metricas.macro(completo), 1.0)
        faltando = metricas.por_painel([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], [0, 1, 0, 1, 1, 1], paineis)
        self.assertIsNone(metricas.macro(faltando))


def _dados(n_clusters=30, por_cluster=6, seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for c in range(n_clusters):
        for i in range(por_cluster):
            linhas.append({"variant_id": f"v{c}_{i}", "overlap_cluster_id": f"c{c}",
                           "primary_panel": ("missense", "splice", "noncoding")[(c + i) % 3],
                           "binary_label": int(rng.random() < 0.5)})
    return pd.DataFrame(linhas)


class BootstrapTests(unittest.TestCase):
    def test_sistemas_identicos_dao_delta_zero_e_ic_degenerado(self):
        tabela = _dados()
        scores = np.random.default_rng(1).random(len(tabela))
        saida = metricas.bootstrap_pareado_por_cluster(scores, scores, tabela["binary_label"], tabela["primary_panel"],
                                                       tabela["overlap_cluster_id"], replicas=200)
        for chave in ("macro", "auroc", "auprc"):
            self.assertEqual(saida[chave]["delta"], 0.0)
            self.assertEqual((saida[chave]["p2_5"], saida[chave]["p97_5"]), (0.0, 0.0))
        self.assertEqual(saida["natureza"], "exploratoria")
        self.assertEqual(saida["clusters"], 30)

    def test_sistema_melhor_tem_delta_positivo(self):
        tabela = _dados()
        y = tabela["binary_label"].to_numpy()
        ruim = np.random.default_rng(2).random(len(tabela))
        bom = y + 0.3 * np.random.default_rng(3).random(len(tabela))
        saida = metricas.bootstrap_pareado_por_cluster(ruim, bom, y, tabela["primary_panel"],
                                                       tabela["overlap_cluster_id"], replicas=200)
        self.assertGreater(saida["auroc"]["p2_5"], 0.0)


class G5Tests(unittest.TestCase):
    def test_politica_mais_isolada_dentro_da_margem(self):
        self.assertEqual(g5.politica_escolhida({"nenhum": 0.80, "janela2048": 0.795, "janela4096": 0.785}), "janela2048")
        self.assertEqual(g5.politica_escolhida({"nenhum": 0.80, "janela2048": 0.79, "janela4096": 0.79}), "janela4096",
                         "exatamente na margem conta como dentro")
        self.assertEqual(g5.politica_escolhida({"nenhum": 0.80, "janela2048": 0.70, "janela4096": 0.70}), "nenhum")

    def test_politica_sem_macro_e_recusada(self):
        with self.assertRaises(ValueError):
            g5.politica_escolhida({"nenhum": 0.8, "janela2048": None, "janela4096": 0.8})

    def test_extracao_comparada_na_politica_que_a_regra_escolhe(self):
        medias = {("cabecas_172", "nenhum"): 0.80, ("cabecas_172", "janela2048"): 0.70,
                  ("cabecas_172", "janela4096"): 0.70,
                  ("leitura_antiga_1344", "nenhum"): 0.79, ("leitura_antiga_1344", "janela2048"): 0.785,
                  ("leitura_antiga_1344", "janela4096"): 0.785}
        decisao = g5.escolha(medias)
        self.assertEqual(decisao["por_extracao"]["leitura_antiga_1344"]["politica"], "janela4096")
        self.assertEqual(decisao["por_extracao"]["cabecas_172"]["politica"], "nenhum")
        self.assertEqual((decisao["extracao"], decisao["politica"]), ("cabecas_172", "nenhum"))

    def test_empate_exato_entre_extracoes_fica_com_172(self):
        medias = {(e, p): 0.8 for e in ("cabecas_172", "leitura_antiga_1344") for p in g5.ISOLAMENTO}
        decisao = g5.escolha(medias)
        self.assertEqual((decisao["extracao"], decisao["politica"]), ("cabecas_172", "janela4096"))
        self.assertTrue(decisao["empate_exato_entre_extracoes"])


class ComparadorTests(unittest.TestCase):
    def test_sistemas_iguais_dao_delta_zero_em_tudo(self):
        tabela = _dados()
        linhas = {"selecao": np.arange(len(tabela))}
        y = tabela["binary_label"].to_numpy()
        rodadas = []
        for semente in (11, 12, 13):
            prob = np.clip(y * 0.6 + np.random.default_rng(semente).random(len(tabela)) * 0.4, 0, 1)
            rodadas.append({"semente": semente, "epoca": 10, "prob_selecao": prob,
                            "selecao": metricas.resumo(prob, y, tabela["primary_panel"])})
        saida = comparar(rodadas, rodadas, tabela, linhas, replicas=100, seed=1)
        self.assertTrue(all(v == 0.0 for v in saida["media_das_probabilidades"]["delta"].values()))
        self.assertTrue(all(linha["delta"]["macro"] == 0.0 for linha in saida["por_semente"]))
        self.assertEqual(saida["bootstrap_da_media"]["auroc"]["p97_5"], 0.0)

    def test_sementes_desalinhadas_sao_recusadas(self):
        tabela = _dados()
        y = tabela["binary_label"].to_numpy()
        base = {"epoca": 1, "prob_selecao": y * 1.0, "selecao": metricas.resumo(y * 1.0, y, tabela["primary_panel"])}
        with self.assertRaises(ValueError):
            comparar([dict(base, semente=11)], [dict(base, semente=12)], tabela,
                     {"selecao": np.arange(len(tabela))}, replicas=10, seed=1)



class NomesTests(unittest.TestCase):
    def test_scripts_do_g5_e_do_comparador_nao_referenciam_nomes_indefinidos(self):
        # Um NameError so apareceria no fim da rodada real, depois de treinar as cabecas.
        import builtins
        import importlib
        import symtable

        for nome in ("scripts.g5_escolher_extracao_e_politica", "scripts.comparar_m0_mr_desenvolvimento"):
            modulo = importlib.import_module(nome)
            caminho = Path(modulo.__file__)
            tabela = symtable.symtable(caminho.read_text(encoding="utf-8"), str(caminho), "exec")
            disponiveis = set(vars(modulo)) | set(vars(builtins))
            for funcao in tabela.get_children():
                indefinidos = {s.get_name() for s in funcao.get_symbols()
                               if s.is_referenced() and s.is_global() and s.get_name() not in disponiveis}
                self.assertEqual(indefinidos, set(), f"{nome}.{funcao.get_name()}")


if __name__ == "__main__":
    unittest.main()
