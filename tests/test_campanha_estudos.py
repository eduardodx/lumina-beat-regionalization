"""Consumidor dos estudos (G7), so com dados SINTETICOS: coortes, cobertura, deltas, interacao e bootstrap."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha import estudos, metricas  # noqa: E402
from eval.campanha.estudos import BASE, CASOS_PAREADOS, COORTE_COMPLETO, CONTROLES, REGIONALIZADO  # noqa: E402

PAINEIS = ("missense", "splice", "noncoding", "plof", "synonymous")


def _membros(n_pares=60, n_sem_par=6, seed=0, estudo=estudos.ESTUDO_CLINICO, mesmo_cluster_a_cada=5):
    """Membership no formato do Mosaic: pares 1:1 com mesmo rotulo e painel, alguns casos sem par."""
    rng = np.random.default_rng(seed)
    linhas = []
    for i in range(n_pares):
        y, painel = int(rng.random() < 0.5), PAINEIS[i % len(PAINEIS)]
        cluster_do_caso = f"c{i // 2}"
        cluster_do_controle = cluster_do_caso if i % mesmo_cluster_a_cada == 0 else f"k{i // 2}"
        comum = {"study_id": estudo, "binary_label": y, "primary_panel": painel}
        linhas.append({**comum, "variant_id": f"caso{i}", "member_role": "case", "matched_variant_id": f"ctrl{i}",
                       "overlap_cluster_id": cluster_do_caso, "present_abraom": i % 4 == 0})
        linhas.append({**comum, "variant_id": f"ctrl{i}", "member_role": "control", "matched_variant_id": f"caso{i}",
                       "overlap_cluster_id": cluster_do_controle, "present_abraom": i % 6 == 0})
    for j in range(n_sem_par):
        linhas.append({"study_id": estudo, "binary_label": j % 2, "primary_panel": PAINEIS[j % 3],
                       "variant_id": f"solto{j}", "member_role": "unmatched_case", "matched_variant_id": None,
                       "overlap_cluster_id": f"s{j}", "present_abraom": j % 2 == 0})
    return pd.DataFrame(linhas)


def _pontos(membros, seed=1, ruido_base=0.8, ruido_reg=0.8):
    rng = np.random.default_rng(seed)
    y = membros["binary_label"].to_numpy(dtype=float)
    ids = membros["variant_id"].astype(str).to_numpy()
    base = 1 / (1 + np.exp(-(2 * y - 1 + ruido_base * rng.normal(size=len(y)))))
    reg = 1 / (1 + np.exp(-(2 * y - 1 + ruido_reg * rng.normal(size=len(y)))))
    return {BASE: pd.Series(base, index=ids), REGIONALIZADO: pd.Series(reg, index=ids)}


class VisoesTests(unittest.TestCase):
    def test_separa_os_papeis_como_o_mosaic(self):
        v = estudos.visoes(_membros(), estudos.ESTUDO_CLINICO)
        self.assertEqual((len(v[COORTE_COMPLETO]), len(v[CASOS_PAREADOS]), len(v[CONTROLES])), (66, 60, 60))
        self.assertEqual(set(v[COORTE_COMPLETO]["member_role"]), {"case", "unmatched_case"})

    def test_recusa_pareamento_quebrado(self):
        casos = {
            "caso aponta para outro caso": lambda m: m.loc[m.variant_id == "caso0", "matched_variant_id"].replace(
                "ctrl0", "caso1"),
            "par nao bidirecional": lambda m: m.loc[m.variant_id == "ctrl0", "matched_variant_id"].replace(
                "caso0", "caso1"),
            "unmatched com par": lambda m: m.loc[m.variant_id == "solto0", "matched_variant_id"].fillna("ctrl3"),
        }
        for nome, alteracao in casos.items():
            m = _membros()
            m.loc[alteracao(m).index, "matched_variant_id"] = alteracao(m)
            with self.assertRaises(estudos.EstudoInvalido, msg=nome):
                estudos.visoes(m, estudos.ESTUDO_CLINICO)

    def test_recusa_variante_repetida_cluster_vazio_e_presenca_nula(self):
        m = _membros()
        repetida = pd.concat([m, m.iloc[[0]]], ignore_index=True)
        sem_cluster = m.copy()
        sem_cluster.loc[3, "overlap_cluster_id"] = None
        presenca_nula = m.copy().astype({"present_abraom": "object"})
        presenca_nula.loc[5, "present_abraom"] = None
        for frame in (repetida, sem_cluster, presenca_nula):
            with self.assertRaises(estudos.EstudoInvalido):
                estudos.visoes(frame, estudos.ESTUDO_CLINICO)


class MetricasTests(unittest.TestCase):
    def test_delta_na_intersecao_e_cobertura_de_cada_sistema(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        base = np.array([0.1, 0.9, np.nan, 0.8, 0.3, 0.7, 0.2, 0.6])
        reg = np.array([0.2, 0.8, 0.4, np.nan, 0.1, 0.9, 0.3, 0.7])
        r = estudos.comparar(y, base, reg)
        ambos = np.isfinite(base) & np.isfinite(reg)
        self.assertEqual(r["intersecao"]["n_scored"], int(ambos.sum()))
        self.assertEqual(r["cobertura"][BASE]["n_scored"], 7)
        self.assertEqual(r["cobertura"][REGIONALIZADO]["n_scored"], 7)
        esperado = metricas.auroc(reg[ambos], y[ambos]) - metricas.auroc(base[ambos], y[ambos])
        self.assertAlmostEqual(r["delta"]["auroc"], esperado)

    def test_uma_classe_so_nao_tem_auroc_nem_delta(self):
        r = estudos.comparar(np.zeros(5, dtype=int), np.linspace(0, 1, 5), np.linspace(1, 0, 5))
        self.assertIsNone(r[BASE]["auroc"])
        self.assertIsNone(r["delta"]["auroc"])

    def test_auroc_e_auprc_sao_as_definicoes_do_mosaic(self):
        # AUROC = Mann-Whitney com empate meio ponto (roc_auc_score); AUPRC = precisao media em degraus, com
        # empates num limiar so (average_precision_score). Conferido contra a definicao, sem sklearn.
        rng = np.random.default_rng(3)
        for _ in range(20):
            n = int(rng.integers(8, 40))
            y = rng.integers(0, 2, size=n)
            if y.min() == y.max():
                continue
            s = np.round(rng.random(n), 1)   # muitos empates de proposito
            pares = [(sp > sn) + 0.5 * (sp == sn) for sp in s[y == 1] for sn in s[y == 0]]
            self.assertAlmostEqual(metricas.auroc(s, y), float(np.mean(pares)), places=12)
            ap, revocacao_anterior = 0.0, 0.0
            for limiar in sorted(set(s), reverse=True):
                previstos = s >= limiar
                vp = int((previstos & (y == 1)).sum())
                revocacao = vp / int((y == 1).sum())
                ap += (revocacao - revocacao_anterior) * vp / int(previstos.sum())
                revocacao_anterior = revocacao
            self.assertAlmostEqual(metricas.auprc(s, y), ap, places=12)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            return
        y = rng.integers(0, 2, size=300)
        s = np.round(rng.random(300), 2)
        self.assertAlmostEqual(metricas.auroc(s, y), roc_auc_score(y, s), places=12)
        self.assertAlmostEqual(metricas.auprc(s, y), average_precision_score(y, s), places=12)

    def test_limiar_segue_a_regra_do_mosaic(self):
        y = np.array([1, 1, 0, 0])
        r = estudos.com_limiar(y, np.array([0.9, 0.4, 0.4, 0.1]), 0.4)
        self.assertEqual((r["sensitivity"], r["specificity"]), (1.0, 0.5))
        self.assertEqual(estudos.com_limiar(y, np.array([0.9, 0.8, 0.7, 0.6]), 0.5)["mcc"], 0.0)
        self.assertIsNone(estudos.com_limiar(np.array([], dtype=int), np.array([]), 0.5)["mcc"])


class BootstrapTests(unittest.TestCase):
    def test_grupos_e_sorteio_mantem_o_cluster_inteiro(self):
        clusters = np.array(["a", "b", "a", "c", "b", "a"])
        g = estudos.grupos(clusters)
        self.assertEqual(sorted(sorted(x.tolist()) for x in g), [[0, 2, 5], [1, 4], [3]])
        indices = estudos.sortear(g, np.random.default_rng(0))
        for grupo in g:
            vezes = [int((indices == i).sum()) for i in grupo]
            self.assertEqual(len(set(vezes)), 1, "linhas do mesmo cluster saem juntas")

    def test_sistemas_identicos_dao_delta_zero_com_ic_degenerado(self):
        m = _membros()
        p = _pontos(m)
        p[REGIONALIZADO] = p[BASE].copy()
        r = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, p, replicas=60)
        for coorte in (COORTE_COMPLETO, CASOS_PAREADOS, CONTROLES):
            delta = r["coortes"][coorte]["coorte"]["delta"]["auroc"]
            self.assertEqual((delta["estimativa"], delta["p2_5"], delta["p97_5"]), (0.0, 0.0, 0.0), coorte)
        for chave in ("interacao", "interacao_sensibilidade_por_par"):
            faixa = r["interacao"]["auroc"][chave]
            self.assertEqual((faixa["estimativa"], faixa["p2_5"], faixa["p97_5"]), (0.0, 0.0, 0.0), chave)

    def test_mesmos_sorteios_para_os_dois_sistemas(self):
        # Transformacao monotona: mesma ordem, AUROC igual em TODA replica -> delta sempre zero, embora a AUROC de
        # cada sistema varie entre replicas. Sorteios diferentes por sistema dariam um IC largo para o delta.
        m = _membros(n_pares=90)
        p = _pontos(m)
        p[REGIONALIZADO] = p[BASE] ** 3
        r = estudos.analise_do_coorte(estudos.visoes(m, estudos.ESTUDO_CLINICO)[COORTE_COMPLETO], p, replicas=80)
        self.assertLess(r["coorte"][BASE]["auroc"]["p2_5"], r["coorte"][BASE]["auroc"]["p97_5"])
        delta = r["coorte"]["delta"]["auroc"]
        self.assertEqual((delta["p2_5"], delta["p97_5"]), (0.0, 0.0))

    def test_interacao_e_a_diferenca_dos_deltas(self):
        m = _membros(n_pares=90, seed=2)
        r = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m, ruido_reg=0.5), replicas=40)
        for k in ("auroc", "auprc"):
            dm = r["coortes"][CASOS_PAREADOS]["coorte"]["delta"][k]["estimativa"]
            dc = r["coortes"][CONTROLES]["coorte"]["delta"][k]["estimativa"]
            self.assertAlmostEqual(r["interacao"][k]["interacao"]["estimativa"], dm - dc, places=12)
            self.assertAlmostEqual(r["interacao"][k]["delta_br_matched"]["estimativa"], dm, places=12)

    def test_sorteio_conjunto_separa_casos_e_controles(self):
        # Sistemas iguais nos CONTROLES e diferentes nos casos: se a uniao misturasse os papeis, o delta do
        # controle variaria entre replicas.
        m = _membros(n_pares=80, seed=4)
        p = _pontos(m, ruido_reg=0.3)
        controles = m.loc[m["member_role"] == "control", "variant_id"]
        p[REGIONALIZADO].loc[controles] = p[BASE].loc[controles]
        v = estudos.visoes(m, estudos.ESTUDO_CLINICO)
        r = estudos.interacao(v[CASOS_PAREADOS], v[CONTROLES], p, replicas=50)
        faixa = r["auroc"]["delta_control"]
        self.assertEqual((faixa["estimativa"], faixa["p2_5"], faixa["p97_5"]), (0.0, 0.0, 0.0))
        self.assertEqual(faixa["replicas_validas"], 50)
        self.assertNotEqual(r["auroc"]["delta_br_matched"]["p2_5"], r["auroc"]["delta_br_matched"]["p97_5"])

    def test_metrica_indefinida_fica_fora_do_ic(self):
        m = _membros(n_pares=30)
        m.loc[m["member_role"] == "control", "binary_label"] = 0
        m.loc[m["member_role"] == "case", "binary_label"] = 0   # pares mantem o mesmo rotulo
        r = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m), replicas=20)
        self.assertIsNone(r["interacao"]["auroc"]["interacao"]["estimativa"])
        self.assertEqual(r["interacao"]["auroc"]["interacao"]["replicas_validas"], 0)


class AnalisesTests(unittest.TestCase):
    def test_cobertura_incompleta_entra_na_cobertura_e_nao_e_imputada(self):
        m = _membros()
        p = _pontos(m)
        p[REGIONALIZADO] = p[REGIONALIZADO].drop(["caso1", "caso2"])
        r = estudos.analise_do_coorte(estudos.visoes(m, estudos.ESTUDO_CLINICO)[COORTE_COMPLETO], p, replicas=5)
        self.assertEqual(r["coorte"]["cobertura"][REGIONALIZADO]["n_scored"], 64)
        self.assertEqual(r["coorte"]["cobertura"][BASE]["n_scored"], 66)
        self.assertEqual(r["coorte"]["intersecao"]["n_scored"], 64)

    def test_paineis_diagnosticos_e_delta_sem_cada_painel(self):
        m = _membros(n_pares=100)
        completo = estudos.visoes(m, estudos.ESTUDO_CLINICO)[COORTE_COMPLETO]
        r = estudos.analise_do_coorte(completo, _pontos(m), replicas=5, por_painel=True)
        self.assertIsNone(r["painel:plof"][BASE]["auroc"]["estimativa"], "plof e guarda: sem AUROC")
        self.assertIsNotNone(r["painel:missense"][BASE]["auroc"]["estimativa"])
        n_missense = int((completo["primary_panel"] == "missense").sum())
        self.assertEqual(r["sem_painel:missense"]["intersecao"]["n_total"], len(completo) - n_missense)

    def test_subconjuntos_so_no_estudo_clinico(self):
        m = _membros()
        clinico = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m), replicas=5)
        self.assertEqual(clinico["subconjuntos"]["present_abraom"]["composicao"]["n"],
                         int(m.loc[m["member_role"].isin(["case", "unmatched_case"]), "present_abraom"].sum()))
        pop = m.assign(study_id=estudos.ESTUDO_POPULACIONAL)
        self.assertNotIn("subconjuntos", estudos.avaliar_estudo(pop, estudos.ESTUDO_POPULACIONAL, _pontos(pop),
                                                                  replicas=5))

    def test_filtrar_pares_nunca_desfaz_um_par(self):
        m = _membros(n_pares=40)
        v = estudos.visoes(m, estudos.ESTUDO_CLINICO)
        manter = np.arange(40) % 3 == 0
        casos, controles = estudos.filtrar_pares(v[CASOS_PAREADOS], v[CONTROLES], manter)
        self.assertEqual(len(casos), int(manter.sum()))
        self.assertEqual(set(controles["variant_id"]), set(casos["matched_variant_id"]))

    def test_pares_ambos_fora_do_abraom_nao_desfazem_pares(self):
        m = _membros(n_pares=120)
        v = estudos.visoes(m, estudos.ESTUDO_CLINICO)
        r = estudos.sensibilidades(estudos.ESTUDO_CLINICO, v[CASOS_PAREADOS], v[CONTROLES], _pontos(m), replicas=5,
                                   seed=1, controles_com_scv_brasileira=None, exposicao=None,
                                   tolerancia_de_exposicao=0)["pares_ambos_fora_do_abraom"]
        esperados = [i for i in range(120) if not (i % 4 == 0) and not (i % 6 == 0)]
        self.assertEqual(r["pares_mantidos"], len(esperados))
        self.assertEqual(r["pares_retirados"], 120 - len(esperados))
        self.assertEqual(r["interacao"]["pares"], len(esperados))

    def test_sem_controles_com_scv_brasileira_tira_o_par_inteiro(self):
        m = _membros(n_pares=50)
        v = estudos.visoes(m, estudos.ESTUDO_CLINICO)
        r = estudos.sensibilidades(estudos.ESTUDO_CLINICO, v[CASOS_PAREADOS], v[CONTROLES], _pontos(m), replicas=5,
                                   seed=1, controles_com_scv_brasileira={"ctrl3", "ctrl7", "caso9"},
                                   exposicao=None, tolerancia_de_exposicao=0)["sem_controles_com_scv_brasileira"]
        # So CONTROLES da lista contam: `caso9` nao retira nada.
        self.assertEqual((r["pares_retirados"], r["pares_mantidos"]), (2, 48))

    def test_exposicao_empatada_e_faltante(self):
        m = _membros(n_pares=30)
        v = estudos.visoes(m, estudos.ESTUDO_CLINICO)
        ids = m["variant_id"].astype(str)
        exposicao = pd.Series([0.0] * len(ids), index=ids)
        exposicao.loc[["caso1", "caso2", "ctrl5"]] = [3.0, 1.0, 2.0]
        args = dict(replicas=5, seed=1, controles_com_scv_brasileira=None, tolerancia_de_exposicao=0)
        r = estudos.sensibilidades(estudos.ESTUDO_CLINICO, v[CASOS_PAREADOS], v[CONTROLES], _pontos(m),
                                   exposicao=exposicao, **args)["exposicao_empatada"]
        self.assertEqual((r["pares_retirados"], r["pares_mantidos"]), (3, 27))
        with self.assertRaises(estudos.EstudoInvalido):
            estudos.sensibilidades(estudos.ESTUDO_CLINICO, v[CASOS_PAREADOS], v[CONTROLES], _pontos(m),
                                   exposicao=exposicao.drop("ctrl4"), **args)

    def test_analise_declarada_sem_entrada_fica_registrada(self):
        m = _membros()
        r = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m), replicas=5)["sensibilidades"]
        self.assertIn("nao_calculada", r["sem_controles_com_scv_brasileira"])
        self.assertIn("nao_calculada", r["exposicao_empatada"])
        self.assertIn("nao_necessaria", r["pares_completos_na_cobertura"])
        pop = m.assign(study_id=estudos.ESTUDO_POPULACIONAL)
        r_pop = estudos.avaliar_estudo(pop, estudos.ESTUDO_POPULACIONAL, _pontos(pop), replicas=5)["sensibilidades"]
        self.assertIn("nao_calculada", r_pop["pares_ambos_fora_do_abraom"])

    def test_cobertura_que_desfaz_pares_e_contada_e_tem_sensibilidade(self):
        m = _membros(n_pares=40)
        p = _pontos(m)
        p[BASE] = p[BASE].drop(["ctrl4"])
        r = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, p, replicas=5)
        self.assertEqual(r["interacao"]["pares_com_os_dois_membros_cobertos"], 39)
        self.assertTrue(r["interacao"]["cobertura_desfaz_pares"])
        self.assertEqual(r["sensibilidades"]["pares_completos_na_cobertura"]["pares_mantidos"], 39)

    def test_limiares_so_quando_declarados(self):
        m = _membros()
        sem = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m), replicas=5)
        self.assertNotIn("mcc", sem["coortes"][COORTE_COMPLETO]["coorte"]["delta"])
        self.assertTrue(sem["metricas_com_limiar"].startswith("omitidas"))
        com = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m), replicas=5,
                                     limiares={BASE: 0.5, REGIONALIZADO: 0.5})
        self.assertIn("mcc", com["coortes"][COORTE_COMPLETO]["coorte"]["delta"])

    def test_mesma_seed_mesmo_resultado(self):
        m = _membros()
        a = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m), replicas=15)
        b = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, _pontos(m), replicas=15)
        self.assertEqual(a["interacao"], b["interacao"])


if __name__ == "__main__":
    unittest.main()
