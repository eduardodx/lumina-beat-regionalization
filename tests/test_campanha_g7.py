"""G7, regras puras (sem torch): Brier, baselines, tabela dos estudos, identidade dos caches e margens."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha import baselines, estudos, g7  # noqa: E402
from eval.campanha.estudos import BASE, CASOS_PAREADOS, CONTROLES, COORTE_COMPLETO, REGIONALIZADO  # noqa: E402

PAINEIS = ("missense", "splice", "noncoding", "plof", "synonymous")


def membros_sinteticos(n_pares=40, n_sem_par=4, seed=0):
    """Os dois estudos no formato do Mosaic: clinico com pares e casos sem par; populacional com casos presentes no
    ABraOM e controles ausentes (como `build_brazil_membership`)."""
    rng = np.random.default_rng(seed)
    linhas = []
    for estudo, prefixo in ((estudos.ESTUDO_CLINICO, "c"), (estudos.ESTUDO_POPULACIONAL, "p")):
        for i in range(n_pares):
            y, painel = int(rng.random() < 0.5), PAINEIS[i % len(PAINEIS)]
            populacional = estudo == estudos.ESTUDO_POPULACIONAL
            comum = {"study_id": estudo, "binary_label": y, "primary_panel": painel, "label_tier": "gold",
                     "chrom": "chr8" if i == 0 else "chr1", "ref": "A", "alt": "G"}
            linhas.append({**comum, "variant_id": f"{prefixo}caso{i}", "member_role": "case",
                           "matched_variant_id": f"{prefixo}ctrl{i}", "overlap_cluster_id": f"{prefixo}k{i // 2}",
                           "present_abraom": True if populacional else i % 4 == 0, "pos_1based": 1000 + 10 * i})
            linhas.append({**comum, "variant_id": f"{prefixo}ctrl{i}", "member_role": "control",
                           "matched_variant_id": f"{prefixo}caso{i}", "overlap_cluster_id": f"{prefixo}q{i // 2}",
                           "present_abraom": False if populacional else i % 7 == 0, "pos_1based": 5000 + 10 * i})
        for j in range(n_sem_par):
            linhas.append({"study_id": estudo, "binary_label": j % 2, "primary_panel": PAINEIS[j % 3],
                           "label_tier": "gold", "chrom": "chr2", "ref": "C", "alt": "T", "pos_1based": 9000 + j,
                           "variant_id": f"{prefixo}solto{j}", "member_role": "unmatched_case",
                           "matched_variant_id": None, "overlap_cluster_id": f"{prefixo}s{j}",
                           "present_abraom": estudo == estudos.ESTUDO_POPULACIONAL or j % 2 == 0})
    return pd.DataFrame(linhas)


def anotacoes_sinteticas(membros, seed=1):
    rng = np.random.default_rng(seed)
    unicos = membros.drop_duplicates("variant_id")
    status = rng.choice(["present", "not_found", "ac0"], size=len(unicos), p=[0.7, 0.25, 0.05])
    af = np.where(status == "present", rng.random(len(unicos)) * 1e-3, np.where(status == "ac0", 0.0, np.nan))
    presente = unicos["present_abraom"].astype(bool).to_numpy()
    return pd.DataFrame({"variant_id": unicos["variant_id"].to_numpy(), "gnomad_v4_af": af, "gnomad_status": status,
                         "abraom_af": np.where(presente, rng.random(len(unicos)) * 1e-2, np.nan),
                         "abraom_status": np.where(presente, "present", "not_found"), "present_abraom": presente})


def pontuar_gnomad(linha):
    """A formula do gnomad_rarity (comparators.yaml) escrita a parte."""
    af = 0.0 if linha.get("gnomad_status") in ("not_found", "ac0") else linha.get("gnomad_v4_af")
    return None if af is None or af != af else -float(af)


class BrierTests(unittest.TestCase):
    def test_brier_e_o_erro_quadratico_medio(self):
        self.assertAlmostEqual(estudos.brier(np.array([1, 0, 1]), np.array([0.9, 0.2, 0.5])),
                               (0.01 + 0.04 + 0.25) / 3)
        self.assertIsNone(estudos.brier(np.array([]), np.array([])))

    def test_brier_sai_nos_coortes_e_com_delta(self):
        m = membros_sinteticos()
        p = g7.pontos_sinteticos(m, seed=3)
        r = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, p, replicas=10, com_brier=True)
        for coorte in (COORTE_COMPLETO, CASOS_PAREADOS, CONTROLES):
            celula = r["coortes"][coorte]["coorte"]
            self.assertIn("brier", celula["delta"])
            self.assertAlmostEqual(celula["delta"]["brier"]["estimativa"],
                                   celula[REGIONALIZADO]["brier"]["estimativa"] - celula[BASE]["brier"]["estimativa"])
            self.assertIn("painel:missense", r["coortes"][coorte], "paineis nos tres coortes")
        self.assertNotIn("brier", r["interacao"])

    def test_brier_so_sobre_probabilidades(self):
        m = membros_sinteticos()
        p = g7.pontos_sinteticos(m, seed=3)
        p[REGIONALIZADO] = -p[REGIONALIZADO]   # um score de ordenacao como -AF
        with self.assertRaises(estudos.EstudoInvalido):
            estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, p, replicas=5, com_brier=True)
        r = estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, p, replicas=5)   # sem Brier, aceita
        self.assertNotIn("brier", r["coortes"][COORTE_COMPLETO]["coorte"]["delta"])


class BaselinesTests(unittest.TestCase):
    def test_regras_e_imputacao(self):
        m = membros_sinteticos()
        a = anotacoes_sinteticas(m)
        s = baselines.scores_das_baselines(m, a, pontuar_gnomad)
        por_id = a.set_index("variant_id")
        for vid in por_id.index[:30]:
            linha = por_id.loc[vid]
            esperado_gnomad = 0.0 if linha["gnomad_status"] in ("not_found", "ac0") else -linha["gnomad_v4_af"]
            self.assertAlmostEqual(s[baselines.GNOMAD_RARITY][vid], esperado_gnomad)
            esperado_abraom = -linha["abraom_af"] if linha["present_abraom"] else 0.0
            self.assertAlmostEqual(s[baselines.RARIDADE_NO_ABRAOM][vid], esperado_abraom)
            self.assertEqual(s[baselines.AUSENCIA_NO_ABRAOM][vid], 0.0 if linha["present_abraom"] else 1.0)

    def test_frequencia_observada_separa_medido_de_imputado(self):
        m = membros_sinteticos()
        a = anotacoes_sinteticas(m)
        f = baselines.frequencia_observada(m, a)
        controle = f[f"{estudos.ESTUDO_POPULACIONAL}/control"]
        self.assertEqual(controle["abraom"]["af_observada"], 0)
        self.assertEqual(controle["abraom"]["ausente_af_imputada"], controle["membros"])
        g = f[f"{estudos.ESTUDO_CLINICO}/case"]["gnomad"]
        self.assertEqual(g["af_observada"] + g["ac0_sitio_chamado"] + g["not_found_af_imputada"] + g["outro_status"],
                         f[f"{estudos.ESTUDO_CLINICO}/case"]["membros"])

    def test_membro_sem_anotacao_ou_presenca_discordante_e_recusado(self):
        m = membros_sinteticos()
        a = anotacoes_sinteticas(m)
        with self.assertRaises(ValueError):
            baselines.scores_das_baselines(m, a.iloc[1:], pontuar_gnomad)
        a.loc[0, "present_abraom"] = not a.loc[0, "present_abraom"]
        with self.assertRaises(ValueError):
            baselines.scores_das_baselines(m, a, pontuar_gnomad)

    def test_baseline_constante_e_nao_aplicavel_saem_sem_metrica(self):
        m = membros_sinteticos()
        a = anotacoes_sinteticas(m)
        s = baselines.scores_das_baselines(m, a, pontuar_gnomad)
        esp = baselines.ESPECIFICACOES
        populacional = estudos.avaliar_baseline(m, estudos.ESTUDO_POPULACIONAL, s[baselines.RARIDADE_NO_ABRAOM],
                                                especificacao=esp[baselines.RARIDADE_NO_ABRAOM], replicas=10)
        controles = populacional["coortes"][CONTROLES]["coorte"]
        self.assertTrue(controles["constante"])
        # Constante com as duas classes: AUROC 0,5 e AUPRC = prevalencia, definidas (revisao de 24/09).
        y = m[(m["study_id"] == estudos.ESTUDO_POPULACIONAL) & (m["member_role"] == "control")]["binary_label"]
        self.assertEqual(controles["auroc"]["estimativa"], 0.5)
        self.assertAlmostEqual(controles["auprc"]["estimativa"], float(y.mean()))
        self.assertEqual((controles["auroc"]["p2_5"], controles["auroc"]["p97_5"]), (0.5, 0.5))
        self.assertEqual(controles["auroc"]["replicas_validas"], 10, "replica constante nao some do IC")
        self.assertIn("nao discrimina", controles["motivo"])
        self.assertFalse(populacional["coortes"][COORTE_COMPLETO]["coorte"]["constante"])
        self.assertEqual(populacional["sem_brier"], "score de ordenacao, nao probabilidade")
        ausencia = estudos.avaliar_baseline(m, estudos.ESTUDO_POPULACIONAL, s[baselines.AUSENCIA_NO_ABRAOM],
                                            especificacao=esp[baselines.AUSENCIA_NO_ABRAOM], replicas=10)
        self.assertIn("define os grupos", ausencia["nao_aplicavel"])
        self.assertNotIn("coortes", ausencia)
        clinico = estudos.avaliar_baseline(m, estudos.ESTUDO_CLINICO, s[baselines.AUSENCIA_NO_ABRAOM],
                                           especificacao=esp[baselines.AUSENCIA_NO_ABRAOM], replicas=10)
        self.assertIn("diferenca_casos_pareados_menos_controles", clinico)

    def test_diferenca_entre_grupos_com_o_mesmo_score_nos_dois_grupos_e_zero(self):
        m = membros_sinteticos(n_pares=60)
        v = estudos.visoes(m, estudos.ESTUDO_CLINICO)
        casos, controles = v[CASOS_PAREADOS], v[CONTROLES]
        # controle com o mesmo rotulo do caso (o pareamento casa o rotulo) e o mesmo score: diferenca 0 exata
        valores = np.random.default_rng(0).random(len(casos))
        score = pd.Series(np.concatenate([valores, valores]),
                          index=np.concatenate([casos["variant_id"].astype(str),
                                                casos["matched_variant_id"].astype(str)]))
        r = estudos.diferenca_entre_grupos(casos, controles, score, replicas=10)
        self.assertEqual(r["auroc"]["estimativa"], 0.0)


class TabelaEIdentidadeTests(unittest.TestCase):
    def test_uma_linha_por_variante_com_papel_estudo(self):
        m = membros_sinteticos()
        dobrada = pd.concat([m, m[m["variant_id"] == "ccaso1"].assign(study_id=estudos.ESTUDO_POPULACIONAL)])
        t = g7.tabela_dos_estudos(dobrada)
        self.assertEqual(len(t), m["variant_id"].nunique())
        self.assertEqual(set(t["papel"]), {"estudo"})
        self.assertIn("chr8", set(t["chrom"]), "chr8 entra no G7")
        self.assertEqual(list(t["variant_id"]), sorted(t["variant_id"]))

    def test_variante_com_atributos_divergentes_e_recusada(self):
        m = membros_sinteticos()
        errada = pd.concat([m, m[m["variant_id"] == "ccaso1"].assign(binary_label=lambda d: 1 - d["binary_label"])])
        with self.assertRaises(g7.G7Invalido):
            g7.tabela_dos_estudos(errada)
        with self.assertRaises(g7.G7Invalido):
            g7.tabela_dos_estudos(m.drop(columns=["pos_1based"]))

    def test_tabela_oficial_vem_da_membership_e_do_pb_examples(self):
        m = membros_sinteticos()
        membership = m.drop(columns=["chrom", "pos_1based", "ref", "alt"])
        exemplos = pd.concat([m.drop_duplicates("variant_id")[["variant_id", "chrom", "pos_1based", "ref", "alt",
                                                                "binary_label", "label_tier"]],
                              pd.DataFrame({"variant_id": ["fora"], "chrom": ["chr3"], "pos_1based": [1], "ref": ["A"],
                                            "alt": ["C"], "binary_label": [0], "label_tier": ["gold"]})])
        oficial = g7.tabela_oficial(membership, exemplos)
        self.assertEqual(g7.diferencas_de_tabela(oficial, g7.tabela_dos_estudos(m)), [])
        # Mesmos ids, outra sequencia: um arquivo intermediario assim nao passa.
        alterado = m.copy()
        alterado.loc[alterado["variant_id"] == "ccaso3", "alt"] = "T"
        self.assertEqual(g7.diferencas_de_tabela(oficial, g7.tabela_dos_estudos(alterado)), ["alt"])
        with self.assertRaises(g7.G7Invalido):
            g7.tabela_oficial(membership, exemplos[exemplos["variant_id"] != "ccaso3"])
        rotulo_errado = exemplos.copy()
        rotulo_errado.loc[rotulo_errado["variant_id"] == "ccaso3", "binary_label"] = 7
        with self.assertRaises(g7.G7Invalido):
            g7.tabela_oficial(membership, rotulo_errado)

    def test_identidade_so_pode_diferir_na_tabela(self):
        dev = {"codigo": {"x": 1}, "ambiente": {"gpu": "L4"}, "tabela_sha256_conteudo": "a", "papeis": ["train"],
               "revisao_do_codigo": "r1", "adapter_sha256": "z"}
        estudo = dict(dev, tabela_sha256_conteudo="b", papeis=["estudo"], revisao_do_codigo="r2")
        self.assertEqual(g7.diferencas_do_estudo(dev, estudo), [])
        self.assertEqual(g7.diferencas_do_estudo(dev, dict(estudo, ambiente={"gpu": "A10"})), ["ambiente"])
        self.assertEqual(g7.diferencas_do_estudo(dev, dict(estudo, adapter_sha256="y")), ["adapter_sha256"])

    def test_pontos_sinteticos_sao_probabilidades(self):
        p = g7.pontos_sinteticos(membros_sinteticos(), seed=1)
        for serie in p.values():
            self.assertTrue(((serie > 0) & (serie < 1)).all())
            self.assertFalse(serie.index.has_duplicates)


def _condicao(limite, estatistica="estimativa", comparacao=">="):
    return [{"estatistica": estatistica, "comparacao": comparacao, "limite": limite}]


def _margens(**troca):
    def regra(limite, **extra):
        return {"estudos": [estudos.ESTUDO_CLINICO], "metrica": "auroc", "condicoes": _condicao(limite), **extra}
    m = {"estado": "DECLARADO (teste)",
         "papel_dos_estudos": {estudos.ESTUDO_CLINICO: "exigido", estudos.ESTUDO_POPULACIONAL: "descritivo"},
         "melhoria_minima_no_coorte_br": regra(0.01, delta="delta_br_full"),
         "regressao_maxima_no_controle": regra(-0.01, delta="delta_control"),
         "paineis_com_regressao_inaceitavel": regra(-0.02, delta="delta_br_full", paineis=["missense"]),
         "beneficio_nao_explicado_por_um_painel": regra(0.0, delta="delta_br_full", suporte_minimo_por_painel=1),
         "interacao": {"criterio_proprio": True, "estudos": [estudos.ESTUDO_CLINICO], "metrica": "auroc",
                       "condicoes": _condicao(0.0)}}
    m.update(troca)
    return m


BOOTSTRAP = {"estado": "DECLARADO (teste)", "unidade_principal": "cluster_conjunto",
             "unidade_de_sensibilidade": "par", "replicas": 1000, "seed": 20260901, "percentis": [2.5, 97.5]}


class MargensTests(unittest.TestCase):
    def setUp(self):
        m = membros_sinteticos(n_pares=60)
        p = g7.pontos_sinteticos(m, seed=2)
        self.r = {estudos.ESTUDO_CLINICO: estudos.avaliar_estudo(m, estudos.ESTUDO_CLINICO, p, replicas=10)}

    def test_margens_nao_declaradas_nao_sao_avaliadas(self):
        saida = g7.avaliar_margens(self.r, {"estado": "ABERTO"}, BOOTSTRAP)
        self.assertFalse(saida["avaliado"])

    def test_regras_aplicadas_ao_delta_declarado(self):
        saida = g7.avaliar_margens(self.r, _margens(), BOOTSTRAP)
        regras = saida["por_estudo"][estudos.ESTUDO_CLINICO]["regras"]
        delta = self.r[estudos.ESTUDO_CLINICO]["coortes"][COORTE_COMPLETO]["coorte"]["delta"]["auroc"]["estimativa"]
        self.assertEqual(regras["melhoria_minima_no_coorte_br"]["condicoes"][0]["valor"], delta)
        self.assertEqual(regras["melhoria_minima_no_coorte_br"]["atende"], delta >= 0.01)
        controle = self.r[estudos.ESTUDO_CLINICO]["coortes"][CONTROLES]["coorte"]["delta"]["auroc"]["estimativa"]
        self.assertEqual(regras["regressao_maxima_no_controle"]["condicoes"][0]["valor"], controle)
        interacao = self.r[estudos.ESTUDO_CLINICO]["interacao"]["auroc"]["interacao"]["estimativa"]
        self.assertEqual(regras["interacao"]["condicoes"][0]["valor"], interacao)
        self.assertTrue(regras["beneficio_nao_explicado_por_um_painel"]["aplicavel"])
        self.assertNotIn(estudos.ESTUDO_POPULACIONAL, saida["por_estudo"])
        self.assertEqual(saida["sucesso"]["estudos_exigidos"], [estudos.ESTUDO_CLINICO])
        self.assertEqual(saida["sucesso"]["atende"], saida["por_estudo"][estudos.ESTUDO_CLINICO]["atende_todas"])

    def test_interacao_separada_nao_entra_no_sucesso(self):
        # Revisao de 25/09: ganho de +0,015 no BR com +0,030 no controle pode passar no criterio clinico; a vantagem
        # diferencial so pode ser afirmada pela regra separada da interacao, que sai a parte do sucesso.
        exigida = g7.avaliar_margens(self.r, _margens(), BOOTSTRAP)
        interacao = dict(_margens()["interacao"], papel="separado", afirmacao="vantagem diferencial (teste)")
        separada = g7.avaliar_margens(self.r, _margens(interacao=interacao), BOOTSTRAP)
        regras = separada["por_estudo"][estudos.ESTUDO_CLINICO]["regras"]
        self.assertNotIn("interacao", regras)
        self.assertEqual(sorted(regras), sorted(set(exigida["por_estudo"][estudos.ESTUDO_CLINICO]["regras"]) -
                                                {"interacao"}))
        vantagem = separada["vantagem_regional"]
        self.assertTrue(vantagem["avaliada"])
        self.assertEqual(vantagem["estudos"], [estudos.ESTUDO_CLINICO])
        self.assertEqual(vantagem["afirmacao_permitida"], "vantagem diferencial (teste)")
        esperado = self.r[estudos.ESTUDO_CLINICO]["interacao"]["auroc"]["por_unidade"]["cluster_conjunto"]["interacao"]
        avaliada = vantagem["por_estudo"][estudos.ESTUDO_CLINICO]
        self.assertEqual(avaliada["condicoes"][0]["valor"], esperado["estimativa"])
        self.assertEqual(vantagem["atende"], avaliada["atende"])
        self.assertEqual(separada["sucesso"]["atende"],
                         g7._combinar(list(regras.values())), "o sucesso so combina as regras do clinico")
        self.assertFalse(exigida["vantagem_regional"]["avaliada"], "interacao exigida entra no sucesso, nao a parte")

    def test_o_sucesso_carrega_a_afirmacao_declarada(self):
        declarado = {"afirmacao_permitida": "criterio clinico (teste)", "nao_permite": ["vantagem regional (teste)"]}
        saida = g7.avaliar_margens(self.r, _margens(sucesso=declarado), BOOTSTRAP)
        self.assertEqual(saida["sucesso"]["afirmacao_permitida"], "criterio clinico (teste)")
        self.assertEqual(saida["sucesso"]["nao_permite"], ["vantagem regional (teste)"])
        self.assertIsNone(g7.avaliar_margens(self.r, _margens(), BOOTSTRAP)["sucesso"]["afirmacao_permitida"])

    def test_as_duas_formas_da_revisao_dao_resultados_diferentes(self):
        forma_a = {"condicoes": _condicao(0.02) + _condicao(0.0, "p2_5", ">")}
        forma_b = {"condicoes": _condicao(0.02, "p2_5")}
        celula = {"estimativa": 0.03, "p2_5": 0.005}
        self.assertTrue(g7._regra(celula, forma_a)["atende"])
        self.assertFalse(g7._regra(celula, forma_b)["atende"], "o limite inferior abaixo de 0,02 reprova (b)")
        # "IC excluindo zero" e estrito: limite inferior exatamente 0 nao exclui zero.
        self.assertFalse(g7._regra({"estimativa": 0.03, "p2_5": 0.0}, forma_a)["atende"])
        self.assertTrue(g7._regra({"estimativa": 0.03, "p2_5": 0.0}, {"condicoes": _condicao(0.0, "p2_5")})["atende"])
        self.assertIsNone(g7._regra({"estimativa": 0.03, "p2_5": None}, forma_a)["atende"])

    def test_unidade_por_par_usa_o_ic_por_par_e_regra_quebrada_reprova(self):
        par = dict(BOOTSTRAP, unidade_principal="par", unidade_de_sensibilidade="cluster_conjunto")
        saida = g7.avaliar_margens(self.r, _margens(melhoria_minima_no_coorte_br={
            "estudos": [estudos.ESTUDO_CLINICO], "delta": "delta_br_full", "metrica": "auroc",
            "condicoes": _condicao(0.99, "p2_5")}), par)
        regras = saida["por_estudo"][estudos.ESTUDO_CLINICO]["regras"]
        self.assertFalse(regras["melhoria_minima_no_coorte_br"]["atende"])
        self.assertFalse(saida["por_estudo"][estudos.ESTUDO_CLINICO]["atende_todas"])
        esperado = self.r[estudos.ESTUDO_CLINICO]["interacao"]["auroc"]["por_unidade"]["par"]["interacao"]["estimativa"]
        self.assertEqual(regras["interacao"]["condicoes"][0]["valor"], esperado)
        self.assertFalse(saida["sucesso"]["atende"])
        self.assertEqual(regras["interacao"]["unidade"], "par")

    def test_condicao_3_sem_suporte_em_dois_paineis_nao_se_aplica(self):
        margens = _margens()
        margens["beneficio_nao_explicado_por_um_painel"] = dict(margens["beneficio_nao_explicado_por_um_painel"],
                                                                suporte_minimo_por_painel=10_000)
        regra = g7.avaliar_margens(self.r, margens, BOOTSTRAP)["por_estudo"][estudos.ESTUDO_CLINICO]["regras"][
            "beneficio_nao_explicado_por_um_painel"]
        self.assertFalse(regra["aplicavel"])
        self.assertIsNone(regra["atende"])

    def test_estudo_sem_regra_declarada_nao_atende_por_vacuidade(self):
        m = membros_sinteticos(n_pares=30)
        p = g7.pontos_sinteticos(m, seed=4)
        r = dict(self.r, **{estudos.ESTUDO_POPULACIONAL: estudos.avaliar_estudo(m, estudos.ESTUDO_POPULACIONAL, p,
                                                                                replicas=5)})
        bloco = g7.avaliar_margens(r, _margens(), BOOTSTRAP)["por_estudo"][estudos.ESTUDO_POPULACIONAL]
        self.assertIsNone(bloco["atende_todas"])
        self.assertIn("nenhuma regra", bloco["nota"])

    def test_limiares_do_manifesto(self):
        manifesto = {"sistemas": {"base": {"limiar": {"threshold": 0.47}},
                                  "regionalized": {"limiar": {"threshold": 0.55}}}}
        self.assertEqual(g7.limiares_do_manifesto(copy.deepcopy(manifesto)), {BASE: 0.47, REGIONALIZADO: 0.55})


if __name__ == "__main__":
    unittest.main()
