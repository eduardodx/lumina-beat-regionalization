"""G6, regras puras (sem torch): limiar do Mosaic, composicao, conferencias, bloqueios e manifesto.

O caminho completo (cabecas salvas, recarregadas e pontuadas sobre caches sinteticos) esta em
tests/test_construir_g6.py, que precisa de torch.
"""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha import g6  # noqa: E402
from eval.campanha.recortes import carregar_campanha  # noqa: E402


def _campanha():
    return carregar_campanha(RAIZ / "configs" / "campanha_r03_desenvolvimento.json")


def _forca_bruta(y, s):
    """A regra do Mosaic escrita do jeito mais direto possivel, para conferir a implementacao."""
    y, s = list(map(int, y)), list(map(float, s))
    distintos = sorted(set(s))
    candidatos = [float(np.nextafter(distintos[0], -np.inf)), *distintos, float(np.nextafter(distintos[-1], np.inf))]
    melhor = None
    for t in candidatos:
        vp = sum(1 for a, b in zip(y, s) if a == 1 and b >= t)
        fp = sum(1 for a, b in zip(y, s) if a == 0 and b >= t)
        vn = sum(1 for a, b in zip(y, s) if a == 0 and b < t)
        fn = sum(1 for a, b in zip(y, s) if a == 1 and b < t)
        d = (vp + fp) * (vp + fn) * (vn + fp) * (vn + fn)
        chave = ((vp * vn - fp * fn) / float(d) ** 0.5 if d else 0.0, vn / (vn + fp), t)
        if melhor is None or chave > melhor:
            melhor = chave
    return melhor


class LimiarTests(unittest.TestCase):
    def test_empate_no_mcc_fica_com_a_maior_especificidade(self):
        # 0,7 e 0,9 empatam em MCC = 2/sqrt(12); o primeiro maximo (regra das cabecas) daria 0,7.
        r = g6.limiar_do_mosaic([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
        self.assertEqual(r["threshold"], 0.9)
        self.assertEqual(r["mcc"], 2 / 12 ** 0.5)
        self.assertEqual((r["specificity"], r["sensitivity"]), (1.0, 0.5))
        self.assertEqual(r["prediction_rule"], "score >= threshold")

    def test_limiar_pode_ficar_acima_do_maior_score(self):
        # Scores invertidos: todo corte tem MCC <= 0; os de MCC 0 empatam e ganha o de especificidade 1, acima do
        # maior score (nao preve nada).
        r = g6.limiar_do_mosaic([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(r["threshold"], float(np.nextafter(0.9, np.inf)))
        self.assertEqual((r["mcc"], r["specificity"]), (0.0, 1.0))

    def test_confere_com_a_forca_bruta_com_empates(self):
        rng = np.random.default_rng(5)
        for _ in range(40):
            n = int(rng.integers(6, 60))
            y = rng.integers(0, 2, size=n)
            if y.min() == y.max():
                continue
            s = np.round(rng.random(n), int(rng.integers(1, 3)))
            r = g6.limiar_do_mosaic(y, s)
            self.assertEqual((r["mcc"], r["specificity"], r["threshold"]), _forca_bruta(y, s))

    def test_score_nao_finito_fica_fora_e_e_contado(self):
        r = g6.limiar_do_mosaic([1, 0, 1, 0, 1], [0.9, 0.2, np.nan, 0.4, 0.8])
        self.assertEqual((r["n_total"], r["n_scored"], r["n_P_scored"], r["n_B_scored"]), (5, 4, 2, 2))
        self.assertAlmostEqual(r["coverage"], 0.8)
        self.assertEqual(r["mcc"], 1.0)

    def test_uma_classe_so_e_recusada(self):
        with self.assertRaises(ValueError):
            g6.limiar_do_mosaic([1, 1, 1], [0.2, 0.5, 0.9])
        with self.assertRaises(ValueError):
            g6.limiar_do_mosaic([1, 0, 1], [0.2, np.nan, 0.9])

    def test_media_exige_os_tres_componentes(self):
        np.testing.assert_allclose(g6.media_dos_componentes([np.array([0.0, 0.3]), np.array([0.3, 0.6]),
                                                             np.array([0.6, 0.9])]), [0.3, 0.6])
        with self.assertRaises(ValueError):
            g6.media_dos_componentes([np.array([0.1]), np.array([0.2])])


class ComposicaoTests(unittest.TestCase):
    def test_a_composicao_declarada_e_o_pareamento(self):
        c = g6.componentes_declarados(_campanha())
        self.assertEqual([x["rotulo"] for x in c["M0"]], ["M0_h11", "M0_h12", "M0_h13"])
        self.assertEqual([x["rotulo"] for x in c["MR"]], ["MR_a20260921_h11", "MR_a20260922_h12", "MR_a20260923_h13"])
        self.assertEqual({x["comparador"] for x in c["M0"]}, {"comparacao_dev_a1"})
        self.assertEqual([x["comparador"] for x in c["MR"]],
                         ["comparacao_dev_a1", "comparacao_dev_a2", "comparacao_dev_a3"])
        self.assertEqual([x["coluna"] for x in c["MR"]], ["MR_h11", "MR_h12", "MR_h13"])

    def test_desvios_da_composicao_sao_recusados(self):
        base = _campanha()
        trocas = [
            lambda k: k["g6"]["composicao_final"]["MR"][1].update(cabeca=11),              # a2 + h11
            lambda k: k["g6"]["composicao_final"]["MR"].reverse(),                          # ordem trocada
            lambda k: k["g6"]["composicao_final"]["M0"][0].update(adapter=20260921),        # M0 com adapter
            lambda k: k["g6"]["composicao_final"]["MR"][2].update(
                arquivo="comparacao_dev_a3/cabeca_MR_h11.pt"),                              # arquivo de outra cabeca
            lambda k: k["g6"]["composicao_final"]["M0"][2].update(
                arquivo="comparacao_dev_a2/cabeca_M0_h13.pt"),                              # M0 de dois comparadores
        ]
        for troca in trocas:
            campanha = copy.deepcopy(base)
            troca(campanha)
            with self.assertRaises(ValueError):
                g6.componentes_declarados(campanha)


def _conferencias(componentes):
    """As tres conferencias como o conferidor as grava, coerentes com a composicao."""
    sha = {c["arquivo"]: hashlib.sha256(c["arquivo"].encode()).hexdigest() for s in g6.SISTEMAS
           for c in componentes[s]}
    conferencias = {}
    for i, nome in enumerate(("comparacao_dev_a1", "comparacao_dev_a2", "comparacao_dev_a3")):
        linhas = [{"arquivo": Path(a).name, "sha256": h, "problemas": []} for a, h in sha.items()
                  if Path(a).parts[0] == nome]
        conferencias[nome] = {"passou": True, "problemas": [], "semente_do_adapter": 20260921 + i, "cabecas": linhas,
                              "m0_de_referencia": None if i == 0 else {
                                  "pasta": "/home/x/artifacts/redesenho/comparacao_dev_a1",
                                  "cabecas": {f"cabeca_M0_h{h}.pt": "identica" for h in (11, 12, 13)}}}
    return conferencias, sha


class ConferenciasTests(unittest.TestCase):
    def setUp(self):
        self.componentes = g6.componentes_declarados(_campanha())

    def test_conferencias_coerentes_passam(self):
        conferencias, sha = _conferencias(self.componentes)
        self.assertEqual(g6.conferir_conferencias(conferencias, self.componentes, sha), [])

    def test_cada_desvio_reprova(self):
        casos = {
            "sha256": lambda c, s: s.update({"comparacao_dev_a2/cabeca_MR_h12.pt": "0" * 64}),
            "nao passou": lambda c, s: c["comparacao_dev_a3"].update(passou=False),
            "conferido contra": lambda c, s: c["comparacao_dev_a2"].update(m0_de_referencia=None),
            "identicas": lambda c, s: c["comparacao_dev_a3"]["m0_de_referencia"]["cabecas"].update(
                {"cabeca_M0_h12.pt": ["pesos diferentes"]}),
            "semente de adapter": lambda c, s: c["comparacao_dev_a2"].update(semente_do_adapter=20260921),
            "ausente da conferencia": lambda c, s: c["comparacao_dev_a1"].update(cabecas=[
                x for x in c["comparacao_dev_a1"]["cabecas"] if x["arquivo"] != "cabeca_M0_h13.pt"]),
            "sem conferencia": lambda c, s: c.pop("comparacao_dev_a3"),
        }
        for trecho, estragar in casos.items():
            conferencias, sha = _conferencias(self.componentes)
            estragar(conferencias, sha)
            problemas = g6.conferir_conferencias(conferencias, self.componentes, sha)
            self.assertTrue(any(trecho in p for p in problemas), (trecho, problemas))

    def test_diferencas_numericas(self):
        self.assertEqual(g6.diferencas_numericas({"a": 1.0, "b": None}, {"a": 1.0 + 1e-9, "b": None}, ("a", "b"),
                                                 tolerancia=1e-6, rotulo="x"), [])
        self.assertEqual(len(g6.diferencas_numericas({"a": 1.0, "b": 0.5}, {"a": 1.1, "b": None}, ("a", "b"),
                                                     tolerancia=1e-6, rotulo="x")), 2)


class EnsembleTests(unittest.TestCase):
    def _dados(self, n=240, seed=0):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.4).astype(int)
        paineis = np.array(["missense", "splice", "noncoding", "plof", "synonymous"])[np.arange(n) % 5]
        clusters = np.array([f"c{i // 3}" for i in range(n)])
        p = 1 / (1 + np.exp(-(2 * y - 1 + rng.normal(size=n))))
        return y, paineis, clusters, p

    def test_sistemas_de_mesma_ordem_dao_delta_zero_em_toda_replica(self):
        y, paineis, clusters, p = self._dados()
        limiares = {"M0": g6.limiar_do_mosaic(y, p), "MR": g6.limiar_do_mosaic(y, p ** 2)}
        r = g6.comparacao_do_ensemble(p, p ** 2, y, paineis, clusters, limiares=limiares, replicas=40, seed=1)
        for chave in ("macro", "auroc", "auprc"):
            self.assertEqual(r["delta"][chave], 0.0)
            self.assertEqual((r["bootstrap"][chave]["p2_5"], r["bootstrap"][chave]["p97_5"]), (0.0, 0.0))
        # Limiar transformado junto com o score: as metricas com limiar tambem coincidem.
        self.assertEqual(r["com_o_limiar_do_fold1"]["M0"]["mcc"], r["com_o_limiar_do_fold1"]["MR"]["mcc"])
        self.assertEqual(r["bootstrap"]["unidade"], "overlap_cluster_id")


def _entradas(campanha, bloqueios=()):
    componentes = []
    for sistema in g6.SISTEMAS:
        for c in g6.componentes_declarados(campanha)[sistema]:
            componentes.append({**c, "sha256": "f" * 64, "platt": {"a": 0.7, "b": 3.3}, "epoca": 200,
                                "cache": {"chave": "M0" if sistema == "M0" else str(c["adapter"]),
                                          "identidade_sha256": "e" * 64}})
    ref = campanha["adapter_do_mr"]["referencia"]["checkpoint_sha256"]
    caches = {"M0": {"checkpoint_sha256": ref, "adapter_sha256": None}}
    for semente in campanha["sementes"]["adapter"]:
        caches[str(semente)] = {"checkpoint_sha256": ref,
                                "adapter_sha256": campanha["adapters_congelados"][str(semente)]["sha256"]}
    limiar = {"threshold": 0.55, "mcc": 0.6, "specificity": 0.8, "sensitivity": 0.75}
    return dict(declaracao_sha256="d" * 64,
                decisao_g5={"arquivo": "g5_decisao.json", "sha256": "5" * 64, "extracao": "leitura_antiga_1344",
                            "politica": "janela2048", "snapshot_sha256": "a" * 64, "snapshot_arquivo": "x.parquet"},
                componentes=componentes, caches=caches, extracao={"extracao": "leitura_antiga_1344"},
                limiares={"M0": dict(limiar), "MR": dict(limiar, threshold=0.6)},
                fold1={"variantes": 1575, "ids_sha256": "1" * 64}, conferencias={}, proveniencia={},
                codigo={"arquivos": {}}, ambiente_da_pontuacao={"dispositivo": "cpu"}, modulos=99,
                bloqueios_atuais=list(bloqueios), criado_em_utc="2026-09-24T20:00:00Z")


CODIGO_OK = {"revisao": "0123456789abcdef0123456789abcdef01234567", "ausentes": [], "nao_rastreados": [],
             "modificados": [], "erro": None}
ABRAOM_OK = {"abraom": {"confere": True}}
ENTRADAS_OK = {"regra_ampla": {"sha256": "a" * 64}, "exposicao": {"sha256": "b" * 64}}


def _regra(estatistica="p2_5", limite=0.0, comparacao=">=", **extra):
    return {"estudos": ["br_clinical_evidence"], "metrica": "auroc",
            "condicoes": [{"estatistica": estatistica, "comparacao": comparacao, "limite": limite}], **extra}


def _margens_declaradas():
    """Um exemplo SINTETICO de declaracao completa -- nao e proposta de margem."""
    return {"estado": "DECLARADO (teste sintetico)",
            "papel_dos_estudos": {"br_clinical_evidence": "exigido", "br_population_observed": "descritivo"},
            "melhoria_minima_no_coorte_br": _regra(delta="delta_br_full", limite=0.0),
            "regressao_maxima_no_controle": _regra(delta="delta_control", limite=-0.01),
            "paineis_com_regressao_inaceitavel": _regra(delta="delta_br_full", limite=-0.02, paineis=["missense"]),
            "beneficio_nao_explicado_por_um_painel": _regra(delta="delta_br_full", limite=0.0,
                                                            suporte_minimo_por_painel=10),
            "interacao": {"criterio_proprio": False, "motivo": "relatada com os absolutos (teste)"}}


def _bootstrap_declarado():
    return {"estado": "DECLARADO (teste sintetico)", "unidade_principal": "cluster_conjunto",
            "unidade_de_sensibilidade": "par", "replicas": 1000, "seed": 20260901, "percentis": [2.5, 97.5]}


def _resolvida(campanha):
    campanha = copy.deepcopy(campanha)
    campanha["g6"]["margens"] = _margens_declaradas()
    campanha["g6"]["bootstrap_da_interacao"] = _bootstrap_declarado()
    for pendencia in campanha["g6"]["pendencias_antes_do_congelamento"]:
        pendencia.update(estado="FEITO", onde="teste sintetico")
    return campanha


class BloqueiosTests(unittest.TestCase):
    def test_a_declaracao_de_hoje_nao_congela(self):
        bloqueios = g6.bloqueios(_campanha(), estado_do_codigo=CODIGO_OK)
        self.assertTrue(any(b.startswith("margens: nao declaradas") for b in bloqueios), bloqueios)
        self.assertTrue(any(b.startswith("bootstrap da interacao: nao declarado") for b in bloqueios))
        abertas = [x for x in _campanha()["g6"]["pendencias_antes_do_congelamento"]
                   if x["estado"] not in g6.ESTADOS_RESOLVIDOS]
        self.assertTrue(abertas)
        self.assertEqual(sum(b.startswith("pendencia ") for b in bloqueios), len(abertas))
        self.assertTrue(any(b.startswith("abraom_snapshot_hash") for b in bloqueios))

    def test_tudo_resolvido_nao_bloqueia(self):
        self.assertEqual(g6.bloqueios(_resolvida(_campanha()), estado_do_codigo=CODIGO_OK, proveniencia=ABRAOM_OK, entradas=ENTRADAS_OK), [])

    def test_so_o_texto_do_estado_nao_basta(self):
        # Reproduz a revisao de 24/09: margens e bootstrap com so {"estado": "DECLARADO"} passavam.
        campanha = _resolvida(_campanha())
        campanha["g6"]["margens"] = {"estado": "DECLARADO"}
        campanha["g6"]["bootstrap_da_interacao"] = {"estado": "DECLARADO"}
        bloqueios = g6.bloqueios(campanha, estado_do_codigo=CODIGO_OK, proveniencia=ABRAOM_OK, entradas=ENTRADAS_OK)
        for nome in g6.MARGENS_EXIGIDAS:
            self.assertIn(f"margens: {nome}: ausente", bloqueios)
        self.assertIn("margens: interacao.criterio_proprio: true ou false, explicito", bloqueios)
        self.assertTrue(any(b.startswith("bootstrap da interacao: unidade_principal") for b in bloqueios))
        self.assertTrue(any(b.startswith("bootstrap da interacao: replicas") for b in bloqueios))

    def test_margens_com_conteudo_invalido_reprovam(self):
        def c0(m, nome):
            return m[nome]["condicoes"][0]

        casos = {
            "limite": lambda m: c0(m, "melhoria_minima_no_coorte_br").update(limite=float("nan")),
            "limite: numero finito": lambda m: c0(m, "regressao_maxima_no_controle").update(limite=True),
            "tem de ser >= 0": lambda m: c0(m, "melhoria_minima_no_coorte_br").update(limite=-0.01),
            "tem de ser <= 0": lambda m: c0(m, "regressao_maxima_no_controle").update(limite=0.01),
            "estatistica": lambda m: c0(m, "melhoria_minima_no_coorte_br").update(estatistica="p97_5"),
            "comparacao": lambda m: c0(m, "melhoria_minima_no_coorte_br").update(comparacao="<="),
            "condicoes: lista nao vazia": lambda m: m["melhoria_minima_no_coorte_br"].update(condicoes=[]),
            "repetida": lambda m: m["melhoria_minima_no_coorte_br"]["condicoes"].append(
                {"estatistica": "p2_5", "comparacao": ">", "limite": 0.0}),
            "papel_dos_estudos": lambda m: m.pop("papel_dos_estudos"),
            "faltam": lambda m: m["papel_dos_estudos"].update(br_population_observed="exigido"),
            "estudo descritivo nao tem regra": lambda m: m["melhoria_minima_no_coorte_br"].update(
                estudos=["br_clinical_evidence", "br_population_observed"]),
            "ao menos um estudo exigido": lambda m: m["papel_dos_estudos"].update(br_clinical_evidence="descritivo"),
            "delta": lambda m: m["regressao_maxima_no_controle"].update(delta="delta_br_full"),
            "estudos": lambda m: m["melhoria_minima_no_coorte_br"].update(estudos=[]),
            "metrica": lambda m: m["melhoria_minima_no_coorte_br"].update(metrica="macro"),
            "paineis": lambda m: m["paineis_com_regressao_inaceitavel"].update(paineis=["plof"]),
            "exige `motivo`": lambda m: m["paineis_com_regressao_inaceitavel"].update(paineis=[]),
            "suporte_minimo": lambda m: m["beneficio_nao_explicado_por_um_painel"].update(suporte_minimo_por_painel=0),
            "criterio_proprio false exige": lambda m: m["interacao"].pop("motivo"),
            "interacao.condicoes": lambda m: m["interacao"].update(criterio_proprio=True),
        }
        for trecho, estragar in casos.items():
            margens = _margens_declaradas()
            estragar(margens)
            problemas = g6.problemas_das_margens(margens)
            self.assertTrue(any(trecho in p for p in problemas), (trecho, problemas))
        self.assertEqual(g6.problemas_das_margens(_margens_declaradas()), [])
        margens = _margens_declaradas()
        margens["paineis_com_regressao_inaceitavel"] = {"paineis": [], "motivo": "nenhum painel critico (teste)"}
        self.assertEqual(g6.problemas_das_margens(margens), [])

    def test_as_duas_formas_de_regra_da_revisao_sao_codificaveis(self):
        # (a) estimativa >= 0,02 e IC excluindo zero; (b) limite inferior >= 0,02: regras diferentes (revisao 25/09).
        forma_a = [{"estatistica": "estimativa", "comparacao": ">=", "limite": 0.02},
                   {"estatistica": "p2_5", "comparacao": ">", "limite": 0.0}]
        forma_b = [{"estatistica": "p2_5", "comparacao": ">=", "limite": 0.02}]
        for forma in (forma_a, forma_b):
            margens = _margens_declaradas()
            margens["melhoria_minima_no_coorte_br"]["condicoes"] = forma
            self.assertEqual(g6.problemas_das_margens(margens), [])

    def test_bootstrap_com_unidade_invalida_reprova(self):
        for campo, valor in (("unidade_principal", "componente_conexo"), ("unidade_de_sensibilidade", "cluster_conjunto"),
                             ("replicas", 100), ("seed", None), ("percentis", [5, 95])):
            bootstrap = dict(_bootstrap_declarado(), **{campo: valor})
            self.assertTrue(any(p.startswith(campo) for p in g6.problemas_do_bootstrap(bootstrap)), campo)
        self.assertEqual(g6.problemas_do_bootstrap(_bootstrap_declarado()), [])

    def test_pendencia_resolvida_exige_onde_ou_motivo(self):
        self.assertEqual(g6.problemas_das_pendencias([{"item": "x", "estado": "FEITO", "onde": "commit abc"},
                                                      {"item": "y", "estado": "RETIRADO", "motivo": "sem suporte"}]), [])
        problemas = g6.problemas_das_pendencias([{"item": "x", "estado": "FEITO"},
                                                 {"item": "y", "estado": "RETIRADO", "motivo": "  "},
                                                 {"item": "z", "estado": "A_FAZER"}])
        self.assertEqual(problemas, ["pendencia FEITO sem `onde`: x", "pendencia RETIRADO sem `motivo`: y",
                                     "pendencia A_FAZER: z"])

    def test_git_que_falha_nao_vale_como_sem_mudancas(self):
        base = _resolvida(_campanha())
        falhou = dict(CODIGO_OK, revisao=None, erro="CalledProcessError: fatal: not a git repository")
        self.assertEqual(g6.bloqueios(base, estado_do_codigo=falhou, proveniencia=ABRAOM_OK, entradas=ENTRADAS_OK),
                         ["estado do codigo nao conferido (git falhou: CalledProcessError: fatal: not a git "
                          "repository)"])
        for revisao in ("desconhecida", "", None, "0123"):
            self.assertTrue(g6.bloqueios(base, estado_do_codigo=dict(CODIGO_OK, revisao=revisao),
                                         proveniencia=ABRAOM_OK, entradas=ENTRADAS_OK))
        sujo = dict(CODIGO_OK, ausentes=["scripts/avaliar_estudos.py"], nao_rastreados=["eval/campanha/g6.py"],
                    modificados=["scripts/construir_g6.py"])
        self.assertEqual(g6.bloqueios(base, estado_do_codigo=sujo, proveniencia=ABRAOM_OK, entradas=ENTRADAS_OK),
                         ["codigo ausente: scripts/avaliar_estudos.py", "codigo fora do git: eval/campanha/g6.py",
                          "codigo com mudanca nao commitada: scripts/construir_g6.py"])

    def test_entrada_das_analises_secundarias_sem_registro_bloqueia(self):
        base = _resolvida(_campanha())
        bloqueios = g6.bloqueios(base, estado_do_codigo=CODIGO_OK, proveniencia=ABRAOM_OK,
                                 entradas={"regra_ampla": {"sha256": "a" * 64}})
        self.assertEqual(len(bloqueios), 1)
        self.assertTrue(bloqueios[0].startswith("entrada da analise secundaria sem registro (--entrada exposicao="))

    def test_margem_aberta_com_outra_palavra_continua_bloqueando(self):
        campanha = _resolvida(_campanha())
        campanha["g6"]["margens"]["estado"] = "em discussao"
        self.assertTrue(g6.bloqueios(campanha, estado_do_codigo=CODIGO_OK, proveniencia=ABRAOM_OK, entradas=ENTRADAS_OK))

    def test_a_declaracao_real_tem_os_campos_das_margens_e_do_bootstrap(self):
        # O modelo da declaracao tem todos os campos que a validacao cobra, nulos ate a decisao.
        g6_real = _campanha()["g6"]
        for nome in g6.MARGENS_EXIGIDAS:
            self.assertIn(nome, g6_real["margens"])
        self.assertIn("criterio_proprio", g6_real["margens"]["interacao"])
        self.assertEqual(set(g6_real["margens"]["papel_dos_estudos"]), set(g6.ESTUDOS))
        for nome in g6.MARGENS_EXIGIDAS:
            self.assertIn("condicoes", g6_real["margens"][nome])
        # A recomendacao operacional esta registrada, mas nao congela sem a confirmacao.
        self.assertEqual(g6_real["bootstrap_da_interacao"]["unidade_principal"], "cluster_conjunto")
        self.assertTrue(g6.problemas_do_bootstrap(g6_real["bootstrap_da_interacao"]))
        for campo in ("unidade_principal", "unidade_de_sensibilidade", "replicas", "seed", "percentis"):
            self.assertIn(campo, g6_real["bootstrap_da_interacao"])


class ManifestoTests(unittest.TestCase):
    def test_campos_do_mosaic_e_proveniencia_em_partes(self):
        campanha = _campanha()
        m = g6.montar_manifesto(campanha, **_entradas(campanha, ["margens"]))
        self.assertEqual(m["estado"], g6.RASCUNHO)
        campos = m["campos_do_mosaic"]
        for campo in ("base_checkpoint_id", "regionalized_checkpoint_id", "base_training_dataset_id",
                      "base_training_dataset_hash", "base_training_cutoff", "abraom_snapshot_hash",
                      "regionalization_method"):
            self.assertIn(campo, campos)
        self.assertEqual(campos["base_training_cutoff"]["pre_treino_do_r03"], "desconhecido")
        self.assertIn("2026-06", campos["base_training_cutoff"]["rotulos_da_cabeca"])
        self.assertIn("g2_final_janela2048", campos["base_training_dataset_id"]["treino_da_cabeca"])
        self.assertFalse(campos["study_membership_used_for_training"])
        self.assertEqual([a["semente"] for a in campos["regionalized_checkpoint_id"]["adapters"]],
                         [20260921, 20260922, 20260923])
        self.assertIn("rsLoRA rank 8, alpha 16", campos["regionalization_method"])
        self.assertIn("99 modulos", campos["regionalization_method"])
        self.assertEqual(set(m["sistemas"]), {"base", "regionalized"})
        self.assertEqual(m["sistemas"]["regionalized"]["limiar"]["threshold"], 0.6)

    def test_congelado_grava_le_e_recusa_adulteracao(self):
        campanha = _campanha()
        with self.assertRaises(g6.ManifestoInvalido):
            g6.montar_manifesto(campanha, **_entradas(campanha, ["margens"]), estado=g6.CONGELADO)
        m = g6.montar_manifesto(campanha, **_entradas(campanha), estado=g6.CONGELADO)
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            sha = g6.gravar_manifesto(pasta, m)
            self.assertEqual(sha, hashlib.sha256((pasta / g6.NOME_DO_MANIFESTO).read_bytes()).hexdigest())
            self.assertEqual((pasta / f"{g6.NOME_DO_MANIFESTO}.sha256").read_text(encoding="utf-8"),
                             f"{sha}  {g6.NOME_DO_MANIFESTO}\n")
            lido, sha_lido = g6.ler_manifesto_congelado(pasta)
            self.assertEqual((lido, sha_lido), (json.loads(json.dumps(m)), sha))
            with self.assertRaises(g6.ManifestoInvalido):
                g6.gravar_manifesto(pasta, m)   # nunca sobrescreve
            dados = (pasta / g6.NOME_DO_MANIFESTO).read_bytes()
            (pasta / g6.NOME_DO_MANIFESTO).write_bytes(dados.replace(b'"threshold": 0.55', b'"threshold": 0.45'))
            with self.assertRaises(g6.ManifestoInvalido):
                g6.ler_manifesto_congelado(pasta)

    def test_rascunho_nao_se_le_como_congelado(self):
        campanha = _campanha()
        rascunho = g6.montar_manifesto(campanha, **_entradas(campanha, ["margens"]))
        with self.assertRaises(g6.ManifestoInvalido):
            g6.gravar_manifesto(Path("."), rascunho)
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            dados = g6.serializar(rascunho)   # mesmo com o sha256 certo ao lado
            (pasta / g6.NOME_DO_MANIFESTO).write_bytes(dados)
            (pasta / f"{g6.NOME_DO_MANIFESTO}.sha256").write_text(hashlib.sha256(dados).hexdigest(), encoding="utf-8")
            with self.assertRaises(g6.ManifestoInvalido):
                g6.ler_manifesto_congelado(pasta)

    def test_bytes_nao_dependem_da_ordem_das_chaves(self):
        self.assertEqual(g6.serializar({"b": 1, "a": [1, 2]}), g6.serializar({"a": [1, 2], "b": 1}))
        with self.assertRaises(ValueError):
            g6.serializar({"a": float("nan")})


if __name__ == "__main__":
    unittest.main()
