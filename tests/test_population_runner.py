"""Regressoes do runner que nao precisam de GPU nem do checkpoint R03."""
import builtins
import random
import symtable
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_population_adapter as runner


class RunnerTests(unittest.TestCase):
    def test_smoke_nao_referencia_variaveis_do_laco(self):
        # aaa9d40 copiava o bloco de delta para o smoke: NameError apos toda a execucao GPU.
        path = Path(runner.__file__)
        table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        smoke = next(c for c in table.get_children() if c.get_name() == "rodar_smoke")
        available = set(vars(runner)) | set(vars(builtins))
        undefined = {s.get_name() for s in smoke.get_symbols()
                     if s.is_referenced() and s.is_global() and s.get_name() not in available}
        self.assertEqual(undefined, set())

    def test_seed_controla_torch_e_random(self):
        manual_seed = Mock()
        with patch.dict(sys.modules, {"torch": SimpleNamespace(manual_seed=manual_seed)}):
            runner.inicializar_aleatoriedade(77)
            first = random.random()
            runner.inicializar_aleatoriedade(77)
            self.assertEqual(first, random.random())
        self.assertEqual(manual_seed.call_count, 2)
        manual_seed.assert_called_with(77)

    def test_limite_nao_pega_so_o_inicio_abraom(self):
        plan = pd.DataFrame({"fonte": ["abraom"] * 400 + ["global"] * 600,
                             "variant_id": range(1000)})
        sample = runner.amostrar_preservando_a_mistura(plan, quantos=160, seed=7)
        self.assertEqual(sample.fonte.value_counts().to_dict(), {"global": 96, "abraom": 64})
        self.assertEqual(sample.variant_id.nunique(), 160)
        pd.testing.assert_frame_equal(sample, runner.amostrar_preservando_a_mistura(
            plan, quantos=160, seed=7))


    def test_selecao_guarda_a_melhor_e_nao_a_ultima(self):
        # piloto 5: a validacao tocou o fundo no passo 89 e degradou ate ficar pior que nao treinar.
        melhor = None
        for passo, valor in ((29, 1.7204), (59, 1.7180), (89, 1.7079), (119, 1.7435), (299, 1.9991)):
            melhor, _ = runner.atualizar_melhor(melhor, passo, {"criterio_primario": {"valor": valor}})
        self.assertEqual(melhor["passo"], 89)
        self.assertAlmostEqual(melhor["valor"], 1.7079)

    def test_selecao_aceita_a_avaliacao_final_fora_da_cadencia(self):
        melhor = None
        for passo, valor in ((29, 1.75), (59, 1.74)):
            melhor, _ = runner.atualizar_melhor(melhor, passo, {"criterio_primario": {"valor": valor}})
        melhor, trocou = runner.atualizar_melhor(melhor, 61, {"criterio_primario": {"valor": 1.70}})
        self.assertTrue(trocou)
        self.assertEqual(melhor["passo"], 61)

    def test_selecao_nao_troca_em_empate(self):
        melhor, _ = runner.atualizar_melhor(None, 10, {"criterio_primario": {"valor": 1.5}})
        melhor, trocou = runner.atualizar_melhor(melhor, 20, {"criterio_primario": {"valor": 1.5}})
        self.assertFalse(trocou)
        self.assertEqual(melhor["passo"], 10)


    @staticmethod
    def _detalhe(ganhos, locos):
        antes, depois = [], []
        for indice, (fonte, ganho) in enumerate(ganhos):
            base = 1.0 + 0.001 * indice
            comum = {"variant_id": f"v{indice}", "locus_id": locos[indice], "fonte": fonte,
                     "focal_index": 100 + indice}
            antes.append({**comum, "focal_ce": base, "termo_massa": base / 2, "termo_escolha": base / 2})
            depois.append({**comum, "focal_ce": base + ganho, "termo_massa": base / 2,
                           "termo_escolha": base / 2 + ganho})
        return antes, depois

    def test_bootstrap_reamostra_locos_e_cobre_o_observado(self):
        # O ganho varia POR LOCO (i // 3), nao dentro dele: senao todo loco teria a mesma media e a
        # reamostragem nao produziria variacao -- o IC sairia degenerado por culpa da fixture.
        ganhos = [("abraom", -0.05 - 0.01 * (i // 3)) for i in range(12)]
        ganhos += [("global", -0.01 - 0.01 * (i // 3)) for i in range(12)]
        locos = [f"L{i // 3}" for i in range(24)]
        saida = runner.bootstrap_do_delta(*self._detalhe(ganhos, locos), replicas=400, seed=3)
        self.assertEqual(saida["locos"], 8)
        self.assertEqual(saida["unidade_de_reamostragem"], "loco")
        for fonte in ("abraom", "global"):
            faixa = saida["por_fonte"][fonte]["focal_ce"]
            self.assertLessEqual(faixa["p2_5"], faixa["delta"])
            self.assertGreaterEqual(faixa["p97_5"], faixa["delta"])
            self.assertLess(faixa["p2_5"], faixa["p97_5"], "com variacao o IC tem de ter largura")
        diferenca = saida["abraom_menos_global"]["focal_ce"]
        self.assertLess(diferenca["diferenca"], 0, "o ABraOM melhorou mais neste sintetico")

    def test_bootstrap_recusa_com_um_unico_loco(self):
        ganhos = [("abraom", -0.05), ("global", -0.01)]
        saida = runner.bootstrap_do_delta(*self._detalhe(ganhos, ["L0", "L0"]), replicas=10)
        self.assertIn("indisponivel", saida)

    def test_bootstrap_recusa_conjuntos_diferentes(self):
        antes, depois = self._detalhe([("abraom", -0.05), ("global", -0.01)], ["L0", "L1"])
        depois[0]["variant_id"] = "outro"
        self.assertIn("indisponivel", runner.bootstrap_do_delta(antes, depois, replicas=10))

    # ------------------------------------------------------------------ pareamento (revisao de 22/09)

    @staticmethod
    def _registro(fonte, variant_id, focal, loco, ce, massa, escolha):
        return {"fonte": fonte, "variant_id": variant_id, "focal_index": focal, "locus_id": loco,
                "focal_ce": ce, "termo_massa": massa, "termo_escolha": escolha}

    def test_bootstrap_com_antes_e_depois_identicos_da_delta_zero_mesmo_com_alelo_nas_duas_fontes(self):
        # O contraexemplo da revisao: `variant_id` e chrom:pos:ref:alt, SEM a fonte. Parear so por ele fazia o
        # registro global sobrescrever o do ABraOM, e o delta do ABraOM saia +0,25 com entradas identicas.
        registros = [
            self._registro("abraom", "chr1:100:A:G", 1500, "L0", 1.0, 0.5, 0.5),
            self._registro("global", "chr1:100:A:G", 2100, "L0", 1.5, 0.7, 0.8),
            self._registro("abraom", "chr2:200:C:T", 900, "L1", 2.0, 1.0, 1.0),
            self._registro("global", "chr3:300:G:A", 3000, "L2", 1.2, 0.6, 0.6),
        ]
        saida = runner.bootstrap_do_delta([dict(r) for r in registros], [dict(r) for r in registros],
                                          replicas=50, seed=1)
        self.assertNotIn("indisponivel", saida)
        for fonte, campos in saida["por_fonte"].items():
            for campo, faixa in campos.items():
                self.assertEqual(faixa["delta"], 0.0, (fonte, campo))
                self.assertEqual((faixa["p2_5"], faixa["p97_5"]), (0.0, 0.0), (fonte, campo))

    def test_bootstrap_pareia_janela_a_janela_mesmo_embaralhado(self):
        antes = [self._registro("abraom", "chr1:100:A:G", 1500, "L0", 1.0, 0.5, 0.5),
                 self._registro("global", "chr1:100:A:G", 2100, "L0", 1.5, 0.7, 0.8),
                 self._registro("global", "chr3:300:G:A", 3000, "L1", 1.2, 0.6, 0.6)]
        depois = [dict(r, focal_ce=r["focal_ce"] - (0.3 if r["fonte"] == "abraom" else 0.1)) for r in antes]
        saida = runner.bootstrap_do_delta(antes, list(reversed(depois)), replicas=50, seed=1)
        self.assertAlmostEqual(saida["por_fonte"]["abraom"]["focal_ce"]["delta"], -0.3)
        self.assertAlmostEqual(saida["por_fonte"]["global"]["focal_ce"]["delta"], -0.1)

    def test_bootstrap_recusa_janela_repetida(self):
        antes = [self._registro("abraom", "chr1:100:A:G", 1500, "L0", 1.0, 0.5, 0.5),
                 self._registro("abraom", "chr1:100:A:G", 1500, "L1", 1.1, 0.5, 0.6)]
        saida = runner.bootstrap_do_delta(antes, [dict(r) for r in antes], replicas=10)
        self.assertIn("indisponivel", saida)
        self.assertTrue(any("repetida" in problema for problema in saida["problemas"]), saida)

    def test_bootstrap_recusa_loco_ausente_em_vez_de_cair_para_o_variant_id(self):
        antes = [self._registro("abraom", "chr1:100:A:G", 1500, None, 1.0, 0.5, 0.5),
                 self._registro("global", "chr3:300:G:A", 3000, "L1", 1.2, 0.6, 0.6)]
        saida = runner.bootstrap_do_delta(antes, [dict(r) for r in antes], replicas=10)
        self.assertIn("indisponivel", saida)
        self.assertTrue(any("locus_id" in problema for problema in saida["problemas"]), saida)

    def test_bootstrap_recusa_janela_que_muda_de_loco(self):
        antes = [self._registro("abraom", "chr1:100:A:G", 1500, "L0", 1.0, 0.5, 0.5),
                 self._registro("global", "chr3:300:G:A", 3000, "L1", 1.2, 0.6, 0.6)]
        depois = [dict(antes[0], locus_id="L9"), dict(antes[1])]
        self.assertIn("indisponivel", runner.bootstrap_do_delta(antes, depois, replicas=10))

    def test_bootstrap_informa_locos_por_fonte(self):
        antes = [self._registro("abraom", "chr1:100:A:G", 1500, "L0", 1.0, 0.5, 0.5),
                 self._registro("global", "chr1:100:A:G", 2100, "L0", 1.5, 0.7, 0.8),
                 self._registro("global", "chr1:900:C:T", 2500, "L0", 1.4, 0.7, 0.7),
                 self._registro("abraom", "chr2:200:C:T", 900, "L1", 2.0, 1.0, 1.0),
                 self._registro("global", "chr3:300:G:A", 3000, "L2", 1.2, 0.6, 0.6)]
        saida = runner.bootstrap_do_delta(antes, [dict(r) for r in antes], replicas=10)
        self.assertEqual(saida["janelas_por_fonte"], {"abraom": 2, "global": 3})
        self.assertEqual(saida["locos_por_fonte"], {"abraom": 2, "global": 2})
        self.assertEqual(saida["locos_com_mais_de_uma_fonte"], 1)
        self.assertEqual(saida["locos"], 3)

    def test_bootstrap_inclui_a_ordem_quando_os_dois_lados_a_trazem(self):
        antes = [dict(self._registro("abraom", f"chr1:{i}:A:G", 100 + i, f"L{i}", 1.0, 0.5, 0.5),
                      alt_em_primeiro=0.0, posto_do_alt=2.0) for i in range(4)]
        depois = [dict(r, alt_em_primeiro=1.0 if i % 2 else 0.0, posto_do_alt=1.0 if i % 2 else 2.0)
                  for i, r in enumerate(antes)]
        saida = runner.bootstrap_do_delta(antes, depois, replicas=20, seed=2)
        self.assertIn("alt_em_primeiro", saida["campos"])
        self.assertAlmostEqual(saida["por_fonte"]["abraom"]["alt_em_primeiro"]["delta"], 0.5)
        self.assertAlmostEqual(saida["por_fonte"]["abraom"]["posto_do_alt"]["delta"], -0.5)

    # ------------------------------------------------------------------ detalhe do final e do melhor

    def test_detalhe_do_final_sobrevive_quando_o_ultimo_passo_cai_na_cadencia(self):
        # 3.000 passos validando a cada 250: o ultimo passo (2999) valida DENTRO do laco. Antes o detalhe dele
        # era descartado ali, e o bootstrap do final saia "indisponivel".
        melhor, detalhes, historico = None, {}, []
        valores = {249: 1.70, 499: 1.69, 749: 1.68, 999: 1.67, 1249: 1.66, 1499: 1.655, 1749: 1.65,
                   1999: 1.649, 2249: 1.66, 2499: 1.67, 2749: 1.675, 2999: 1.68}
        for passo in range(3000):
            if (passo + 1) % 250 == 0:
                validacao = {"criterio_primario": {"valor": valores[passo]},
                             "detalhe": [{"passo": passo}]}
                melhor, _ = runner.registrar_validacao(passo, validacao, melhor, detalhes)
                historico.append({"passo": passo, "validacao": validacao})
        self.assertEqual(detalhes["final"]["passo"], 2999)
        self.assertEqual(detalhes["final"]["detalhe"], [{"passo": 2999}])
        self.assertEqual(detalhes["melhor"]["passo"], 1999)
        self.assertEqual(melhor["passo"], 1999)
        self.assertTrue(all("detalhe" not in linha["validacao"] for linha in historico),
                        "o detalhe nao pode inchar o historico")

    # ------------------------------------------------------------------ semente da validacao (revisao de 23/09)

    @staticmethod
    def _config(**mudancas):
        valores = {"plano_validacao": Path("plano_validacao.parquet"), "limite_validacao": 800,
                   "seed": 20260922, "seed_da_validacao": 20260922, "recorte_da_validacao_igual_a": None,
                   "window_bp": 4096}
        valores.update(mudancas)
        return SimpleNamespace(**valores)

    @staticmethod
    def _janelas(chaves):
        return [SimpleNamespace(fonte=f, variant_id=v, focal_index=i, locus_id=f"L{n}")
                for n, (f, v, i) in enumerate(chaves)]

    def test_subamostra_da_validacao_exige_semente_propria(self):
        self.assertIsNotNone(runner.problema_da_validacao(self._config(seed_da_validacao=None)))
        self.assertIsNone(runner.problema_da_validacao(self._config()))
        # Sem subamostra (validacao inteira), a semente nao entra.
        self.assertIsNone(runner.problema_da_validacao(self._config(limite_validacao=None, seed_da_validacao=None)))
        self.assertIsNotNone(runner.problema_da_validacao(
            self._config(plano_validacao=None, recorte_da_validacao_igual_a=Path("x"))))

    def test_validacao_usa_a_semente_propria_e_nao_a_do_adapter(self):
        # Antes: seed + 1. A a_2 (seed 20260922) seria validada com 20260923, outro recorte.
        chamadas = []

        def carregar(caminho, fetch, *, window_bp, limite, seed=0):
            chamadas.append(seed)
            return self._janelas([("abraom", "chr1:1:A:G", 10)]), {}

        with patch.object(runner, "carregar_exemplos", carregar):
            for seed_do_adapter in (20260921, 20260922, 20260923):
                runner.carregar_validacao(self._config(seed=seed_do_adapter), fetch=None)
        self.assertEqual(chamadas, [20260922, 20260922, 20260922])

    def test_recorte_ignora_ordem_e_distingue_fonte_e_janela(self):
        chaves = [("abraom", "chr1:100:A:G", 1500), ("global", "chr1:100:A:G", 2100), ("global", "chr2:5:C:T", 7)]
        a = runner.hash_do_recorte(chaves)
        self.assertEqual(a, runner.hash_do_recorte(list(reversed(chaves))))
        self.assertEqual(a["janelas"], 3)
        self.assertNotEqual(a, runner.hash_do_recorte([("global", *chaves[0][1:])] + chaves[1:]))
        self.assertNotEqual(a, runner.hash_do_recorte([(chaves[0][0], chaves[0][1], 1501)] + chaves[1:]))
        self.assertEqual(a, runner.recorte_dos_exemplos(self._janelas(chaves)))

    def test_recorte_de_referencia_confere_ou_aborta_antes_do_modelo(self):
        import json
        import tempfile

        chaves = [("abraom", "chr1:100:A:G", 1500), ("global", "chr3:300:G:A", 3000)]
        with tempfile.TemporaryDirectory() as pasta:
            detalhe = {"chave": "fonte|variant_id|focal_index", "linha_de_base": [
                {"fonte": f, "variant_id": v, "focal_index": i, "locus_id": "L0", "focal_ce": 1.0}
                for f, v, i in reversed(chaves)], "melhor": None, "final": None}
            Path(pasta, "detalhe_da_validacao.json").write_text(json.dumps(detalhe), encoding="utf-8")
            self.assertEqual(runner.recorte_de_referencia(Path(pasta))["sha256"],
                             runner.hash_do_recorte(chaves)["sha256"])

            def mesmas(caminho, fetch, *, window_bp, limite, seed=0):
                return self._janelas(chaves), {}

            def outras(caminho, fetch, *, window_bp, limite, seed=0):
                return self._janelas(chaves[:1] + [("global", "chr9:9:T:C", 5)]), {}

            with patch.object(runner, "carregar_exemplos", mesmas):
                _, _, recorte, problema = runner.carregar_validacao(
                    self._config(recorte_da_validacao_igual_a=Path(pasta)), fetch=None)
            self.assertIsNone(problema)
            self.assertIn("confere_com", recorte)
            with patch.object(runner, "carregar_exemplos", outras):
                _, _, _, problema = runner.carregar_validacao(
                    self._config(recorte_da_validacao_igual_a=Path(pasta)), fetch=None)
            self.assertIn("MESMA validacao", problema)

    def test_rodar_treino_confere_a_validacao_antes_de_montar_o_modelo(self):
        # A ordem importa: descobrir a validacao errada depois de carregar o R03 e so desperdicio; depois do
        # treino, seria perder a corrida.
        fonte = Path(runner.__file__).read_text(encoding="utf-8")
        corpo = fonte[fonte.index("def rodar_treino"):]
        self.assertLess(corpo.index("problema_da_validacao(config)"), corpo.index("carregar_exemplos("))
        self.assertLess(corpo.index("carregar_validacao(config, fetch)"), corpo.index("montar(config, device)"))
        self.assertNotIn("config.seed + 1", fonte)

    def test_rodar_treino_seleciona_sempre_por_registrar_validacao(self):
        # A regressao era de FLUXO: um `pop` do detalhe fora de lugar. A selecao no treino passa toda por
        # `registrar_validacao`, que guarda o detalhe do final e do melhor.
        path = Path(runner.__file__)
        table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        treino_fn = next(c for c in table.get_children() if c.get_name() == "rodar_treino")
        referenciados = {s.get_name() for s in treino_fn.get_symbols() if s.is_referenced()}
        self.assertIn("registrar_validacao", referenciados)
        self.assertNotIn("atualizar_melhor", referenciados)


if __name__ == "__main__":
    unittest.main()
