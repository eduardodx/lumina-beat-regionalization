"""Regra de congelamento dos adapters, sem torch. Inclui os tres casos que a revisao de 24/09 reproduziu: melhor pior
que a base com `supera_a_base: true`, janela e dropout do LoRA fora da receita, e outro recorte de 800 janelas
coerente so consigo mesmo -- os tres eram aprovados."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha.congelamento import (  # noqa: E402
    comparar_estados,
    conferir_corrida,
    diferencas_da_receita,
    melhor_do_historico,
    receita_da_configuracao,
)
from eval.campanha.recortes import carregar_campanha  # noqa: E402

CAMPANHA = carregar_campanha(Path(__file__).resolve().parents[1] / "configs" / "campanha_r03_desenvolvimento.json")
REFERENCIA = CAMPANHA["adapter_do_mr"]["referencia"]
RECORTE = {"sha256": "ab" * 32, "janelas": 800}
PLANOS = {"treino": REFERENCIA["plano_treino_sha256"], "validacao": REFERENCIA["plano_validacao_sha256"]}


def _config(**mudancas):
    """Argumentos gravados no checkpoint, como texto (o runner grava `str(v)`)."""
    config = {"seed": "20260922", "seed_da_validacao": "20260922", "lr": "5e-06", "passos": "3000",
              "exemplos_por_passo": "8", "batch": "1", "aquecimento": "2", "weight_decay": "0.01",
              "clip_norma": "1.0", "limite_validacao": "800", "validar_a_cada": "250", "backbone_em_eval": "True",
              "window_bp": "4096", "limite_treino": "None", "retomar": "None", "model_version": "r03",
              "modulos_esperados": "configs/adapter_r03_superficie.json", "lora_rank": "8", "lora_alpha": "16.0",
              "lora_dropout": "0.0", "sem_rslora": "False", "peso_focal": "1.0", "peso_contexto": "0.5",
              "peso_referencia": "0.5", "plano_treino": "/home/x/plano_treino.parquet"}
    config.update(mudancas)
    return config


def _relatorio():
    valores = {249: 1.7131, 999: 1.6424, 1999: 1.6076, 2999: 1.6044}
    historico = [{"passo": p, "validacao": {"criterio_primario": {"valor": v}}} if p in valores else {"passo": p}
                 for p in range(0, 3000, 250)] + [{"passo": p, "validacao": {"criterio_primario": {"valor": v}}}
                                                 for p, v in valores.items()]
    historico.sort(key=lambda linha: linha["passo"])
    return {
        "receita": {"lr": 5e-06, "passos": 3000, "exemplos_por_passo": 8, "batch": 1, "aquecimento": 2,
                    "weight_decay": 0.01, "clip_norma": 1.0, "seed": 20260922, "seed_da_validacao": 20260922,
                    "limite_validacao": 800, "validar_a_cada": 250, "window_bp": 4096, "backbone_em_eval": True,
                    "lora": {"rank": 8, "alpha": 16.0, "dropout": 0.0, "rslora": True},
                    "pesos": {"focal_alt": 1.0, "contexto_da_variante": 0.5, "referencia": 0.5}},
        "motivo_de_parada": "passos concluidos", "backbone_congelado_intacto": True,
        "atualizacoes_do_otimizador": 3000, "retomada": None, "historico": historico,
        "entradas": {"plano_treino_sha256": PLANOS["treino"], "plano_validacao_sha256": PLANOS["validacao"],
                     "checkpoint_sha256": REFERENCIA["checkpoint_sha256"], "exemplos_de_treino": 44645,
                     "exemplos_de_validacao": 800, "falhas_treino": {}, "falhas_validacao": {},
                     "recorte_da_validacao": dict(RECORTE)},
        "melhor_por_validacao": {"passo": 2999, "valor": 1.6044, "supera_a_base": True, "sha256": "cd" * 32},
        "linha_de_base": {"criterio_primario": {"valor": 1.7331}},
    }


def _conferir(relatorio, config=None, **mudancas):
    argumentos = {"semente": 20260922, "sha256_do_arquivo": "cd" * 32, "recorte": RECORTE,
                  "recorte_esperado": RECORTE, "sha256_dos_planos": dict(PLANOS),
                  "receita_do_checkpoint": receita_da_configuracao(config or _config()), **mudancas}
    return conferir_corrida(relatorio, CAMPANHA, **argumentos)


class CongelamentoTests(unittest.TestCase):
    def test_corrida_conforme_gera_a_entrada(self):
        entrada, problemas = _conferir(_relatorio())
        self.assertEqual(problemas, [])
        self.assertEqual(entrada["focal_alt_validacao"], {"base": 1.7331, "adapter": 1.6044, "delta": -0.1287})
        self.assertTrue(entrada["supera_a_base"])
        self.assertEqual(entrada["recorte_da_validacao_sha256"], RECORTE["sha256"])

    # -- os tres casos da revisao

    def test_melhor_pior_que_a_base_nao_passa_mesmo_com_a_flag(self):
        relatorio = _relatorio()
        relatorio["melhor_por_validacao"].update(valor=2.0)
        relatorio["historico"][-1]["validacao"]["criterio_primario"]["valor"] = 2.0
        for linha in relatorio["historico"]:
            if "validacao" in linha:
                linha["validacao"]["criterio_primario"]["valor"] = 2.0 + linha["passo"] * 1e-6
        relatorio["melhor_por_validacao"].update(passo=249, valor=2.0 + 249e-6)
        entrada, problemas = _conferir(relatorio)
        self.assertIsNone(entrada)
        self.assertTrue(any("nao e menor que a base" in p for p in problemas), problemas)
        self.assertTrue(any("supera_a_base=True, recalculado False" in p for p in problemas), problemas)

    def test_janela_e_dropout_do_lora_fora_da_receita(self):
        relatorio = _relatorio()
        relatorio["receita"]["window_bp"] = 2048
        relatorio["receita"]["lora"]["dropout"] = 0.5
        entrada, problemas = _conferir(relatorio, _config(window_bp="2048", lora_dropout="0.5"))
        self.assertIsNone(entrada)
        self.assertTrue(any("window_bp" in p for p in problemas))
        self.assertTrue(any("lora.dropout" in p for p in problemas))

    def test_outro_recorte_coerente_consigo_mesmo_nao_passa(self):
        outro = {"sha256": "99" * 32, "janelas": 800}
        relatorio = _relatorio()
        relatorio["entradas"]["recorte_da_validacao"] = dict(outro)
        entrada, problemas = _conferir(relatorio, recorte=outro)
        self.assertIsNone(entrada)
        self.assertTrue(any("re-sorteado do plano declarado" in p for p in problemas), problemas)

    # -- demais desvios

    def test_relatorio_e_checkpoint_que_discordam(self):
        relatorio = _relatorio()
        entrada, problemas = _conferir(relatorio, _config(lr="1e-05"))
        self.assertIsNone(entrada)
        self.assertTrue(any("no checkpoint" in p and "lr" in p for p in problemas), problemas)

    def test_cada_desvio_recusa(self):
        casos = {
            "orcamento": lambda r: r.update(atualizacoes_do_otimizador=2500),
            "parada": lambda r: r.update(motivo_de_parada="loss nao finita no passo 12"),
            "backbone": lambda r: r.update(backbone_congelado_intacto=False),
            "retomada": lambda r: r.update(retomada={"passo": 1000}),
            "sem registro de retomada": lambda r: r.pop("retomada"),
            "plano no relatorio": lambda r: r["entradas"].update(plano_validacao_sha256="0" * 64),
            "checkpoint": lambda r: r["entradas"].update(checkpoint_sha256="f2983560" + "0" * 56),
            "falha de janela": lambda r: r["entradas"].update(falhas_treino={"ref_mismatch": 1}),
            "treino parcial": lambda r: r["entradas"].update(exemplos_de_treino=200),
            "melhor fora do historico": lambda r: r["melhor_por_validacao"].update(passo=1999),
            "sem validacao no historico": lambda r: r.update(historico=[{"passo": 0}]),
            "base nao finita": lambda r: r["linha_de_base"]["criterio_primario"].update(valor=float("nan")),
        }
        for nome, alterar in casos.items():
            relatorio = copy.deepcopy(_relatorio())
            alterar(relatorio)
            entrada, problemas = _conferir(relatorio)
            self.assertIsNone(entrada, nome)
            self.assertTrue(problemas, nome)

    def test_arquivo_do_plano_mudado_sha_e_semente(self):
        self.assertTrue(_conferir(_relatorio(), sha256_dos_planos={**PLANOS, "treino": "0" * 64})[1])
        self.assertTrue(_conferir(_relatorio(), sha256_do_arquivo="ee" * 32)[1])
        self.assertTrue(_conferir(_relatorio(), semente=20260923)[1], "a corrida usou outra semente")
        self.assertTrue(_conferir(_relatorio(), semente=7)[1], "semente fora das declaradas")
        self.assertTrue(_conferir(_relatorio(), config=_config(retomar="/x/adapter.pt"))[1], "retomada no checkpoint")

    def test_a1_completa_a_receita_pelo_checkpoint(self):
        # A a_1 e anterior ao registro completo da receita no relatorio e a flag da semente da validacao: os
        # argumentos gravados completam, e a semente da validacao sai de seed + 1.
        relatorio = _relatorio()
        for chave in ("seed_da_validacao", "limite_validacao", "validar_a_cada", "backbone_em_eval", "lora",
                      "window_bp"):
            relatorio["receita"].pop(chave)
        relatorio["receita"]["seed"] = 20260921
        relatorio["entradas"].pop("recorte_da_validacao")
        config = _config(seed="20260921")
        config.pop("seed_da_validacao")
        receita = receita_da_configuracao(config)
        self.assertEqual(receita["seed_da_validacao"], 20260922)
        self.assertEqual(receita["lora"], {"rank": 8, "alpha": 16.0, "dropout": 0.0, "rslora": True})
        entrada, problemas = _conferir(relatorio, config, semente=20260921)
        self.assertEqual(problemas, [])
        sem_checkpoint = conferir_corrida(relatorio, CAMPANHA, semente=20260921, sha256_do_arquivo="cd" * 32,
                                          recorte=RECORTE, recorte_esperado=RECORTE, sha256_dos_planos=PLANOS)
        self.assertTrue(sem_checkpoint[1], "sem o checkpoint, falta receita")

    def test_chave_declarada_ausente_nas_duas_fontes(self):
        declarada = CAMPANHA["adapter_do_mr"]["receita"]
        problemas = diferencas_da_receita(declarada, {"relatorio": {"lr": 5e-06}})
        self.assertIn("receita: validar_a_cada ausente no relatorio e no checkpoint", problemas)

    def test_melhor_do_historico_fica_com_o_primeiro_no_empate(self):
        relatorio = {"historico": [{"passo": 249, "validacao": {"criterio_primario": {"valor": 1.5}}},
                                   {"passo": 499, "validacao": {"criterio_primario": {"valor": 1.5}}}]}
        self.assertEqual(melhor_do_historico(relatorio), (249, 1.5))


class EstadosTests(unittest.TestCase):
    def test_estado_vazio_ou_ausente_nunca_e_igual(self):
        self.assertFalse(comparar_estados({}, [], {}, [], np.array_equal)["iguais"])

    def test_chaves_fora_das_declaradas_nao_sao_iguais(self):
        a = {"x": np.zeros(2)}
        self.assertFalse(comparar_estados(a, ["x", "y"], dict(a), ["x", "y"], np.array_equal)["iguais"])

    def test_iguais_e_diferentes(self):
        a = {"x": np.zeros(2), "y": np.ones(3)}
        self.assertTrue(comparar_estados(a, ["x", "y"], {k: v.copy() for k, v in a.items()}, ["x", "y"],
                                         np.array_equal)["iguais"])
        b = {"x": np.zeros(2), "y": np.array([1.0, 1.0, 2.0])}
        r = comparar_estados(a, ["x", "y"], b, ["x", "y"], np.array_equal)
        self.assertFalse(r["iguais"])
        self.assertEqual(r["diferentes"], ["y"])


if __name__ == "__main__":
    unittest.main()
