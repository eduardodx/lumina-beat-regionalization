"""Regra de congelamento dos adapters, sem torch: receita, orcamento, planos, recorte e o criterio primario."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha.congelamento import conferir_corrida, diferencas_da_receita, receita_da_configuracao  # noqa: E402
from eval.campanha.recortes import carregar_campanha  # noqa: E402

CAMPANHA = carregar_campanha(Path(__file__).resolve().parents[1] / "configs" / "campanha_r03_desenvolvimento.json")
RECORTE = {"sha256": "ab" * 32, "janelas": 800}


def _relatorio():
    return {
        "receita": {"lr": 5e-06, "passos": 3000, "exemplos_por_passo": 8, "batch": 1, "aquecimento": 2,
                    "weight_decay": 0.01, "clip_norma": 1.0, "seed": 20260922, "seed_da_validacao": 20260922,
                    "limite_validacao": 800, "validar_a_cada": 250, "window_bp": 4096, "backbone_em_eval": True,
                    "lora": {"rank": 8, "alpha": 16.0, "dropout": 0.0, "rslora": True},
                    "pesos": {"focal_alt": 1.0, "contexto_da_variante": 0.5, "referencia": 0.5}},
        "motivo_de_parada": "passos concluidos", "backbone_congelado_intacto": True,
        "atualizacoes_do_otimizador": 3000,
        "entradas": {"plano_treino_sha256": "c76d08d4" + "0" * 56, "plano_validacao_sha256": "034eca34" + "0" * 56,
                     "checkpoint_sha256": "f2983560" + "0" * 56, "falhas_treino": {}, "falhas_validacao": {},
                     "recorte_da_validacao": dict(RECORTE)},
        "melhor_por_validacao": {"passo": 2999, "valor": 1.6044, "supera_a_base": True, "sha256": "cd" * 32},
        "linha_de_base": {"criterio_primario": {"valor": 1.7331}},
    }


def _conferir(relatorio, **mudancas):
    argumentos = {"semente": 20260922, "sha256_do_arquivo": "cd" * 32, "recorte": RECORTE,
                  "prefixo_do_checkpoint": "f2983560", **mudancas}
    return conferir_corrida(relatorio, CAMPANHA, **argumentos)


class CongelamentoTests(unittest.TestCase):
    def test_corrida_conforme_gera_a_entrada(self):
        entrada, problemas = _conferir(_relatorio())
        self.assertEqual(problemas, [])
        self.assertEqual(entrada["focal_alt_validacao"], {"base": 1.7331, "adapter": 1.6044, "delta": -0.1287})
        self.assertEqual(entrada["recorte_da_validacao_sha256"], RECORTE["sha256"])
        self.assertEqual(entrada["sha256"], "cd" * 32)

    def test_cada_desvio_da_regra_recusa(self):
        casos = {
            "receita": lambda r: r["receita"].update(lr=1e-5),
            "orcamento": lambda r: r.update(atualizacoes_do_otimizador=2500),
            "parada": lambda r: r.update(motivo_de_parada="loss nao finita no passo 12"),
            "backbone": lambda r: r.update(backbone_congelado_intacto=False),
            "plano": lambda r: r["entradas"].update(plano_validacao_sha256="ffff" + "0" * 60),
            "checkpoint": lambda r: r["entradas"].update(checkpoint_sha256="0" * 64),
            "falha de janela": lambda r: r["entradas"].update(falhas_treino={"ref_mismatch": 1}),
            "nao supera a base": lambda r: r["melhor_por_validacao"].update(supera_a_base=False),
            "semente da validacao": lambda r: r["receita"].update(seed_da_validacao=20260923),
            "lora": lambda r: r["receita"]["lora"].update(rslora=False),
            "pesos": lambda r: r["receita"]["pesos"].update(referencia=1.0),
        }
        for nome, alterar in casos.items():
            relatorio = copy.deepcopy(_relatorio())
            alterar(relatorio)
            entrada, problemas = _conferir(relatorio)
            self.assertIsNone(entrada, nome)
            self.assertTrue(problemas, nome)

    def test_sha_e_recorte_e_semente_conferidos(self):
        self.assertTrue(_conferir(_relatorio(), sha256_do_arquivo="ee" * 32)[1])
        self.assertTrue(_conferir(_relatorio(), recorte={"sha256": "ab" * 32, "janelas": 799})[1])
        self.assertTrue(_conferir(_relatorio(), recorte={"sha256": "99" * 32, "janelas": 800})[1])
        self.assertTrue(_conferir(_relatorio(), semente=20260923)[1], "a corrida usou outra semente")
        self.assertTrue(_conferir(_relatorio(), semente=7)[1], "semente fora das declaradas")

    def test_a1_completa_a_receita_pelo_checkpoint(self):
        # A a_1 e anterior ao registro completo da receita: o relatorio nao tem validacao nem lora, e a semente da
        # validacao era seed + 1. Os argumentos gravados no checkpoint completam, sem sobrescrever o relatorio.
        relatorio = _relatorio()
        for chave in ("seed_da_validacao", "limite_validacao", "validar_a_cada", "backbone_em_eval", "lora"):
            relatorio["receita"].pop(chave)
        relatorio["receita"]["seed"] = 20260921
        relatorio["entradas"].pop("recorte_da_validacao")
        self.assertTrue(_conferir(relatorio, semente=20260921)[1], "sem o checkpoint, falta receita")
        config = {"seed": "20260921", "limite_validacao": "800", "validar_a_cada": "250", "backbone_em_eval": "True",
                  "lora_rank": "8", "lora_alpha": "16.0", "sem_rslora": "False", "peso_focal": "1.0",
                  "peso_contexto": "0.5", "peso_referencia": "0.5", "lr": "5e-06", "passos": "3000",
                  "plano_treino": "/home/x/plano_treino.parquet"}
        receita = receita_da_configuracao(config)
        self.assertEqual(receita["seed_da_validacao"], 20260922)
        self.assertEqual(receita["lora"], {"rank": 8, "alpha": 16.0, "rslora": True})
        entrada, problemas = _conferir(relatorio, semente=20260921, receita_do_checkpoint=receita)
        self.assertEqual(problemas, [])

    def test_chave_declarada_ausente_e_diferenca(self):
        declarada = CAMPANHA["adapter_do_mr"]["receita"]
        self.assertIn("receita: validar_a_cada ausente na corrida",
                      diferencas_da_receita(declarada, {k: v for k, v in _relatorio()["receita"].items()
                                                        if k != "validar_a_cada"}))


if __name__ == "__main__":
    unittest.main()
