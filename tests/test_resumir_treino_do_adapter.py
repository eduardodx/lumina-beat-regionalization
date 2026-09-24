"""O resumidor tem de funcionar no relatorio novo e nos antigos (antes do bootstrap e da ordem do ALT)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import resumir_treino_do_adapter as resumo


def _validacao(focal, abraom, glob, ref, ordem=None):
    diagnostico = {}
    if ordem is not None:
        diagnostico = {"abraom": {"alt_em_primeiro_entre_nao_ref": ordem[0]},
                       "global": {"alt_em_primeiro_entre_nao_ref": ordem[1]}}
    return {"criterio_primario": {"categoria": "focal_alt", "valor": focal, "posicoes": 800},
            "por_categoria": {"focal_alt": {"media": focal, "posicoes": 800},
                              "referencia": {"media": ref, "posicoes": 5000}},
            "por_fonte": {"abraom": {"focal_alt": {"media": abraom, "posicoes": 320}},
                          "global": {"focal_alt": {"media": glob, "posicoes": 480}}},
            "diagnostico_do_focal": diagnostico}


def _relatorio(com_bootstrap=True):
    relatorio = {
        "receita": {"lr": 5e-6, "passos": 3000, "exemplos_por_passo": 8, "batch": 1, "seed": 20260921,
                    "pesos": {"focal_alt": 1.0}},
        "entradas": {"exemplos_de_treino": 44645, "exemplos_de_validacao": 800,
                     "mistura_do_treino": {"fracao": {"abraom": 0.4, "global": 0.6}},
                     "plano_treino_sha256": "c76d08d4" * 8},
        "custo": {"segundos_no_laco": 5400.0, "segundos_por_atualizacao": 1.8, "pico_de_memoria_mb": 3922.6},
        "atualizacoes_do_otimizador": 3000, "motivo_de_parada": "passos concluidos",
        "backbone_congelado_intacto": True,
        "linha_de_base": _validacao(1.74, 1.80, 1.70, 1.06, (0.45, 0.41)),
        "historico": [{"passo": 0}, {"passo": 249, "validacao": _validacao(1.72, 1.77, 1.69, 1.061, (0.45, 0.41))},
                      {"passo": 2999, "validacao": _validacao(1.68, 1.72, 1.66, 1.063, (0.46, 0.41))}],
        "melhor_por_validacao": {"passo": 2999, "valor": 1.68, "delta_contra_a_base": -0.06,
                                 "recomendacao": "usar este adapter", "sha256": "ab" * 32},
        "delta_da_validacao": {"por_categoria": {"focal_alt": -0.06, "referencia": 0.003},
                               "por_fonte": {"abraom": {"focal_alt": -0.08}},
                               "diagnostico_do_focal": {"abraom": {"termo_escolha": -0.01}}},
        "saidas": {"adapter": "/x/adapter.pt", "adapter_sha256": "cd" * 32},
    }
    if com_bootstrap:
        relatorio["delta_da_validacao"]["bootstrap_por_loco"] = {
            "natureza": "exploratoria", "unidade_de_reamostragem": "loco", "replicas": 2000, "locos": 600,
            "janelas": 800, "janelas_por_fonte": {"abraom": 320, "global": 480},
            "locos_por_fonte": {"abraom": 310, "global": 295}, "locos_com_mais_de_uma_fonte": 5,
            "por_fonte": {"abraom": {"focal_ce": {"delta": -0.08, "p2_5": -0.11, "p97_5": -0.05}}},
            "abraom_menos_global": {"focal_ce": {"diferenca": -0.02, "p2_5": -0.05, "p97_5": 0.01}}}
        relatorio["delta_da_validacao"]["bootstrap_do_melhor"] = "o melhor e o final"
    return relatorio


class ResumoTests(unittest.TestCase):
    def test_curva_tem_a_base_e_cada_validacao(self):
        texto = "\n".join(resumo.resumir(_relatorio()))
        self.assertIn("base  1.7400", texto)
        self.assertIn("249  1.7200", texto)
        self.assertIn("2999  1.6800", texto)
        self.assertIn("0.450/0.410", texto)

    def test_bootstrap_sai_rotulado_como_exploratorio(self):
        texto = "\n".join(resumo.resumir(_relatorio()))
        self.assertIn("EXPLORATORIO", texto)
        self.assertIn("focal_ce -0.0800 [-0.1100; -0.0500]", texto)
        self.assertIn("abraom-global focal_ce -0.0200 [-0.0500; +0.0100]", texto)
        self.assertIn("locos por fonte {'abraom': 310, 'global': 295}", texto)
        self.assertIn("bootstrap do melhor: o melhor e o final", texto)

    def test_relatorio_antigo_sem_bootstrap_nem_ordem(self):
        # A corrida g4_lr5e6 e anterior ao bootstrap e a ordem do ALT: o resumo tem de sair mesmo assim,
        # porque e por ele que se ve a curva que nunca foi colada.
        relatorio = _relatorio(com_bootstrap=False)
        relatorio["linha_de_base"]["diagnostico_do_focal"] = {}
        texto = "\n".join(resumo.resumir(relatorio))
        self.assertIn("bootstrap do final: ausente", texto)
        self.assertIn("base  1.7400", texto)

    def test_bootstrap_indisponivel_aparece_com_o_motivo(self):
        relatorio = _relatorio()
        relatorio["delta_da_validacao"]["bootstrap_por_loco"] = {"indisponivel": "detalhe nao pareavel",
                                                                 "problemas": ["antes[0] sem ['locus_id']"]}
        self.assertIn("INDISPONIVEL (detalhe nao pareavel)", "\n".join(resumo.resumir(relatorio)))

    def test_main_aceita_o_diretorio(self):
        with tempfile.TemporaryDirectory() as pasta:
            Path(pasta, "treino_do_adapter.json").write_text(json.dumps(_relatorio()), encoding="utf-8")
            self.assertEqual(resumo.main([pasta]), 0)
            self.assertEqual(resumo.main([str(Path(pasta, "nao_existe"))]), 2)

    def test_recorte_da_validacao_sai_no_resumo(self):
        # A a_1 e anterior ao registro: o recorte vem do detalhe da validacao, e a semente aparece como nao
        # registrada. Uma corrida nova traz os dois no relatorio.
        antigo = "\n".join(resumo.resumir(_relatorio(), {"sha256": "ab12" * 16, "janelas": 800}))
        self.assertIn("nao registrada (antes da flag era seed + 1)", antigo)
        self.assertIn("recorte ab12ab12ab12ab12 (800 janelas, lido do detalhe_da_validacao.json)", antigo)
        novo = _relatorio()
        novo["receita"]["seed_da_validacao"] = 20260922
        novo["entradas"]["recorte_da_validacao"] = {"sha256": "cd34" * 16, "janelas": 800,
                                                    "confere_com": {"sha256": "cd34" * 16}}
        texto = "\n".join(resumo.resumir(novo))
        self.assertIn("semente da subamostra 20260922", texto)
        self.assertIn("recorte cd34cd34cd34cd34 (800 janelas) | IDENTICO ao da referencia", texto)

    def test_main_calcula_o_recorte_do_detalhe_quando_o_relatorio_nao_o_tem(self):
        with tempfile.TemporaryDirectory() as pasta:
            Path(pasta, "treino_do_adapter.json").write_text(json.dumps(_relatorio()), encoding="utf-8")
            Path(pasta, "detalhe_da_validacao.json").write_text(json.dumps({"linha_de_base": [
                {"fonte": "abraom", "variant_id": "chr1:1:A:G", "focal_index": 10, "locus_id": "L0"}]}),
                encoding="utf-8")
            self.assertEqual(resumo.main([pasta]), 0)


if __name__ == "__main__":
    unittest.main()
