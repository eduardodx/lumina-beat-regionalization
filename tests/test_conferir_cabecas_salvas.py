"""Conferencia das cabecas salvas: as regras puras, sem torch. O caminho completo (salvar, recarregar, reproduzir)
esta em tests/test_campanha_cabeca.py, que precisa de torch."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha import metricas  # noqa: E402
from eval.campanha.cabeca import FORMATO_DA_CABECA, RECEITA  # noqa: E402
from scripts import conferir_cabecas_salvas as conferencia  # noqa: E402


def _cenario():
    platt = {"a": 0.8, "b": -1.2, "inverte_a_ordem": False}
    limiar = {"limiar": 0.41, "mcc": 0.7}
    relatorio = {"extracao": "leitura_antiga_1344", "politica": "janela2048", "semente_do_adapter": 20260921,
                 "cabecas": {"M0": [{"semente": 11, "platt": platt, "limiar_de_mcc": limiar, "epoca": 120}],
                             "MR": [{"semente": 11, "platt": platt, "limiar_de_mcc": limiar, "epoca": 90}]}}
    esperado = {"decisao_g5_sha256": "d" * 64, "snapshot_sha256": "s" * 64,
                "cache_identidade_sha256": {"M0": "0" * 64, "MR": "1" * 64}}
    carga = {"formato": FORMATO_DA_CABECA, "sistema": "MR", "semente": 11, "extracao": "leitura_antiga_1344",
             "politica": "janela2048", "decisao_g5_sha256": "d" * 64, "snapshot_sha256": "s" * 64,
             "cache_identidade_sha256": "1" * 64, "semente_do_adapter": 20260921, "receita": dict(RECEITA),
             "platt": dict(platt), "limiar_de_mcc": dict(limiar), "epoca": 90}
    return carga, relatorio, esperado


class ConferenciaTests(unittest.TestCase):
    def _problemas(self, carga, relatorio, esperado, sistema="MR"):
        return conferencia.conferir_metadados(carga, sistema=sistema, semente=11, relatorio=relatorio,
                                              esperado=esperado)

    def test_cabeca_coerente_nao_tem_problema(self):
        self.assertEqual(self._problemas(*_cenario()), [])

    def test_platt_que_inverte_a_ordem_reprova(self):
        carga, relatorio, esperado = _cenario()
        for a in (0.0, -0.3):
            carga["platt"] = {"a": a, "b": 0.1, "inverte_a_ordem": True}
            relatorio["cabecas"]["MR"][0]["platt"] = dict(carga["platt"])
            self.assertTrue(any("inverteria a ordem" in p for p in self._problemas(carga, relatorio, esperado)))

    def test_calibrador_diferente_do_relatorio_reprova(self):
        carga, relatorio, esperado = _cenario()
        carga["platt"] = {"a": 0.81, "b": -1.2, "inverte_a_ordem": False}
        self.assertTrue(any(p.startswith("platt diferente") for p in self._problemas(carga, relatorio, esperado)))

    def test_identidades_trocadas_reprovam(self):
        for campo, valor in (("snapshot_sha256", "x" * 64), ("decisao_g5_sha256", "y" * 64),
                             ("cache_identidade_sha256", "0" * 64), ("semente_do_adapter", 20260922),
                             ("politica", "nenhum"), ("formato", "outro")):
            carga, relatorio, esperado = _cenario()
            carga[campo] = valor
            self.assertTrue(any(p.startswith(campo) for p in self._problemas(carga, relatorio, esperado)), campo)

    def test_m0_nao_exige_semente_do_adapter(self):
        carga, relatorio, esperado = _cenario()
        carga.update(sistema="M0", cache_identidade_sha256="0" * 64, epoca=120)
        del carga["semente_do_adapter"]
        self.assertEqual(self._problemas(carga, relatorio, esperado, sistema="M0"), [])

    def test_receita_mudada_no_codigo_aparece(self):
        carga, relatorio, esperado = _cenario()
        carga["receita"] = {**RECEITA, "oculta": 128}
        self.assertIn("receita diferente da do codigo atual", self._problemas(carga, relatorio, esperado))

    def test_ensemble_confere_a_media_das_probabilidades(self):
        rng = np.random.default_rng(0)
        y = (rng.random(300) < 0.4).astype(int)
        paineis = np.array(["missense", "splice", "noncoding"])[np.arange(300) % 3]
        probs = {s: [np.clip(0.5 * y + rng.random(300) * 0.6, 0, 1) for _ in range(3)] for s in ("M0", "MR")}
        relatorio = {"media_das_probabilidades": {
            s: metricas.resumo(np.mean(v, axis=0), y, paineis) for s, v in probs.items()}}
        _, problemas = conferencia.conferir_ensemble(probs, y, paineis, relatorio)
        self.assertEqual(problemas, [])
        probs["MR"][0] = probs["MR"][0][::-1].copy()
        _, problemas = conferencia.conferir_ensemble(probs, y, paineis, relatorio)
        self.assertTrue(problemas and all("ensemble MR" in p for p in problemas))


if __name__ == "__main__":
    unittest.main()
