"""Platt estavel, sigmoide sem estouro e a conferencia dos snapshots entre G5 e comparador. Sem torch."""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha import cabeca, metricas  # noqa: E402
from scripts.comparar_m0_mr_desenvolvimento import snapshots_diferentes  # noqa: E402


def _otimo_em_grade(s, y):
    """Referencia independente: minimiza a mesma perda por busca densa em a (b livre na grade grossa)."""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    alvo = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    melhor = None
    for a in np.linspace(0.0, 1.0, 2001):
        for b in np.linspace(-2.0, 2.0, 81):
            z = a * s + b
            valor = float(np.sum(alvo * np.logaddexp(0, -z) + (1 - alvo) * np.logaddexp(0, z)))
            if melhor is None or valor < melhor[0]:
                melhor = (valor, a, b)
    return melhor


class PlattTests(unittest.TestCase):
    def setUp(self):
        self._avisos = warnings.catch_warnings()
        self._avisos.__enter__()
        warnings.simplefilter("error")  # estouro em exp vira falha do teste

    def tearDown(self):
        self._avisos.__exit__(None, None, None)

    def test_caso_da_revisao_converge_para_o_otimo(self):
        # Antes: a = 6,8e9, probabilidades 0/1 e aviso de overflow.
        s, y = np.array([-10.0, -8, 8, 10]), np.array([0, 0, 1, 1])
        a, b = cabeca.platt(s, y)
        _, a_ref, b_ref = _otimo_em_grade(s, y)
        self.assertAlmostEqual(a, a_ref, places=3)
        self.assertAlmostEqual(b, b_ref, places=1)
        p = cabeca.calibrar(s, a, b)
        self.assertTrue(0.2 < p.min() < p.max() < 0.8, p)

    def test_logits_enormes_dao_a_mesma_solucao_reescalada(self):
        pequeno = cabeca.platt(np.array([-10.0, -8, 8, 10]), np.array([0, 0, 1, 1]))
        grande = cabeca.platt(np.array([-1000.0, -800, 800, 1000]), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(grande[0] * 100, pequeno[0], places=4)

    def test_logits_constantes_dao_a_taxa_da_validacao(self):
        y = np.array([0] * 30 + [1] * 20)
        a, b = cabeca.platt(np.full(50, 3.0), y)
        p = cabeca.calibrar(np.full(50, 3.0), a, b)
        # Com logits constantes o otimo e a media dos alvos suavizados de Platt: 20 alvos 21/22 e 30 alvos 1/32.
        self.assertAlmostEqual(float(p[0]), (20 * 21 / 22 + 30 / 32) / 50, places=4)

    def test_desbalanceado_e_finito_e_monotono(self):
        rng = np.random.default_rng(0)
        s = np.r_[rng.normal(-1, 1, 999), [2.5]]
        y = np.r_[np.zeros(999, dtype=int), [1]]
        a, b = cabeca.platt(s, y)
        self.assertGreater(a, 0.0)
        p = cabeca.calibrar(s, a, b)
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(metricas.auroc(s, y), metricas.auroc(p, y))

    def test_uma_classe_so_e_recusada(self):
        with self.assertRaises(ValueError):
            cabeca.platt(np.array([1.0, 2.0]), np.array([1, 1]))

    def test_sigmoide_sem_estouro_nos_extremos(self):
        p = cabeca.calibrar(np.array([-1e6, -50.0, 0.0, 50.0, 1e6]), 1.0, 0.0)
        self.assertEqual(p[0], 0.0)
        self.assertEqual(p[2], 0.5)
        self.assertEqual(p[-1], 1.0)


class SnapshotTests(unittest.TestCase):
    def test_mesmos_arquivos_passam(self):
        hashes = {"nenhum": "a", "janela2048": "b", "janela4096": "c"}
        self.assertEqual(snapshots_diferentes({"snapshots_sha256": dict(hashes)}, hashes), [])

    def test_snapshot_trocado_e_apontado(self):
        registrados = {"nenhum": "a", "janela2048": "b", "janela4096": "c"}
        self.assertEqual(snapshots_diferentes({"snapshots_sha256": registrados},
                                              {"nenhum": "a", "janela2048": "b", "janela4096": "X"}), ["janela4096"])

    def test_decisao_sem_hashes_e_recusada(self):
        self.assertTrue(snapshots_diferentes({}, {"nenhum": "a"}))


if __name__ == "__main__":
    unittest.main()
