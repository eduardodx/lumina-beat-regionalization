"""Acompanhador da cadeia: so a leitura do log (sem processos)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.acompanhar_cadeia import resultado_final, ultimo_progresso  # noqa: E402


class AcompanharTests(unittest.TestCase):
    def test_ultimo_progresso_ignora_avisos(self):
        linhas = ["interpretador: /opt/conda/bin/python3", "exit_ambiente=0",
                  "  fragmento 00013: 57,344/122,568  (0.0740 s/variante, faltam ~1.3 h)",
                  "[WARNING] Field span duplicates an ancestor field"]
        self.assertTrue(ultimo_progresso(linhas).startswith("fragmento 00013"))
        self.assertEqual(ultimo_progresso(["[WARNING] x"]), "(sem progresso ainda)")

    def test_resultado_final(self):
        completa, relevantes = resultado_final(["exit_extracao_a2=0", "algo", "CADEIA COMPLETA 2026-09-24 20:00:00"])
        self.assertTrue(completa)
        self.assertEqual(relevantes, ["exit_extracao_a2=0", "CADEIA COMPLETA 2026-09-24 20:00:00"])
        completa, relevantes = resultado_final(["exit_extracao_a2=0", "exit_extracao_a3=1"])
        self.assertFalse(completa)
        self.assertEqual(relevantes[-1], "exit_extracao_a3=1")


if __name__ == "__main__":
    unittest.main()
