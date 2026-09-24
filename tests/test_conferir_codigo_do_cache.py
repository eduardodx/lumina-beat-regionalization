"""A conferencia previa de codigo e ambiente, sem torch nem lumina (so a comparacao das identidades)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.conferir_codigo_do_cache import diferencas  # noqa: E402


class DiferencasTests(unittest.TestCase):
    def setUp(self):
        self.codigo = {"arquivos": {"eval/adapter/treino.py": "1a", "eval/campanha/cache.py": "2b"},
                       "pacote_lumina": {"pasta": "/home/x/lumina", "sha256": "9c"}}

    def test_identidades_iguais_nao_tem_diferenca(self):
        self.assertEqual(diferencas(self.codigo, dict(self.codigo)), [])

    def test_arquivo_mudado_aparece_com_o_nome(self):
        atual = {**self.codigo, "arquivos": {**self.codigo["arquivos"], "eval/adapter/treino.py": "ff"}}
        self.assertEqual(diferencas(self.codigo, atual, "codigo."),
                         ["codigo.arquivos.eval/adapter/treino.py: '1a' -> 'ff'"])

    def test_arquivo_novo_ou_faltando_tambem_e_diferenca(self):
        atual = {**self.codigo, "arquivos": {"eval/adapter/treino.py": "1a"}}
        self.assertEqual(len(diferencas(self.codigo, atual)), 1)

    def test_ambiente_diferente_aparece(self):
        self.assertEqual(diferencas({"torch": "2.5.1", "gpu": "NVIDIA A10G"}, {"torch": "2.6.0", "gpu": "NVIDIA A10G"}),
                         ["torch: '2.5.1' -> '2.6.0'"])


if __name__ == "__main__":
    unittest.main()
