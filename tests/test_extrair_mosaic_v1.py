"""Partes sem GPU da extracao do Mosaic v1: tabela do release, selecao (complemento e amostra) e comparacao."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import extrair_mosaic_v1 as ext  # noqa: E402


def _release(raiz: Path, n: int = 12) -> None:
    ids = [f"v{i:02d}" for i in range(n)]
    (raiz / "views" / "4kb").mkdir(parents=True)
    pd.DataFrame({"variant_id": ids, "chrom": ["chr1"] * n, "pos_1based": [100 + i for i in range(n)],
                  "ref": ["A"] * n, "alt": ["G"] * n, "binary_label": np.array([i % 2 for i in range(n)], dtype="int8"),
                  "label_tier": ["gold"] * n}).to_parquet(raiz / "clinical-variants.parquet", index=False)
    pd.DataFrame({"variant_id": ids, "primary_panel": ["missense"] * n}).to_parquet(
        raiz / "evaluation-panels.parquet", index=False)
    pd.DataFrame({"variant_id": ids, "sequence_eligible": [i != 11 for i in range(n)],
                  "overlap_cluster_id": [f"c{i}" for i in range(n)]}).to_parquet(raiz / ext.VISTA, index=False)


class SelecaoTests(unittest.TestCase):
    def test_tabela_tem_as_colunas_da_extracao_e_so_elegiveis(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            _release(raiz)
            tabela = ext.tabela_do_release(raiz)
            self.assertEqual(list(tabela.columns), list(ext.COLUNAS))
            self.assertEqual(len(tabela), 11)

    def test_complemento_e_amostra(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            _release(raiz)
            tabela = ext.tabela_do_release(raiz)
            caches = {"a": {"v00", "v01", "v02", "v03"}, "b": {"v04", "v11"}}   # v11 nao e elegivel
            comp = ext.selecionar(tabela, "complemento", caches, 4)
            self.assertEqual(sorted(comp["variant_id"]), [f"v{i:02d}" for i in range(5, 11)])
            self.assertTrue((comp["papel"] == "mosaic_v1_complemento").all())
            amostra = ext.selecionar(tabela, "conferencia", caches, 4)
            self.assertTrue(set(amostra["variant_id"]) <= {"v00", "v01", "v02", "v03", "v04"})
            self.assertIn("v04", set(amostra["variant_id"]), "cada cache contribui para a amostra")
            self.assertEqual(len(amostra), 3)   # 2 do cache a + o unico elegivel do cache b
            again = ext.selecionar(tabela, "conferencia", caches, 4)
            self.assertEqual(list(amostra["variant_id"]), list(again["variant_id"]), "amostra deterministica")


class ComparacaoTests(unittest.TestCase):
    def test_tolerancia_e_faltantes(self):
        a = np.zeros(3, dtype=np.float32)
        novos = {"x": {"v1": a, "v2": a + 2e-6}}
        antigos = {"x": {"v1": a, "v2": a}}
        self.assertTrue(ext.comparar(novos, antigos, 1e-5)["passou"])
        self.assertFalse(ext.comparar({"x": {"v1": a + 1e-3}}, antigos, 1e-5)["passou"])
        r = ext.comparar({"x": {"v1": a, "v9": a}}, antigos, 1e-5)
        self.assertFalse(r["passou"], "variante da amostra sem vetor antigo reprova")
        self.assertEqual(r["extracoes"]["x"]["faltando_no_antigo"], 1)

    def test_vetores_lidos_dos_fragmentos(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            np.savez(pasta / "fragmento_00000.npz", variant_id=np.array(["v1", "v2"]), papel=np.array(["p", "p"]),
                     leitura_antiga_1344=np.arange(4, dtype=np.float32).reshape(2, 2))
            lidos = ext.vetores(pasta, {"v2"}, ["leitura_antiga_1344"])
            np.testing.assert_array_equal(lidos["leitura_antiga_1344"]["v2"], np.array([2, 3], dtype=np.float32))
            self.assertEqual(ext.ids_do_cache(pasta), {"v1", "v2"})


if __name__ == "__main__":
    unittest.main()
