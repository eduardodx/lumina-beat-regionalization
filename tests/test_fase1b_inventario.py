"""Inventario da Fase 1b com um avaliador substituto (sem o pacote mosaic) e caches sinteticos."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import fase1b_inventario as inv  # noqa: E402


def _entrega(pasta: Path) -> None:
    celulas = pasta / inv.CELULAS
    for sub in ("scoring", "views/4kb"):
        (celulas / sub).mkdir(parents=True)
    for nome in ("cells.parquet", "scoring/examples.parquet", "scoring/annotations.parquet", "views/4kb/examples.parquet"):
        pd.DataFrame({"x": [1]}).to_parquet(celulas / nome)
    (pasta / "scripts").mkdir()
    (pasta / inv.AVALIADOR).write_text("# substituto\n", encoding="utf-8")
    (pasta / inv.RELEASE).mkdir(parents=True)
    pd.DataFrame({"variant_id": ["r1", "r2"]}).to_parquet(pasta / inv.RELEASE / "clinical-variants.parquet")
    (pasta / "outputs" / "candidate-manifest").mkdir(parents=True)
    pd.DataFrame({"variant_id": ["r1", "x1", "x2", "x3", "x9"], "study": "regional",
                  "run": pd.array([0, 1, 2, 4, pd.NA], dtype="Int64"), "role": "test"}).to_parquet(pasta / inv.PEDIDOS)


def _linhas():
    """main: r1, x1 (primaria) e x2 (comparavel); sensibilidade: x3; x4 sem trained_run nao entra."""
    anotadas = pd.DataFrame({"variant_id": ["r1", "x1", "x2", "x3", "x4"], "trained_run": [0, 1, 2, 3, None],
                             "primary_panel": ["missense", "noncoding", "noncoding", "splice", "noncoding"]})
    linhas = anotadas[anotadas["trained_run"].notna()].assign(run=lambda d: d["trained_run"].astype(int))
    frames = {
        "main": pd.DataFrame({"variant_id": ["r1", "x1", "x2"], "cell": ["primary", "primary", "comparable"],
                              "weight": [1.0, 2.5, 1.0], "primary_panel": ["missense", "noncoding", "noncoding"]}),
        "primary_3x": pd.DataFrame({"variant_id": ["x3", "x2"], "cell": ["primary", "comparable"],
                                    "weight": [1.0, 1.0], "primary_panel": ["splice", "noncoding"]}),
    }
    return frames, linhas, anotadas


class InventarioTests(unittest.TestCase):
    def test_conta_desconta_caches_e_confere_pedidos(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            _entrega(pasta)
            cache = pasta / "cache"
            cache.mkdir()
            np.savez(cache / "fragmento_00000.npz", variant_id=np.array(["r1", "outra"]))
            with patch.object(inv, "carregar_avaliador", return_value=object()), \
                    patch.object(inv, "ler_cortes", return_value=[0.0, 1.0, 2.0]), \
                    patch.object(inv, "linhas_do_avaliador", return_value=_linhas()):
                codigo = inv.main(["--entrega", str(pasta), "--cache", str(cache), "--out-dir", str(pasta / "out")])
            self.assertEqual(codigo, 0)
            r = json.loads((pasta / "out" / "inventario_1b.json").read_text(encoding="utf-8"))
            self.assertEqual((r["linhas"]["anotadas"], r["linhas"]["sem_trained_run"]), (5, 1))
            self.assertEqual(r["linhas"]["por_definicao"]["main"]["primary"]["variantes"], 2)
            self.assertEqual(r["linhas"]["por_definicao"]["main"]["primary"]["peso_total"], 3.5)
            self.assertEqual(r["alvos"]["main"], {"variantes": 3, "no_release": 1, "nos_caches": 1, "complemento": 2,
                                                   "horas_de_gpu_estimadas": round(2 * inv.SEGUNDOS_POR_VARIANTE / 3600, 1)})
            self.assertEqual((r["alvos"]["todas"]["variantes"], r["alvos"]["todas"]["complemento"]), (4, 3))
            self.assertEqual((pasta / "out" / "complemento_main.txt").read_text(encoding="utf-8").split(), ["x1", "x2"])
            pedidos = r["pedidos"]
            self.assertEqual((pedidos["pedidos_sem_run"], pedidos["alvo_fora_dos_pedidos"],
                              pedidos["pedidos_fora_do_alvo"]), (1, 0, 1))   # x9 pedido fora das celulas
            self.assertEqual(pedidos["execucao_diferente_do_trained_run"], 1)  # x3: pedido run 4, trained_run 3
            again = inv.main(["--entrega", str(pasta), "--out-dir", str(pasta / "out")])
            self.assertEqual(again, 2, "nunca grava por cima")


if __name__ == "__main__":
    unittest.main()
