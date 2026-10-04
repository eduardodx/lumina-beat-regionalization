"""Inventario do Mosaic v1 novo sobre um release sintetico minimo (sem torch; roda no Windows)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import inventario_mosaic_v1 as inv  # noqa: E402

CRITICAS = [
    {"gene": "HBB", "hgvs": "HbS", "grch38": {"chrom": "chr11", "pos": 1000, "ref": "T", "alt": "A"}},
    {"gene": "TP53", "hgvs": "R337H", "grch38": {"chrom": "chr17", "pos": 7670699, "ref": "C", "alt": "T"}},
]


def _release(pasta: Path) -> None:
    n = 20
    ids = [f"v{i:02d}" for i in range(n)]
    raiz = pasta / inv.RELEASE
    (raiz / "views" / "4kb").mkdir(parents=True)
    (raiz / "studies" / "brazilian-proxies").mkdir(parents=True)
    tier = ["gold" if i % 2 == 0 else "consensus" for i in range(n)]
    label = [i % 3 == 0 for i in range(n)]
    pd.DataFrame({"variant_id": ids, "chrom": ["chr11"] * n, "pos_1based": [1000 + i for i in range(n)],
                  "ref": ["T"] * n, "alt": ["A"] * n, "binary_label": np.array(label, dtype="int8"),
                  "label_tier": tier, "br_lab_any": [i % 4 == 1 for i in range(n)]}).to_parquet(
        raiz / "clinical-variants.parquet", index=False)
    pd.DataFrame({"variant_id": ids, "sequence_eligible": [i != 19 for i in range(n)],
                  "overlap_cluster_id": [f"c{i // 2}" for i in range(n)], "core_unit_id": [f"u{i}" for i in range(n)],
                  "core_fold": np.array([i % 5 for i in range(n)], dtype="int8"),
                  "core_purged_runs": [[0] if i == 7 else [] for i in range(n)],
                  "gene_transfer_group_id": [f"g{i // 4}" for i in range(n)]}).to_parquet(raiz / inv.VISTA, index=False)
    pd.DataFrame({"variant_id": ids, "primary_panel": ["missense", "splice", "noncoding", "plof"] * 5}).to_parquet(
        raiz / "evaluation-panels.parquet", index=False)
    pd.DataFrame({"variant_id": ids, "present_abraom": [i < 6 for i in range(n)],
                  "abraom_status": ["present" if i < 6 else "not_found" for i in range(n)],
                  "abraom_filter": ["PASS"] * n, "gnomad_status": ["present"] * n}).to_parquet(
        raiz / "variant-annotations.parquet", index=False)
    pd.DataFrame({"variant_id": ["v00", "v02", "v04", "v10"], "study_id": ["br_population_observed"] * 4,
                  "member_role": ["case", "control", "unmatched_case", "case"],
                  "matched_variant_id": ["v02", "v00", None, None], "primary_panel": ["missense"] * 4,
                  "binary_label": np.array([1, 1, 0, 0], dtype="int8")}).to_parquet(raiz / inv.MEMBERSHIP, index=False)
    pedidos = pasta / inv.PEDIDOS
    pedidos.mkdir(parents=True)
    pd.DataFrame({"variant_id": ["x1", "x2", "v01"], "study": ["regional", "regional", "vus"],
                  "run": pd.array([0, None, 1], dtype="Int64"), "role": ["test", "test", "validation"]}).to_parquet(
        pedidos / "requests.parquet", index=False)
    pd.DataFrame({"variant_id": ["x1", "x2", "v01"], "in_release": [False, False, True]}).to_parquet(
        pedidos / "variants.parquet", index=False)
    cache = pasta / "cache_M0"
    cache.mkdir()
    np.savez(cache / "fragmento_00000.npz", variant_id=np.array(["v00", "v01", "v02", "x1"]), papel=np.array(["a"] * 4))
    (cache / "identidade.json").write_text(json.dumps({"janela_bp": 4096, "indice_focal": 2047}), encoding="utf-8")


class InventarioTests(unittest.TestCase):
    def setUp(self):
        self.original = inv.ler_criticas
        inv.ler_criticas = lambda caminho: CRITICAS

    def tearDown(self):
        inv.ler_criticas = self.original

    def test_inventario_de_ponta_a_ponta(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            _release(pasta)
            args = ["--entrega", str(pasta), "--cache-antigo", str(pasta / "cache_M0"), "--out-dir", str(pasta / "out")]
            self.assertEqual(inv.main(args), 0)
            r = json.loads((pasta / "out" / "inventario.json").read_text(encoding="utf-8"))
            self.assertEqual(r["release"]["elegiveis_4kb"], 19)
            run0 = r["execucoes_4kb"][0]
            # run 0: teste = fold 0, validation = fold 1, treino = folds 2-4 sem a linha purgada (v07, fold 2).
            self.assertEqual(run0["teste_todos"], 4)
            self.assertEqual(run0["teste_gold"], 2)
            self.assertEqual(run0["treino"], 11)
            self.assertEqual(run0["purgadas"], 1)
            pop = r["proxies"]["br_population_observed"]
            self.assertEqual(pop["case"]["n"], 2)
            self.assertEqual(pop["sem_par_por_painel"]["missense"]["n"], 1)
            # A critica fora do release aparece no inventario: e a lacuna do avaliador de seguranca.
            fora = [c for c in r["criticas"] if not c["no_release"]]
            self.assertEqual([c["gene"] for c in fora], ["TP53"])
            dentro = [c for c in r["criticas"] if c["no_release"]][0]
            self.assertTrue(dentro["present_abraom"])
            self.assertEqual(r["pedidos"]["por_estudo"]["regional"]["run_nulo"], 1)
            cache = r["caches_antigos"][str(pasta / "cache_M0")]
            self.assertEqual(cache["cobre"]["release_elegivel_4kb"], 3)
            self.assertEqual(cache["cobre"]["pedidos:regional"], 1)
            self.assertEqual(r["complemento"]["release_elegivel_4kb"], 16)
            self.assertEqual(len((pasta / "out" / "complemento_release_4kb.txt").read_text().split()), 16)
            self.assertEqual(inv.main(args), 2, "pasta de saida existente e recusada")

    def test_beneficio_e_pbr_contam_so_elegiveis(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            _release(pasta)
            df = inv.carregar_release(pasta / inv.RELEASE)
            bloco = inv.beneficio_e_pbr(df)
            # presentes: v00-v05; gold entre eles: v00, v02, v04; P (i % 3 == 0): v00, v03.
            self.assertEqual(bloco["beneficio"]["n"], 3)
            self.assertEqual(bloco["beneficio"]["n_P"], 1)
            self.assertEqual(bloco["p_br"]["n"], 2)
            self.assertEqual(bloco["p_br"]["por_tier"], {"consensus": 1, "gold": 1})


if __name__ == "__main__":
    unittest.main()
