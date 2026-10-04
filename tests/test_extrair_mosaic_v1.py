"""Partes sem GPU da extracao do Mosaic v1: tabela do release, selecao (complemento e amostra) e comparacao."""
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

    def test_nao_finito_forma_e_leituras_ausentes_reprovam(self):
        antigo = {"x": {"v": np.zeros(3)}}
        for valor in (np.nan, np.inf, -np.inf):
            with self.subTest(valor=valor):
                self.assertFalse(ext.comparar({"x": {"v": np.array([valor, 0, 0])}}, antigo, 1e-5)["passou"])
                self.assertFalse(ext.comparar(antigo, {"x": {"v": np.array([valor, 0, 0])}}, 1e-5)["passou"])
        self.assertFalse(ext.comparar({"x": {"v": np.zeros(1)}}, antigo, 1e-5)["passou"], "nao aceitar broadcasting")
        self.assertFalse(ext.comparar({}, antigo, 1e-5)["passou"])
        self.assertFalse(ext.comparar(antigo, {**antigo, "y": antigo["x"]}, 1e-5)["passou"])
        self.assertFalse(ext.comparar(antigo, antigo, np.nan)["passou"])
        self.assertFalse(ext.comparar(antigo, antigo, -1)["passou"])

    def test_variante_esperada_sem_vetor_novo_reprova(self):
        antigo = {"x": {"v1": np.zeros(2), "v2": np.zeros(2)}}
        r = ext.comparar({"x": {"v1": np.zeros(2)}}, antigo, 1e-5)
        self.assertFalse(r["passou"])
        self.assertEqual(r["extracoes"]["x"]["faltando_no_novo"], 1)

    def test_vetores_lidos_dos_fragmentos(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            np.savez(pasta / "fragmento_00000.npz", variant_id=np.array(["v1", "v2"]), papel=np.array(["p", "p"]),
                     leitura_antiga_1344=np.arange(4, dtype=np.float32).reshape(2, 2))
            lidos = ext.vetores(pasta, {"v2"}, ["leitura_antiga_1344"])
            np.testing.assert_array_equal(lidos["leitura_antiga_1344"]["v2"], np.array([2, 3], dtype=np.float32))
            self.assertEqual(ext.ids_do_cache(pasta), {"v1", "v2"})


class CacheAntigoTests(unittest.TestCase):
    """Arquivos NPZ reais; tabela em memoria para testar a validacao sem GPU ou codec Parquet."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pasta = Path(self.temp.name)
        self.tabela = pd.DataFrame({"variant_id": ["v1", "v2"], "chrom": ["chr1"] * 2,
                                   "pos_1based": [100, 200], "ref": ["A"] * 2, "alt": ["G"] * 2,
                                   "binary_label": [1, 0], "primary_panel": ["missense"] * 2,
                                   "overlap_cluster_id": ["c1", "c2"], "label_tier": ["gold"] * 2,
                                   "papel": ["train"] * 2})
        self.ident = {"sistema": "M0", "adapter_sha256": None, "checkpoint_sha256": "base",
                      "codigo": {"arquivo": "sha"}, "ambiente": {"torch": "versao"},
                      "revisao_do_codigo": "anterior", "tabela_sha256_conteudo": ext.hash_do_conteudo(self.tabela),
                      "tabela_sha256_composicao": ext.hash_da_tabela(self.tabela), "papeis": ["train"]}
        self.referencia = dict(self.ident)
        self.manifesto = {"completo": True, "identidade": self.ident, "variantes_na_tabela": 2,
                          "variantes_no_cache": 2, "faltando": 0}
        self.gravar_metadados()
        (self.pasta / "tabela.parquet").write_bytes(b"tabela simulada; leitura mockada")
        self.gravar_fragmento()
        mock = patch.object(ext.pd, "read_parquet", return_value=self.tabela)
        mock.start()
        self.addCleanup(mock.stop)

    def gravar_metadados(self):
        (self.pasta / "identidade.json").write_text(json.dumps(self.ident), encoding="utf-8")
        (self.pasta / "manifesto.json").write_text(json.dumps(self.manifesto), encoding="utf-8")

    def gravar_fragmento(self, indice=0, valor=0.0, ids=("v1", "v2")):
        np.savez(self.pasta / f"fragmento_{indice:05d}.npz", variant_id=np.array(ids), papel=np.array(["train"] * len(ids)),
                 cabecas_172=np.full((len(ids), 172), valor, dtype=np.float32),
                 leitura_antiga_1344=np.full((len(ids), 1344), valor, dtype=np.float32))

    def test_cache_valido_e_metadados_do_release_podem_mudar(self):
        release = self.tabela.drop(columns="papel").copy()
        release["binary_label"] = [0, 1]
        release["overlap_cluster_id"] = ["novo1", "novo2"]
        self.ident["revisao_do_codigo"] = "outro_commit"
        self.gravar_metadados()
        ids, fonte = ext.validar_cache_antigo(self.pasta, self.referencia, release)
        self.assertEqual(ids, {"v1", "v2"})
        self.assertEqual(fonte["n"], 2)
        self.assertIn("fragmento_00000.npz", fonte["sha256"])

    def test_identidade_do_segundo_cache_divergente_reprova(self):
        self.ident["codigo"] = {"arquivo": "outro_sha"}
        self.gravar_metadados()
        with self.assertRaisesRegex(ValueError, "identidade difere.*codigo"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)

    def test_cli_recusa_segundo_cache_divergente_antes_de_importar_torch(self):
        outro = self.pasta / "outro_cache"
        outro.mkdir()
        for nome in ("identidade.json", "manifesto.json", "tabela.parquet", "fragmento_00000.npz"):
            (outro / nome).write_bytes((self.pasta / nome).read_bytes())
        ident = {**self.ident, "codigo": {"arquivo": "outro_sha"}}
        (outro / "identidade.json").write_text(json.dumps(ident), encoding="utf-8")
        # Se o runner tentar importar torch antes de recusar, o teste falha com ImportError.
        with patch.object(ext, "tabela_do_release", return_value=self.tabela.drop(columns="papel")), \
                patch.dict(sys.modules, {"torch": None}):
            codigo = ext.main(["--modo", "complemento", "--entrega", str(self.pasta),
                               "--referencia", str(self.pasta), "--cache-antigo", str(self.pasta),
                               "--cache-antigo", str(outro), "--checkpoint", "nao_usado.pt",
                               "--fasta", "nao_usado.fa", "--out-dir", str(self.pasta / "saida")])
        self.assertEqual(codigo, 2)
        self.assertFalse((self.pasta / "saida").exists())

    def test_incompleto_e_manifesto_divergente_reprovam(self):
        self.manifesto["completo"] = False
        self.gravar_metadados()
        with self.assertRaisesRegex(ValueError, "cache completo"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)
        self.manifesto["completo"] = True
        self.manifesto["identidade"] = {**self.ident, "checkpoint_sha256": "outro"}
        self.gravar_metadados()
        with self.assertRaisesRegex(ValueError, "identidade do manifesto"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)

    def test_fragmentos_corrompidos_nao_finitos_duplicados_e_incompletos_reprovam(self):
        self.gravar_fragmento(valor=np.nan)
        with self.assertRaisesRegex(ValueError, "nao finitos"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)
        (self.pasta / "fragmento_00000.npz").write_bytes(b"corrompido")
        with self.assertRaisesRegex(ValueError, "ilegivel"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)
        self.gravar_fragmento()
        self.gravar_fragmento(indice=1)
        with self.assertRaisesRegex(ValueError, "repetidas"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)
        (self.pasta / "fragmento_00001.npz").unlink()
        self.gravar_fragmento(ids=("v1",))
        with self.assertRaisesRegex(ValueError, "tabela inteira"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)

    def test_mesmo_id_com_outro_alelo_reprova(self):
        release = self.tabela.copy()
        release.loc[0, "alt"] = "T"
        with self.assertRaisesRegex(ValueError, "coordenadas/alelos"):
            ext.validar_cache_antigo(self.pasta, self.referencia, release)

    def test_tabela_adulterada_reprova(self):
        self.tabela.loc[0, "ref"] = "C"
        with self.assertRaisesRegex(ValueError, "hashes da tabela"):
            ext.validar_cache_antigo(self.pasta, self.referencia, self.tabela)


if __name__ == "__main__":
    unittest.main()
