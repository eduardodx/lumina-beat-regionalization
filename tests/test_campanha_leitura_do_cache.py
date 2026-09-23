"""Leitura do cache para a cabeca: completude, identidade, alinhamento a tabela e pareamento M0 x MR. Sem torch."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha import cache as cache_io  # noqa: E402
from eval.campanha.leitura_do_cache import (  # noqa: E402
    CacheInvalido,
    carregar_cache,
    conferir_par,
    linhas_da_politica,
)
from eval.campanha.recortes import hash_do_conteudo  # noqa: E402


def _tabela():
    linhas = []
    for i, papel in enumerate(["train"] * 4 + ["validation"] * 2 + ["selecao"] * 2):
        linhas.append({"variant_id": f"v{i}", "chrom": "chr1", "pos_1based": 100 + i, "ref": "A", "alt": "G",
                       "binary_label": i % 2, "primary_panel": "missense", "overlap_cluster_id": f"c{i}",
                       "label_tier": "gold", "papel": papel})
    return pd.DataFrame(linhas)


def _montar(pasta: Path, *, sistema="M0", completo=True, faltar=None, identidade_extra=None):
    tabela = _tabela()
    identidade = {"sistema": sistema, "tabela_sha256_conteudo": hash_do_conteudo(tabela),
                  "codigo": {"arquivos": {"x.py": "1"}}, "adapter_sha256": None if sistema == "M0" else "abc",
                  "semente_do_adapter": None if sistema == "M0" else 20260921}
    identidade.update(identidade_extra or {})
    (pasta / "identidade.json").write_text(json.dumps(identidade), encoding="utf-8")
    (pasta / "manifesto.json").write_text(json.dumps({"completo": completo}), encoding="utf-8")
    tabela.to_parquet(pasta / "tabela.parquet", index=False)
    ids = [v for v in tabela["variant_id"] if v != faltar]
    # Fragmentos em ORDEM DIFERENTE da tabela: o leitor tem de realinhar. O valor de cada linha e o indice da
    # variante, para o teste ver qual linha foi parar onde.
    ordem = list(reversed(ids))
    for indice, pedaco in enumerate((ordem[:3], ordem[3:])):
        numeros = np.array([int(v[1:]) for v in pedaco], dtype=np.float32)
        cache_io.gravar_fragmento(
            pasta, indice, variant_id=np.array(pedaco),
            papel=tabela.set_index("variant_id").loc[pedaco, "papel"].to_numpy(),
            matrizes={"cabecas_172": np.repeat(numeros[:, None], 172, axis=1),
                      "leitura_antiga_1344": np.repeat(numeros[:, None], 1344, axis=1)})
    return tabela


class LeituraTests(unittest.TestCase):
    def test_matriz_alinhada_a_tabela_mesmo_com_fragmentos_fora_de_ordem(self):
        with tempfile.TemporaryDirectory() as pasta:
            tabela = _montar(Path(pasta))
            lido = carregar_cache(Path(pasta), "cabecas_172")
            self.assertEqual(lido["matriz"][:, 0].tolist(), [float(v[1:]) for v in tabela["variant_id"]])
            self.assertEqual(lido["matriz"].shape, (8, 172))

    def test_manifesto_incompleto_e_recusado(self):
        with tempfile.TemporaryDirectory() as pasta:
            _montar(Path(pasta), completo=False)
            with self.assertRaisesRegex(CacheInvalido, "completo"):
                carregar_cache(Path(pasta), "cabecas_172")

    def test_tabela_que_nao_confere_com_a_identidade_e_recusada(self):
        with tempfile.TemporaryDirectory() as pasta:
            _montar(Path(pasta), identidade_extra={"tabela_sha256_conteudo": "outro"})
            with self.assertRaisesRegex(CacheInvalido, "nao confere"):
                carregar_cache(Path(pasta), "cabecas_172")

    def test_variante_da_tabela_fora_dos_fragmentos_e_recusada(self):
        with tempfile.TemporaryDirectory() as pasta:
            _montar(Path(pasta), faltar="v3")
            with self.assertRaisesRegex(CacheInvalido, "fora dos fragmentos"):
                carregar_cache(Path(pasta), "cabecas_172")


class ParTests(unittest.TestCase):
    def _par(self, **extra_mr):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _montar(Path(a))
            _montar(Path(b), sistema="MR", identidade_extra=extra_mr)
            return carregar_cache(Path(a), "cabecas_172"), carregar_cache(Path(b), "cabecas_172")

    def test_par_que_so_difere_pelo_adapter_passa(self):
        m0, mr = self._par()
        conferir_par(m0, mr)

    def test_par_com_codigo_diferente_e_recusado(self):
        m0, mr = self._par(codigo={"arquivos": {"x.py": "2"}})
        with self.assertRaisesRegex(CacheInvalido, "codigo"):
            conferir_par(m0, mr)

    def test_sistemas_trocados_sao_recusados(self):
        m0, mr = self._par()
        with self.assertRaisesRegex(CacheInvalido, "trocados"):
            conferir_par(mr, m0)


class PoliticaTests(unittest.TestCase):
    def test_treino_da_politica_e_subconjunto_do_cache(self):
        with tempfile.TemporaryDirectory() as pasta:
            _montar(Path(pasta))
            lido = carregar_cache(Path(pasta), "cabecas_172")
            politica = pd.DataFrame({"variant_id": ["v0", "v2", "v4"], "role": ["train", "train", "validation"]})
            linhas = linhas_da_politica(lido, politica)
            self.assertEqual(lido["tabela"]["variant_id"].to_numpy()[linhas["train"]].tolist(), ["v0", "v2"])
            self.assertEqual(len(linhas["validation"]), 2)
            self.assertEqual(len(linhas["selecao"]), 2)

    def test_politica_com_treino_fora_do_cache_e_recusada(self):
        with tempfile.TemporaryDirectory() as pasta:
            _montar(Path(pasta))
            lido = carregar_cache(Path(pasta), "cabecas_172")
            politica = pd.DataFrame({"variant_id": ["v0", "zz"], "role": ["train", "train"]})
            with self.assertRaisesRegex(CacheInvalido, "aninhadas"):
                linhas_da_politica(lido, politica)


if __name__ == "__main__":
    unittest.main()
