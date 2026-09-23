"""Cache do G3: completude estrita, gravacao atomica, validacao na retomada, numeracao e trava. Sem torch.

Cada teste reproduz um defeito apontado na revisao de 23/09, antes da extracao longa.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.campanha import cache  # noqa: E402


def _tabela(ids=("a", "b", "c"), papeis=("train", "train", "selecao")):
    return pd.DataFrame({"variant_id": list(ids), "papel": list(papeis)})


def _matrizes(n: int, valor: float = 0.5):
    return {"cabecas_172": np.full((n, 172), valor, dtype=np.float32),
            "leitura_antiga_1344": np.full((n, 1344), valor, dtype=np.float32)}


def _gravar(destino, indice, ids, papeis, **troca):
    matrizes = _matrizes(len(ids))
    matrizes.update(troca)
    return cache.gravar_fragmento(destino, indice, variant_id=np.array(ids), papel=np.array(papeis),
                                  matrizes=matrizes)


class CompletudeTests(unittest.TestCase):
    def test_falha_de_janela_nao_vira_completo(self):
        # O contraexemplo da revisao: 2 variantes, 1 falha non_acgt -> antes saia completo=true e exit 0.
        estado = cache.estado_do_cache(_tabela(("a", "b"), ("train", "train")), {"a"})
        self.assertFalse(estado["completo"])
        self.assertEqual(estado["faltando"], 1)
        self.assertEqual(estado["exemplos_faltando"], ["b"])

    def test_tabela_inteira_no_cache_e_completo(self):
        self.assertTrue(cache.estado_do_cache(_tabela(), {"a", "b", "c"})["completo"])


class GravacaoTests(unittest.TestCase):
    def test_grava_e_rele_sem_deixar_temporario(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            _gravar(destino, 0, ["a", "b"], ["train", "train"])
            self.assertEqual(list(destino.glob("*.tmp")), [])
            feitas, problemas = cache.ler_fragmentos(destino, _tabela())
            self.assertEqual((feitas, problemas), ({"a", "b"}, []))

    def test_nao_finito_e_recusado_antes_de_gravar(self):
        with tempfile.TemporaryDirectory() as pasta:
            ruim = np.full((1, 172), np.nan, dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "nao finitos"):
                _gravar(Path(pasta), 0, ["a"], ["train"], cabecas_172=ruim)
            self.assertEqual(list(Path(pasta).iterdir()), [], "nada pode ficar em disco")

    def test_forma_errada_e_recusada_antes_de_gravar(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaisesRegex(ValueError, "forma"):
                _gravar(Path(pasta), 0, ["a"], ["train"], cabecas_172=np.zeros((1, 171), dtype=np.float32))

    def test_nao_grava_por_cima_de_fragmento_existente(self):
        with tempfile.TemporaryDirectory() as pasta:
            _gravar(Path(pasta), 0, ["a"], ["train"])
            with self.assertRaises(FileExistsError):
                _gravar(Path(pasta), 0, ["b"], ["train"])

    def test_temporario_de_queda_e_limpo_e_nao_conta_como_feito(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            (destino / "fragmento_00000.npz.tmp").write_bytes(b"meio arquivo")
            self.assertEqual(cache.limpar_temporarios(destino), ["fragmento_00000.npz.tmp"])
            self.assertEqual(cache.ler_fragmentos(destino, _tabela()), (set(), []))


class RetomadaTests(unittest.TestCase):
    def test_fragmento_truncado_e_problema_e_nao_excecao(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            _gravar(destino, 0, ["a"], ["train"])
            conteudo = (destino / "fragmento_00000.npz").read_bytes()
            (destino / "fragmento_00001.npz").write_bytes(conteudo[: len(conteudo) // 2])
            feitas, problemas = cache.ler_fragmentos(destino, _tabela())
            self.assertEqual(feitas, {"a"})
            self.assertTrue(any("ilegivel" in p for p in problemas), problemas)

    def test_variante_repetida_entre_fragmentos_e_problema(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            _gravar(destino, 0, ["a"], ["train"])
            _gravar(destino, 1, ["a", "b"], ["train", "train"])
            _, problemas = cache.ler_fragmentos(destino, _tabela())
            self.assertTrue(any("repetidas" in p for p in problemas), problemas)

    def test_variante_fora_da_tabela_ou_com_outro_papel_e_problema(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            _gravar(destino, 0, ["z", "c"], ["train", "train"])  # z nao existe; c e selecao na tabela
            _, problemas = cache.ler_fragmentos(destino, _tabela())
            self.assertTrue(any("fora da tabela" in p for p in problemas), problemas)
            self.assertTrue(any("papel diferente" in p for p in problemas), problemas)

    def test_nao_finito_gravado_por_fora_e_pego_na_releitura(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            ruim = _matrizes(1)
            ruim["leitura_antiga_1344"][0, 5] = np.inf
            np.savez(destino / "fragmento_00000.npz", variant_id=np.array(["a"]), papel=np.array(["train"]), **ruim)
            _, problemas = cache.ler_fragmentos(destino, _tabela())
            self.assertTrue(any("nao finitos" in p for p in problemas), problemas)

    def test_proximo_indice_e_o_maior_mais_um_mesmo_com_lacuna(self):
        # Antes: a CONTAGEM de arquivos. Com 00000 e 00002, a contagem (2) sobrescreveria o 00002.
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            _gravar(destino, 0, ["a"], ["train"])
            _gravar(destino, 2, ["b"], ["train"])
            self.assertEqual(cache.proximo_indice(destino), 3)


class TravaTests(unittest.TestCase):
    def test_segunda_extracao_no_mesmo_cache_e_recusada(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            (destino / cache.ARQUIVO_DA_TRAVA).write_text(str(os.getppid()), encoding="utf-8")  # processo vivo
            with self.assertRaisesRegex(RuntimeError, "ja esta sendo extraido"):
                with cache.trava(destino):
                    pass

    def test_trava_velha_e_removida_e_a_propria_e_liberada(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            (destino / cache.ARQUIVO_DA_TRAVA).write_text("999999999", encoding="utf-8")  # PID inexistente
            with cache.trava(destino):
                self.assertEqual((destino / cache.ARQUIVO_DA_TRAVA).read_text(encoding="utf-8"), str(os.getpid()))
            self.assertFalse((destino / cache.ARQUIVO_DA_TRAVA).exists())


class CodigoTests(unittest.TestCase):
    def test_mudar_um_arquivo_muda_a_identidade_do_codigo(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz / "pacote").mkdir()
            (raiz / "a.py").write_text("x = 1\n", encoding="utf-8")
            (raiz / "pacote" / "m.py").write_text("y = 1\n", encoding="utf-8")
            antes = cache.identidade_do_codigo(raiz, ("a.py",), {"pacote": raiz / "pacote"})
            (raiz / "pacote" / "m.py").write_text("y = 2\n", encoding="utf-8")
            depois = cache.identidade_do_codigo(raiz, ("a.py",), {"pacote": raiz / "pacote"})
            self.assertEqual(antes["arquivos"], depois["arquivos"])
            self.assertNotEqual(antes["pacote_pacote"], depois["pacote_pacote"])


if __name__ == "__main__":
    unittest.main()
