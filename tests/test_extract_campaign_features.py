"""Extrator do G3: o que se testa sem GPU -- identidade do cache, retomada e nomes indefinidos."""
import builtins
import json
import symtable
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import extract_campaign_features as extrator  # noqa: E402


def _tabela():
    return pd.DataFrame({"variant_id": ["a", "b", "c"], "papel": ["train", "validation", "selecao"]})


def _ident(**troca):
    args = Namespace(sistema="MR", semente_do_adapter=20260921, window_bp=4096, variantes_por_lote=8)
    base = extrator.identidade(args, tabela=_tabela(), checkpoint_sha="f2983560" + "0" * 56,
                               adapter_sha="6327a9fa" + "0" * 56, fasta_sha="056974f6" + "0" * 56,
                               revisao="abc")
    base.update(troca)
    return base


class ExtratorTests(unittest.TestCase):
    def test_funcoes_nao_referenciam_nomes_indefinidos(self):
        # O mesmo cuidado do runner do adapter: um NameError so apareceria depois de horas de GPU.
        caminho = Path(extrator.__file__)
        tabela = symtable.symtable(caminho.read_text(encoding="utf-8"), str(caminho), "exec")
        disponiveis = set(vars(extrator)) | set(vars(builtins))
        for funcao in tabela.get_children():
            indefinidos = {s.get_name() for s in funcao.get_symbols()
                           if s.is_referenced() and s.is_global() and s.get_name() not in disponiveis}
            self.assertEqual(indefinidos, set(), funcao.get_name())

    def test_identidade_carrega_o_que_muda_numero(self):
        ident = _ident()
        self.assertEqual(ident["lote"]["sequencias_por_forward"], 16)
        self.assertEqual(ident["indice_focal"], 2047)
        self.assertEqual(ident["extracoes"]["cabecas_172"]["dims"], 172)
        self.assertEqual(ident["papeis"], ["selecao", "train", "validation"])
        self.assertEqual(len(ident["tabela_sha256_composicao"]), 64)

    def test_cache_novo_grava_e_retomada_igual_passa(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta, "M0")
            self.assertIsNone(extrator.conferir_identidade(destino, _ident()))
            self.assertTrue((destino / "identidade.json").exists())
            self.assertIsNone(extrator.conferir_identidade(destino, _ident(revisao_do_codigo="outra")),
                              "revisao do codigo nao muda numero: nao bloqueia a retomada")

    def test_retomada_com_outro_sistema_ou_outro_lote_e_recusada(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta, "cache")
            extrator.conferir_identidade(destino, _ident())
            problema = extrator.conferir_identidade(destino, _ident(adapter_sha256="outro"))
            self.assertIn("adapter_sha256", problema)
            args = Namespace(sistema="MR", semente_do_adapter=20260921, window_bp=4096, variantes_por_lote=4)
            outro_lote = extrator.identidade(args, tabela=_tabela(), checkpoint_sha="f2983560" + "0" * 56,
                                             adapter_sha="6327a9fa" + "0" * 56, fasta_sha="056974f6" + "0" * 56,
                                             revisao="abc")
            self.assertIn("lote", extrator.conferir_identidade(destino, outro_lote))

    def test_ja_extraidas_junta_os_fragmentos(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            for i, ids in enumerate((["a", "b"], ["c"])):
                np.savez(destino / f"fragmento_{i:05d}.npz", variant_id=np.array(ids),
                         papel=np.array(["train"] * len(ids)), cabecas_172=np.zeros((len(ids), 172)))
            self.assertEqual(extrator.ja_extraidas(destino), {"a", "b", "c"})

    def test_mr_sem_adapter_e_recusado_antes_de_carregar_qualquer_coisa(self):
        self.assertEqual(extrator.main(["--sistema", "MR", "--checkpoint", "x", "--fasta", "x", "--snapshot", "x",
                                        "--selecao", "x", "--estudos", "x", "--out-dir", "x"]), 2)
        self.assertEqual(extrator.main(["--sistema", "M0", "--adapter", "a.pt", "--semente-do-adapter", "1",
                                        "--checkpoint", "x", "--fasta", "x", "--snapshot", "x", "--selecao", "x",
                                        "--estudos", "x", "--out-dir", "x"]), 2)


if __name__ == "__main__":
    unittest.main()
