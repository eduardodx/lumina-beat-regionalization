"""Extrator do G3: o que se testa sem GPU -- identidade do cache, tabela gravada, retomada e nomes indefinidos."""
import builtins
import symtable
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import extract_campaign_features as extrator  # noqa: E402


def _tabela(**troca):
    linhas = [
        {"variant_id": "a", "papel": "train", "chrom": "chr1", "pos_1based": 100, "ref": "A", "alt": "G",
         "binary_label": 1, "primary_panel": "missense", "overlap_cluster_id": "c1", "label_tier": "gold"},
        {"variant_id": "b", "papel": "validation", "chrom": "chr2", "pos_1based": 200, "ref": "C", "alt": "T",
         "binary_label": 0, "primary_panel": "splice", "overlap_cluster_id": "c2", "label_tier": "gold"},
        {"variant_id": "c", "papel": "selecao", "chrom": "chr3", "pos_1based": 300, "ref": "G", "alt": "A",
         "binary_label": 0, "primary_panel": "plof", "overlap_cluster_id": "c3", "label_tier": "consensus"},
    ]
    tabela = pd.DataFrame(linhas)
    for coluna, valor in troca.items():
        tabela.loc[0, coluna] = valor
    return tabela


def _ident(tabela=None, variantes_por_lote=8, **troca):
    args = Namespace(sistema="MR", semente_do_adapter=20260921, window_bp=4096,
                     variantes_por_lote=variantes_por_lote, fragmento=4096)
    base = extrator.identidade(args, tabela=_tabela() if tabela is None else tabela,
                               checkpoint_sha="f2983560" + "0" * 56, adapter_sha="6327a9fa" + "0" * 56,
                               fasta_sha="056974f6" + "0" * 56, revisao="abc",
                               codigo={"arquivos": {"eval/campanha/leituras.py": "1" * 64}},
                               ambiente={"torch": "2.x", "gpu": "A10G"})
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
        self.assertEqual(len(ident["tabela_sha256_conteudo"]), 64)
        self.assertIn("codigo", ident)
        self.assertIn("ambiente", ident)

    def test_arquivos_declarados_existem(self):
        raiz = Path(extrator.__file__).resolve().parents[1]
        faltando = [a for a in extrator.ARQUIVOS_QUE_DETERMINAM_AS_FEATURES if not (raiz / a).exists()]
        self.assertEqual(faltando, [])

    def test_cache_novo_grava_e_retomada_igual_passa(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta, "M0")
            self.assertIsNone(extrator.conferir_identidade(destino, _ident()))
            self.assertTrue((destino / "identidade.json").exists())
            self.assertIsNone(extrator.conferir_identidade(destino, _ident(revisao_do_codigo="outra")),
                              "o commit do git sozinho nao muda numero: nao bloqueia a retomada")

    def test_retomada_com_outro_adapter_lote_codigo_ou_ambiente_e_recusada(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta, "cache")
            extrator.conferir_identidade(destino, _ident())
            for troca, campo in ((_ident(adapter_sha256="outro"), "adapter_sha256"),
                                 (_ident(variantes_por_lote=4), "lote"),
                                 (_ident(codigo={"arquivos": {"eval/campanha/leituras.py": "2" * 64}}), "codigo"),
                                 (_ident(ambiente={"torch": "2.x", "gpu": "H100"}), "ambiente")):
                self.assertIn(campo, extrator.conferir_identidade(destino, troca))

    def test_coordenada_ou_rotulo_trocado_muda_a_identidade(self):
        # Antes o hash era so de variant_id + papel: trocar uma coordenada preservava a identidade.
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta, "cache")
            extrator.conferir_identidade(destino, _ident())
            for tabela in (_tabela(pos_1based=101), _tabela(binary_label=0)):
                self.assertIn("tabela_sha256_conteudo", extrator.conferir_identidade(destino, _ident(tabela)))

    def test_tabela_gravada_na_criacao_e_so_conferida_na_retomada(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            self.assertIsNone(extrator.conferir_tabela_gravada(destino, _tabela(), novo=True))
            gravada = (destino / "tabela.parquet").read_bytes()
            self.assertIsNone(extrator.conferir_tabela_gravada(destino, _tabela(), novo=False))
            self.assertIn("nao confere", extrator.conferir_tabela_gravada(destino, _tabela(pos_1based=5), novo=False))
            self.assertEqual((destino / "tabela.parquet").read_bytes(), gravada, "a retomada nao reescreve")

    def test_mr_sem_adapter_e_recusado_antes_de_carregar_qualquer_coisa(self):
        self.assertEqual(extrator.main(["--sistema", "MR", "--checkpoint", "x", "--fasta", "x", "--snapshot", "x",
                                        "--selecao", "x", "--estudos", "x", "--out-dir", "x"]), 2)
        self.assertEqual(extrator.main(["--sistema", "M0", "--adapter", "a.pt", "--semente-do-adapter", "1",
                                        "--checkpoint", "x", "--fasta", "x", "--snapshot", "x", "--selecao", "x",
                                        "--estudos", "x", "--out-dir", "x"]), 2)


if __name__ == "__main__":
    unittest.main()
