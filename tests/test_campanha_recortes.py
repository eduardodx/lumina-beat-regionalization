"""Recortes de desenvolvimento: o fold 0 e os estudos brasileiros nao entram, nem por engano. Sem torch."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha import recortes  # noqa: E402
from eval.campanha.layout import EXTRACOES, RAIO_DO_CONTEXTO, limites_do_contexto, lote_pareado  # noqa: E402

DECLARACAO = RAIZ / "configs" / "campanha_r03_desenvolvimento.json"


def _linha(vid, role, chrom="chr1", label=1, painel="missense", cluster="c1"):
    return {"variant_id": vid, "role": role, "chrom": chrom, "pos_1based": 1000, "ref": "A", "alt": "G",
            "binary_label": label, "primary_panel": painel, "overlap_cluster_id": cluster, "label_tier": "gold"}


def _entradas():
    snapshot = pd.DataFrame([_linha("t1", "train"), _linha("t2", "train", label=0, cluster="c2"),
                             _linha("v1", "validation", cluster="c3"), _linha("x1", "test", cluster="c4")])
    selecao = pd.DataFrame([_linha("s1", "selection", cluster="c9")]).drop(columns=["role"])
    estudos = pd.DataFrame({"variant_id": ["m1", "m2"]})
    return snapshot, selecao, estudos


class RecortesTests(unittest.TestCase):
    def test_declaracao_real_e_valida_e_pareada(self):
        campanha = recortes.carregar_campanha(DECLARACAO)
        combinacoes = campanha["sementes"]["combinacoes"]
        self.assertEqual([c["adapter"] for c in combinacoes], [20260921, 20260922, 20260923])
        self.assertEqual([c["cabeca"] for c in combinacoes], [11, 12, 13])

    def test_declaracao_desalinhada_e_recusada(self):
        campanha = json.loads(DECLARACAO.read_text(encoding="utf-8"))
        errada = copy.deepcopy(campanha)
        errada["sementes"]["combinacoes"][0]["cabeca"] = 12
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta, "c.json")
            caminho.write_text(json.dumps(errada), encoding="utf-8")
            with self.assertRaises(ValueError):
                recortes.carregar_campanha(caminho)

    def test_adapter_a1_congelado_esta_declarado(self):
        registro = recortes.adapter_congelado(recortes.carregar_campanha(DECLARACAO), 20260921)
        self.assertTrue(registro["sha256"].startswith("6327a9fa"))
        with self.assertRaises(ValueError):
            recortes.adapter_congelado(recortes.carregar_campanha(DECLARACAO), 20260922)

    def test_tabela_tem_so_os_papeis_de_desenvolvimento(self):
        tabela = recortes.tabela_de_extracao(*_entradas())
        self.assertEqual(sorted(tabela["papel"].unique()), ["selecao", "train", "validation"])
        self.assertNotIn("x1", set(tabela["variant_id"]), "o fold 0 nao entra")
        self.assertEqual(dict(zip(tabela["variant_id"], tabela["papel"]))["s1"], "selecao")

    def test_pedir_o_papel_test_e_recusado(self):
        with self.assertRaises(ValueError):
            recortes.tabela_de_extracao(*_entradas(), papeis=("train", "test"))

    def test_variante_do_fold0_na_selecao_e_recusada(self):
        snapshot, selecao, estudos = _entradas()
        selecao = pd.concat([selecao, pd.DataFrame([_linha("x1", "selection")]).drop(columns=["role"])])
        with self.assertRaisesRegex(ValueError, "fold 0"):
            recortes.tabela_de_extracao(snapshot, selecao, estudos)

    def test_membro_dos_estudos_e_recusado(self):
        snapshot, selecao, estudos = _entradas()
        estudos = pd.DataFrame({"variant_id": ["t1"]})
        with self.assertRaisesRegex(ValueError, "estudos brasileiros"):
            recortes.tabela_de_extracao(snapshot, selecao, estudos)

    def test_chr8_e_recusado_com_ou_sem_prefixo(self):
        for chrom in ("chr8", "8"):
            snapshot, selecao, estudos = _entradas()
            snapshot.loc[0, "chrom"] = chrom
            with self.assertRaisesRegex(ValueError, "chr8"):
                recortes.tabela_de_extracao(snapshot, selecao, estudos)

    def test_variante_em_dois_papeis_e_recusada(self):
        snapshot, selecao, estudos = _entradas()
        selecao = pd.concat([selecao, pd.DataFrame([_linha("t1", "selection")]).drop(columns=["role"])])
        with self.assertRaisesRegex(ValueError, "mais de um papel"):
            recortes.tabela_de_extracao(snapshot, selecao, estudos)

    def test_hash_nao_depende_da_ordem(self):
        tabela = recortes.tabela_de_extracao(*_entradas())
        embaralhada = tabela.sample(frac=1.0, random_state=3)
        self.assertEqual(recortes.hash_da_tabela(tabela), recortes.hash_da_tabela(embaralhada))

    def test_hash_de_conteudo_pega_coordenada_e_rotulo(self):
        # O de composicao (variant_id + papel) nao mudava com eles: foi o que a revisao de 23/09 reproduziu.
        tabela = recortes.tabela_de_extracao(*_entradas())
        for coluna, valor in (("pos_1based", 1001), ("binary_label", 0), ("alt", "T"),
                              ("overlap_cluster_id", "outro")):
            trocada = tabela.copy()
            trocada.loc[0, coluna] = valor
            self.assertEqual(recortes.hash_da_tabela(tabela), recortes.hash_da_tabela(trocada))
            self.assertNotEqual(recortes.hash_do_conteudo(tabela), recortes.hash_do_conteudo(trocada), coluna)
        self.assertEqual(recortes.hash_do_conteudo(tabela), recortes.hash_do_conteudo(tabela.iloc[::-1]))

    def test_resumo_conta_rotulos_e_clusters(self):
        resumo = recortes.resumo_da_tabela(recortes.tabela_de_extracao(*_entradas()))
        self.assertEqual(resumo["train"]["variantes"], 2)
        self.assertEqual(resumo["train"]["por_rotulo"], {"0": 1, "1": 1})
        self.assertEqual(resumo["train"]["clusters"], 2)


class LayoutTests(unittest.TestCase):
    def test_lote_pareado_intercala_e_completa_com_copias(self):
        sequencias, reais = lote_pareado(["R0", "R1"], ["A0", "A1"], 4)
        self.assertEqual(reais, 2)
        self.assertEqual(sequencias, ["R0", "A0", "R1", "A1", "R1", "A1", "R1", "A1"])

    def test_lote_pareado_recusa_excesso_e_vazio(self):
        with self.assertRaises(ValueError):
            lote_pareado(["R"] * 3, ["A"] * 3, 2)
        with self.assertRaises(ValueError):
            lote_pareado([], [], 2)

    def test_contexto_local_e_o_da_leitura_antiga(self):
        # compute_pool_bounds(2047, 64, 4096) = (max(0, 2047-64), min(4096, 2047+64)) = (1983, 2111): 128 bases.
        self.assertEqual(limites_do_contexto(2047, 4096), (1983, 2111))
        self.assertEqual(limites_do_contexto(10, 4096), (0, 74))
        self.assertEqual(RAIO_DO_CONTEXTO, 64)

    def test_dimensoes_declaradas(self):
        self.assertEqual(EXTRACOES["cabecas_172"][1], 68 + 10 + 78 + 16)
        self.assertEqual(EXTRACOES["leitura_antiga_1344"][1], 3 * 448)


if __name__ == "__main__":
    unittest.main()
