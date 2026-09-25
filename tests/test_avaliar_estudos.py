"""G7, o script inteiro no modo ENSAIO sobre um release sintetico (hash logico real do Mosaic). Sem torch: roda no
Windows. Com pyyaml (notebook), usa o `comparator_score` real do Mosaic; sem ele, a mesma formula escrita a parte.

    PYTHONPATH=. REQUIRE_NO_SKIP=1 MOSAIC_ROOT=~/testeArq/lumina-mosaic python3 tests/test_avaliar_estudos.py
"""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha import estudos  # noqa: E402
from scripts import avaliar_estudos as avaliar  # noqa: E402
from scripts import conferir_cobertura_das_baselines as cobertura  # noqa: E402

_g7 = importlib.util.spec_from_file_location("teste_g7", RAIZ / "tests" / "test_campanha_g7.py")
teste_g7 = importlib.util.module_from_spec(_g7)
_g7.loader.exec_module(teste_g7)


def _raiz_do_mosaic():
    for candidata in (os.environ.get("MOSAIC_ROOT"), Path.home() / "testeArq" / "lumina-mosaic",
                      RAIZ.parent / "lumina-mosaic"):
        if candidata and (Path(candidata).expanduser() / "src" / "mosaic" / "hashing.py").exists():
            return Path(candidata).expanduser()
    return None


MOSAIC = _raiz_do_mosaic()
YAML = importlib.util.find_spec("yaml") is not None


def _exige(disponivel, motivo):
    if os.environ.get("REQUIRE_NO_SKIP"):
        return lambda alvo: alvo
    return unittest.skipUnless(disponivel, motivo)


def _release(pasta: Path, *, resolvida=False):
    """Release minimo (membership + anotacoes), entradas das analises secundarias e a declaracao com a referencia
    desse release. `resolvida` declara margens e bootstrap (exemplo sintetico, nao proposta)."""
    import pyarrow.parquet as pq

    sys.path.insert(0, str(MOSAIC / "src"))
    from mosaic.hashing import logical_contract

    membros = teste_g7.membros_sinteticos(n_pares=50)
    membership = membros.drop(columns=["chrom", "pos_1based", "ref", "alt"]).assign(
        stratum="x", gnomad_af_bin="rare", core_fold=2, br_lab_any=False)
    (pasta / "release" / "studies" / "brazil").mkdir(parents=True)
    membership.to_parquet(pasta / "release" / cobertura.MEMBERSHIP, index=False)
    teste_g7.anotacoes_sinteticas(membros).to_parquet(pasta / "release" / cobertura.ANOTACOES, index=False)
    campanha = json.loads((RAIZ / "configs" / "campanha_r03_desenvolvimento.json").read_text(encoding="utf-8"))
    campanha["g6"]["proveniencia"]["release_do_mosaic"]["logical_hash"] = {
        caminho: {k: v for k, v in logical_contract(pq.read_table(pasta / "release" / caminho), chave).items()
                  if k in ("n", "logical_hash")} for caminho, chave in cobertura.CHAVES.items()}
    if resolvida:
        margens = teste_g7._margens()
        campanha["g6"]["margens"] = margens
        campanha["g6"]["bootstrap_da_interacao"] = dict(teste_g7.BOOTSTRAP)
    (pasta / "campanha.json").write_text(json.dumps(campanha), encoding="utf-8")
    controles = membros[(membros["study_id"] == estudos.ESTUDO_CLINICO) & (membros["member_role"] == "control")]
    (pasta / "regra_ampla.txt").write_text("\n".join(controles["variant_id"].iloc[:3]) + "\n", encoding="utf-8")
    pd.DataFrame({"variant_id": membros["variant_id"], "study_id": membros["study_id"],
                  "n_janela": np.arange(len(membros)) % 3}).to_parquet(pasta / "exposicao.parquet", index=False)
    return ["--release-root", str(pasta / "release"), "--mosaic-root", str(MOSAIC), "--campanha",
            str(pasta / "campanha.json"), "--entrada", f"regra_ampla={pasta / 'regra_ampla.txt'}",
            "--entrada", f"exposicao={pasta / 'exposicao.parquet'}", "--replicas", "10"]


@_exige(MOSAIC, "sem o repositorio do Mosaic (MOSAIC_ROOT)")
class EnsaioTests(unittest.TestCase):
    def setUp(self):
        self.original = cobertura.carregar_mosaic
        if not YAML:   # sem pyyaml o comparators.yaml nao carrega: a mesma formula, escrita a parte
            sys.path.insert(0, str(MOSAIC / "src"))
            from mosaic.hashing import logical_contract

            cobertura.carregar_mosaic = lambda raiz: (
                logical_contract, dict(cobertura.GNOMAD_RARITY, domain="all_panels"),
                lambda linha, spec: teste_g7.pontuar_gnomad(linha),
                {"mosaic_root": str(raiz), "commit": "teste", "comparators_yaml_sha256": "x" * 64})

    def tearDown(self):
        cobertura.carregar_mosaic = self.original

    def test_ensaio_na_membership_com_scores_sinteticos(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            argumentos = _release(pasta)
            self.assertEqual(avaliar.main(["--ensaio-sintetico", *argumentos, "--out-dir", str(pasta / "e1")]), 0)
            r = json.loads((pasta / "e1" / "ensaio_relatorio.json").read_text(encoding="utf-8"))
            self.assertEqual(r["modo"], "ENSAIO")
            self.assertIn("SINTETICOS", r["aviso"])
            self.assertEqual(sorted(r["estudos"]), sorted(estudos.ESTUDOS))
            clinico = r["estudos"][estudos.ESTUDO_CLINICO]
            self.assertIn("brier", clinico["sistemas"]["coortes"][estudos.COORTE_COMPLETO]["coorte"]["delta"])
            self.assertEqual(clinico["sistemas"]["sensibilidades"]["sem_controles_com_scv_brasileira"]["pares_retirados"],
                             3)
            self.assertIn("pares_mantidos", clinico["sistemas"]["sensibilidades"]["exposicao_empatada"])
            populacional = r["estudos"][estudos.ESTUDO_POPULACIONAL]
            self.assertIn("nao_aplicavel", populacional["baselines"]["ausencia_no_abraom"])
            self.assertFalse(r["margens"]["avaliado"], "a declaracao de hoje nao tem margens")
            self.assertEqual(r["baselines"]["cobertura_do_score"]["gnomad_rarity"]["pontuadas"],
                             r["baselines"]["cobertura_do_score"]["gnomad_rarity"]["membros_unicos"])
            self.assertFalse((pasta / "e1" / "g7_pontos.parquet").exists(), "o ensaio nao grava scores")
            self.assertEqual(avaliar.main(["--ensaio-sintetico", *argumentos, "--out-dir", str(pasta / "e1")]), 2)

    def test_ensaio_com_margens_declaradas_aplica_as_regras(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            argumentos = _release(pasta, resolvida=True)
            self.assertEqual(avaliar.main(["--ensaio-sintetico", *argumentos, "--out-dir", str(pasta / "e2")]), 0)
            r = json.loads((pasta / "e2" / "ensaio_relatorio.json").read_text(encoding="utf-8"))
            self.assertTrue(r["margens"]["avaliado"])
            regras = r["margens"]["por_estudo"][estudos.ESTUDO_CLINICO]["regras"]
            self.assertIn(regras["melhoria_minima_no_coorte_br"]["atende"], (True, False))

    def test_entradas_erradas_sao_recusadas(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            argumentos = _release(pasta)
            self.assertEqual(avaliar.main(["--ensaio-sintetico", "--manifesto", str(pasta), *argumentos,
                                           "--out-dir", str(pasta / "x1")]), 2, "o ensaio nao le manifesto")
            self.assertEqual(avaliar.main([*argumentos, "--out-dir", str(pasta / "x2")]), 2,
                             "o modo real exige manifesto")
            indice = next(i for i, a in enumerate(argumentos) if a.startswith("exposicao="))
            sem_exposicao = argumentos[:indice - 1] + argumentos[indice + 1:]
            self.assertEqual(avaliar.main(["--ensaio-sintetico", *sem_exposicao, "--out-dir", str(pasta / "x3")]), 2)
            anotacoes = pd.read_parquet(pasta / "release" / cobertura.ANOTACOES)
            anotacoes.loc[0, "gnomad_v4_af"] = 0.5
            anotacoes.to_parquet(pasta / "release" / cobertura.ANOTACOES, index=False)
            self.assertEqual(avaliar.main(["--ensaio-sintetico", *argumentos, "--out-dir", str(pasta / "x4")]), 2,
                             "anotacoes com outro hash logico")
            self.assertFalse(any((pasta / f"x{i}").exists() for i in range(1, 5)))


if __name__ == "__main__":
    unittest.main()
