"""Cobertura das baselines (G6): regras puras sempre; hash logico real do Mosaic se o repositorio estiver ao lado;
o script inteiro se houver tambem pyyaml (no notebook). So dados sinteticos."""
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
from scripts import conferir_cobertura_das_baselines as cob  # noqa: E402


def _raiz_do_mosaic():
    for candidata in (os.environ.get("MOSAIC_ROOT"), Path.home() / "testeArq" / "lumina-mosaic",
                      RAIZ.parent / "lumina-mosaic"):
        if candidata and (Path(candidata) / "src" / "mosaic" / "hashing.py").exists():
            return Path(candidata)
    return None


MOSAIC = _raiz_do_mosaic()
YAML = importlib.util.find_spec("yaml") is not None


def _exige(disponivel, motivo):
    """skipUnless -- exceto com REQUIRE_NO_SKIP=1 (convencao do repositorio): ai o teste roda e FALHA se faltar a
    dependencia, porque teste pulado nao e PASS."""
    if os.environ.get("REQUIRE_NO_SKIP"):
        return lambda alvo: alvo
    return unittest.skipUnless(disponivel, motivo)


def _pontuar(linha):
    """A formula do gnomad_rarity (config/comparators.yaml) escrita a parte, para os testes puros."""
    af = linha.get("gnomad_v4_af")
    af = 0.0 if linha.get("gnomad_status") in ("not_found", "ac0") else af
    return None if af is None or af != af else -float(af)


def _dados():
    membros = pd.DataFrame({
        "variant_id": ["a", "b", "c", "d", "e", "f"],
        "study_id": ["br_clinical_evidence"] * 3 + ["br_population_observed"] * 3,
        "member_role": ["case", "control", "unmatched_case", "case", "control", "unmatched_case"],
        "present_abraom": [True, False, True, True, False, True]})
    anotacoes = pd.DataFrame({
        "variant_id": ["a", "b", "c", "d", "e", "f", "z"],
        "gnomad_v4_af": [1e-5, np.nan, 0.0, 2e-4, np.nan, np.nan, 0.3],
        "gnomad_status": ["present", "not_found", "ac0", "present", "not_found", None, "present"],
        "abraom_af": [0.001, np.nan, 0.01, 0.002, np.nan, 0.004, np.nan],
        "abraom_status": ["present", "not_found", "present", "present", "not_found", "present", "not_found"],
        "present_abraom": [True, False, True, True, False, True, False]})
    return membros, anotacoes


class CoberturaTests(unittest.TestCase):
    def test_contagens_por_estudo_e_papel_sem_rotulo(self):
        membros, anotacoes = _dados()
        c = cob.cobertura(membros, anotacoes, _pontuar)
        self.assertEqual(sorted(c), sorted(f"{e}/{p}" for e, p in zip(membros.study_id, membros.member_role)))
        self.assertEqual(c["br_clinical_evidence/control"]["gnomad_rarity_definido"], 1, "not_found vira AF 0")
        self.assertEqual(c["br_population_observed/unmatched_case"]["gnomad_rarity_definido"], 0, "status ausente")
        self.assertEqual(c["br_population_observed/unmatched_case"]["gnomad_status"], {"ausente": 1})
        self.assertEqual(c["br_clinical_evidence/control"]["abraom_af_finito"], 0)
        self.assertEqual(cob.problemas_da_cobertura(c), [], "cobertura incompleta do gnomad_rarity nao reprova")
        for grupo in c.values():
            self.assertNotIn("binary_label", json.dumps(grupo))

    def test_membro_sem_anotacao_ou_presenca_discordante_reprova(self):
        membros, anotacoes = _dados()
        problemas = cob.problemas_da_cobertura(cob.cobertura(membros, anotacoes[anotacoes.variant_id != "b"],
                                                             _pontuar))
        self.assertEqual(problemas, ["br_clinical_evidence/control: 1 membros sem linha em pb_annotations"])
        anotacoes.loc[anotacoes.variant_id == "d", "present_abraom"] = False
        problemas = cob.problemas_da_cobertura(cob.cobertura(membros, anotacoes, _pontuar))
        self.assertEqual(problemas, ["br_population_observed/case: 1 membros com present_abraom discordante"])

    def test_anotacao_com_id_repetido_e_recusada(self):
        membros, anotacoes = _dados()
        with self.assertRaises(ValueError):
            cob.cobertura(membros, pd.concat([anotacoes, anotacoes.iloc[:1]]), _pontuar)

    def test_especificacao_e_contrato(self):
        self.assertEqual(cob.conferir_especificacao(dict(cob.GNOMAD_RARITY, domain="all_panels")), [])
        self.assertTrue(cob.conferir_especificacao(dict(cob.GNOMAD_RARITY, missing="leave_null")))
        self.assertEqual(cob.conferir_especificacao(None), ["gnomad_rarity ausente dos comparadores oficiais"])
        contrato = {"logical_hash_version": "v2", "logical_hash": "x", "n": 3, "schema": [], "primary_key": ["id"]}
        self.assertEqual(cob.comparar_contrato(contrato, dict(contrato), "p"), [])
        self.assertEqual(cob.comparar_contrato(contrato, dict(contrato, n=4), "p"), ["p: n recalculado != manifesto"])
        self.assertEqual(cob.comparar_contrato(None, contrato, "p"), ["p: sem contrato no manifesto"])
        self.assertEqual(cob.comparar_contrato(contrato, dict(contrato, schema=["x"]), "p", campos=("logical_hash", "n")),
                         [], "contra a referencia declarada so contam hash e n")

    def test_a_referencia_declarada_e_a_do_mosaic(self):
        from eval.campanha.recortes import carregar_campanha

        referencia = carregar_campanha(RAIZ / "configs" / "campanha_r03_desenvolvimento.json")["g6"][
            "proveniencia"]["release_do_mosaic"]
        self.assertEqual(sorted(referencia["logical_hash"]), sorted(cob.CHAVES))
        self.assertTrue(referencia["logical_hash"]["studies/brazil/membership.parquet"]["logical_hash"]
                        .startswith("1c1cd65d"), "o conferido no G1")
        self.assertEqual(referencia["logical_hash"]["studies/brazil/membership.parquet"]["n"], 8875)


def _release(pasta: Path, membros: pd.DataFrame, anotacoes: pd.DataFrame, manifesto="bundle.manifest.json"):
    """Um release minimo, com o manifesto calculado pelo proprio Mosaic (no layout anterior a ADR 0006, como o da
    copia do notebook), e a declaracao da campanha com a referencia desse release sintetico."""
    import pyarrow.parquet as pq

    sys.path.insert(0, str(MOSAIC / "src"))
    from mosaic.hashing import logical_contract

    (pasta / "studies" / "brazil").mkdir(parents=True)
    anotacoes.to_parquet(pasta / cob.ANOTACOES, index=False)
    membros.to_parquet(pasta / cob.MEMBERSHIP, index=False)
    contratos = [{"path": caminho, **logical_contract(pq.read_table(pasta / caminho), chave)}
                 for caminho, chave in cob.CHAVES.items()]
    (pasta / manifesto).write_text(json.dumps({"release_id": "teste", "artifact_contracts": contratos}),
                                   encoding="utf-8")
    campanha = json.loads((RAIZ / "configs" / "campanha_r03_desenvolvimento.json").read_text(encoding="utf-8"))
    campanha["g6"]["proveniencia"]["release_do_mosaic"]["logical_hash"] = {
        c["path"]: {"n": c["n"], "logical_hash": c["logical_hash"]} for c in contratos}
    campanha["g6"]["proveniencia"]["release_do_mosaic"]["artifact_contracts_hash"] = None
    (pasta / "campanha.json").write_text(json.dumps(campanha), encoding="utf-8")
    return pasta / "campanha.json"


@_exige(MOSAIC, "sem o repositorio do Mosaic (MOSAIC_ROOT)")
class ContratoRealTests(unittest.TestCase):
    def test_hash_logico_do_mosaic_detecta_anotacao_mudada(self):
        import pyarrow.parquet as pq

        membros, anotacoes = _dados()
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            _release(pasta, membros, anotacoes)
            from mosaic.hashing import logical_contract

            publicados = {c["path"]: c for c in json.loads((pasta / "bundle.manifest.json").read_text())[
                "artifact_contracts"]}
            recalculado = logical_contract(pq.read_table(pasta / cob.ANOTACOES), ("variant_id",))
            self.assertEqual(cob.comparar_contrato(publicados[cob.ANOTACOES], recalculado, cob.ANOTACOES), [])
            mudada = anotacoes.copy()
            mudada.loc[0, "gnomad_v4_af"] = 2e-5
            mudada.to_parquet(pasta / cob.ANOTACOES, index=False)
            recalculado = logical_contract(pq.read_table(pasta / cob.ANOTACOES), ("variant_id",))
            self.assertIn(f"{cob.ANOTACOES}: logical_hash recalculado != manifesto",
                          cob.comparar_contrato(publicados[cob.ANOTACOES], recalculado, cob.ANOTACOES))

    @_exige(YAML, "sem pyyaml: o comparators.yaml do Mosaic nao carrega aqui (roda no notebook)")
    def test_script_inteiro(self):
        membros, anotacoes = _dados()
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            campanha = _release(pasta / "release", membros, anotacoes)
            argumentos = ["--release-root", str(pasta / "release"), "--mosaic-root", str(MOSAIC),
                          "--campanha", str(campanha)]
            self.assertEqual(cob.main([*argumentos, "--out", str(pasta / "saida.json")]), 0)
            saida = json.loads((pasta / "saida.json").read_text(encoding="utf-8"))
            self.assertTrue(saida["passou"])
            self.assertEqual(saida["cobertura"]["br_clinical_evidence/control"]["gnomad_rarity_definido"], 1)
            self.assertEqual(cob.main([*argumentos, "--out", str(pasta / "saida.json")]), 2, "nao sobrescreve")
            mudada = copy.deepcopy(anotacoes)
            mudada.loc[0, "abraom_af"] = 0.5
            mudada.to_parquet(pasta / "release" / cob.ANOTACOES, index=False)
            self.assertEqual(cob.main([*argumentos, "--out", str(pasta / "saida2.json")]), 2)


if __name__ == "__main__":
    unittest.main()
