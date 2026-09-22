"""Regressoes do runner que nao precisam de GPU nem do checkpoint R03."""
import builtins
import random
import symtable
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_population_adapter as runner


class RunnerTests(unittest.TestCase):
    def test_smoke_nao_referencia_variaveis_do_laco(self):
        # aaa9d40 copiava o bloco de delta para o smoke: NameError apos toda a execucao GPU.
        path = Path(runner.__file__)
        table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        smoke = next(c for c in table.get_children() if c.get_name() == "rodar_smoke")
        available = set(vars(runner)) | set(vars(builtins))
        undefined = {s.get_name() for s in smoke.get_symbols()
                     if s.is_referenced() and s.is_global() and s.get_name() not in available}
        self.assertEqual(undefined, set())

    def test_seed_controla_torch_e_random(self):
        manual_seed = Mock()
        with patch.dict(sys.modules, {"torch": SimpleNamespace(manual_seed=manual_seed)}):
            runner.inicializar_aleatoriedade(77)
            first = random.random()
            runner.inicializar_aleatoriedade(77)
            self.assertEqual(first, random.random())
        self.assertEqual(manual_seed.call_count, 2)
        manual_seed.assert_called_with(77)

    def test_limite_nao_pega_so_o_inicio_abraom(self):
        plan = pd.DataFrame({"fonte": ["abraom"] * 400 + ["global"] * 600,
                             "variant_id": range(1000)})
        sample = runner.amostrar_preservando_a_mistura(plan, quantos=160, seed=7)
        self.assertEqual(sample.fonte.value_counts().to_dict(), {"global": 96, "abraom": 64})
        self.assertEqual(sample.variant_id.nunique(), 160)
        pd.testing.assert_frame_equal(sample, runner.amostrar_preservando_a_mistura(
            plan, quantos=160, seed=7))


    def test_selecao_guarda_a_melhor_e_nao_a_ultima(self):
        # piloto 5: a validacao tocou o fundo no passo 89 e degradou ate ficar pior que nao treinar.
        melhor = None
        for passo, valor in ((29, 1.7204), (59, 1.7180), (89, 1.7079), (119, 1.7435), (299, 1.9991)):
            melhor, _ = runner.atualizar_melhor(melhor, passo, {"criterio_primario": {"valor": valor}})
        self.assertEqual(melhor["passo"], 89)
        self.assertAlmostEqual(melhor["valor"], 1.7079)

    def test_selecao_aceita_a_avaliacao_final_fora_da_cadencia(self):
        melhor = None
        for passo, valor in ((29, 1.75), (59, 1.74)):
            melhor, _ = runner.atualizar_melhor(melhor, passo, {"criterio_primario": {"valor": valor}})
        melhor, trocou = runner.atualizar_melhor(melhor, 61, {"criterio_primario": {"valor": 1.70}})
        self.assertTrue(trocou)
        self.assertEqual(melhor["passo"], 61)

    def test_selecao_nao_troca_em_empate(self):
        melhor, _ = runner.atualizar_melhor(None, 10, {"criterio_primario": {"valor": 1.5}})
        melhor, trocou = runner.atualizar_melhor(melhor, 20, {"criterio_primario": {"valor": 1.5}})
        self.assertFalse(trocou)
        self.assertEqual(melhor["passo"], 10)


if __name__ == "__main__":
    unittest.main()
