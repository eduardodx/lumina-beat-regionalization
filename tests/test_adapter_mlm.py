"""Prova a contabilidade do MLM do adapter: quem e alvo, de que categoria, e como as perdas se combinam.

O teste que mais importa e o do alvo focal: o span que cobre a variante tambem cobre bases de referencia, entao
"loss do span da variante" NAO e "loss do alelo variante". Se as duas se misturarem, a medida da campanha dilui.
    PYTHONPATH=. python3 tests/test_adapter_mlm.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.adapter import mlm  # noqa: E402


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def test_vocabulario_bate_com_o_do_modelo():
    """O modulo redeclara o vocabulario para rodar sem torch. Se o do modelo mudar, isto tem de quebrar."""
    caminho = Path(__file__).resolve().parents[1] / "lumina" / "constants.py"
    spec = importlib.util.spec_from_file_location("_constants_do_modelo", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)  # constants.py nao importa torch; o __init__ do pacote e que importa
    assert mlm.DNA_VOCAB == modulo.DNA_VOCAB, (mlm.DNA_VOCAB, modulo.DNA_VOCAB)
    assert mlm.MASK_ID == modulo.MASK_ID
    assert mlm.VOCAB_SIZE == modulo.VOCAB_SIZE


def test_codificar_usa_os_ids_do_treino():
    assert mlm.codificar("ACGTN") == (1, 2, 3, 4, 5)
    try:
        mlm.codificar("ACGX")
    except ValueError:
        return
    raise AssertionError("base fora do vocabulario tinha de falhar")


def _exemplo_simples():
    # Janela de 20 bp. Focal em 10, ALT = G (a referencia ali era A).
    alt_seq = "A" * 10 + "G" + "A" * 9
    spans = [(9, 12, mlm.TIPO_VARIANTE), (2, 5, mlm.TIPO_REFERENCIA)]
    return alt_seq, spans


def test_alvo_do_focal_e_o_alt_e_nao_a_referencia():
    """O caso que a revisao de 21/09 separou: dentro do span da variante so UMA posicao carrega o ALT."""
    alt_seq, spans = _exemplo_simples()
    exemplo = mlm.montar_exemplo(alt_seq, spans, variant_id="v1", fonte="abraom", focal_index=10)
    por_posicao = dict(zip(exemplo.posicoes, zip(exemplo.alvos, exemplo.categorias)))

    assert por_posicao[10] == (mlm.DNA_VOCAB["G"], mlm.CATEGORIA_FOCAL), por_posicao[10]
    # 9 e 11 estao DENTRO do span da variante, mas carregam base de referencia.
    assert por_posicao[9] == (mlm.DNA_VOCAB["A"], mlm.CATEGORIA_CONTEXTO)
    assert por_posicao[11] == (mlm.DNA_VOCAB["A"], mlm.CATEGORIA_CONTEXTO)
    for posicao in (2, 3, 4):
        assert por_posicao[posicao] == (mlm.DNA_VOCAB["A"], mlm.CATEGORIA_REFERENCIA)


def test_uma_unica_posicao_e_focal():
    alt_seq, spans = _exemplo_simples()
    exemplo = mlm.montar_exemplo(alt_seq, spans, variant_id="v1", fonte="abraom", focal_index=10)
    assert exemplo.categorias.count(mlm.CATEGORIA_FOCAL) == 1
    assert len(exemplo.por_categoria()[mlm.CATEGORIA_FOCAL]) == 1


def test_mascara_so_o_que_e_alvo():
    alt_seq, spans = _exemplo_simples()
    exemplo = mlm.montar_exemplo(alt_seq, spans, variant_id="v1", fonte="abraom", focal_index=10)
    mascaradas = set(exemplo.posicoes)
    assert mascaradas == {2, 3, 4, 9, 10, 11}
    for posicao, token in enumerate(exemplo.input_ids):
        if posicao in mascaradas:
            assert token == mlm.MASK_ID, posicao
        else:
            assert token == mlm.DNA_VOCAB[alt_seq[posicao]], posicao


def test_span_de_referencia_que_cobre_o_focal_nao_o_torna_focal():
    """A categoria vem do TIPO do span, nao da coincidencia de posicao: um span de referencia sobre o focal e
    plano invalido, e o auditor de janelas ja reprova. Aqui o contrato e o outro lado: sem span de variante
    cobrindo o focal, montar_exemplo falha."""
    alt_seq = "A" * 10 + "G" + "A" * 9
    try:
        mlm.montar_exemplo(alt_seq, [(9, 12, mlm.TIPO_REFERENCIA)], variant_id="v", fonte="f", focal_index=10)
    except ValueError as exc:
        assert "cobrem o focal" in str(exc), str(exc)
        return
    raise AssertionError("sem span de variante no focal, tinha de falhar")


def test_spans_sobrepostos_falham():
    alt_seq = "A" * 20
    try:
        mlm.montar_exemplo(alt_seq, [(9, 12, mlm.TIPO_VARIANTE), (11, 14, mlm.TIPO_REFERENCIA)],
                           variant_id="v", fonte="f", focal_index=10)
    except ValueError as exc:
        assert "mais de um span" in str(exc)
        return
    raise AssertionError("spans sobrepostos tinham de falhar")


def test_span_fora_da_janela_falha():
    try:
        mlm.montar_exemplo("A" * 20, [(18, 25, mlm.TIPO_VARIANTE)], variant_id="v", fonte="f",
                           focal_index=19)
    except ValueError as exc:
        assert "fora de" in str(exc)
        return
    raise AssertionError("span fora da janela tinha de falhar")


def test_spans_do_plano_leem_o_json_do_gerador():
    bruto = json.dumps([[9, 12, "variante"], [2, 5, "referencia"]])
    assert mlm.spans_do_plano(bruto) == [(9, 12, "variante"), (2, 5, "referencia")]
    assert mlm.spans_do_plano([[1, 2, "variante"]]) == [(1, 2, "variante")]


def test_decompor_separa_as_tres_categorias():
    perdas = [1.0, 2.0, 4.0, 0.0]
    categorias = [mlm.CATEGORIA_FOCAL, mlm.CATEGORIA_CONTEXTO, mlm.CATEGORIA_CONTEXTO,
                  mlm.CATEGORIA_REFERENCIA]
    fora = mlm.decompor_perdas(perdas, categorias)
    assert fora[mlm.CATEGORIA_FOCAL] == {"media": 1.0, "posicoes": 1}
    assert fora[mlm.CATEGORIA_CONTEXTO] == {"media": 3.0, "posicoes": 2}
    assert fora[mlm.CATEGORIA_REFERENCIA] == {"media": 0.0, "posicoes": 1}


def test_categoria_vazia_nao_quebra_a_decomposicao():
    fora = mlm.decompor_perdas([1.0], [mlm.CATEGORIA_FOCAL])
    assert fora[mlm.CATEGORIA_FOCAL]["posicoes"] == 1
    assert math.isnan(fora[mlm.CATEGORIA_REFERENCIA]["media"])
    assert fora[mlm.CATEGORIA_REFERENCIA]["posicoes"] == 0


def test_perda_ponderada_pesa_por_posicao():
    """Com pesos iguais, a ponderada tem de dar a media simples das posicoes -- nao a media das categorias."""
    por_categoria = {
        mlm.CATEGORIA_FOCAL: {"media": 4.0, "posicoes": 1},
        mlm.CATEGORIA_CONTEXTO: {"media": 1.0, "posicoes": 2},
        mlm.CATEGORIA_REFERENCIA: {"media": 1.0, "posicoes": 9},
    }
    iguais = {c: 1.0 for c in mlm.CATEGORIAS}
    esperado = (4.0 * 1 + 1.0 * 2 + 1.0 * 9) / 12
    assert abs(mlm.perda_ponderada(por_categoria, iguais) - esperado) < 1e-12

    so_focal = {mlm.CATEGORIA_FOCAL: 1.0, mlm.CATEGORIA_CONTEXTO: 0.0, mlm.CATEGORIA_REFERENCIA: 0.0}
    assert abs(mlm.perda_ponderada(por_categoria, so_focal) - 4.0) < 1e-12


def test_peso_faltando_falha_em_vez_de_assumir():
    por_categoria = {c: {"media": 1.0, "posicoes": 1} for c in mlm.CATEGORIAS}
    try:
        mlm.perda_ponderada(por_categoria, {mlm.CATEGORIA_FOCAL: 1.0})
    except ValueError as exc:
        assert "pesos faltando" in str(exc)
        return
    raise AssertionError("peso ausente tinha de falhar, nao virar zero")


def test_a_armadilha_da_varredura_de_pesos():
    """DUAS configuracoes de peso sobre AS MESMAS perdas dao losses de treino diferentes e criterio IGUAL.

    E por isso que a comparacao entre configuracoes tem de usar a decomposicao NAO ponderada: comparar pela loss
    de treino de cada uma elegeria a de pesos mais frouxos, nao a que aprendeu mais.
    """
    perdas = [2.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    categorias = ([mlm.CATEGORIA_FOCAL] + [mlm.CATEGORIA_CONTEXTO] * 2 + [mlm.CATEGORIA_REFERENCIA] * 3)
    decomposicao = mlm.decompor_perdas(perdas, categorias)

    frouxo = {mlm.CATEGORIA_FOCAL: 1.0, mlm.CATEGORIA_CONTEXTO: 0.1, mlm.CATEGORIA_REFERENCIA: 0.1}
    apertado = {mlm.CATEGORIA_FOCAL: 1.0, mlm.CATEGORIA_CONTEXTO: 1.0, mlm.CATEGORIA_REFERENCIA: 1.0}
    assert mlm.perda_ponderada(decomposicao, frouxo) != mlm.perda_ponderada(decomposicao, apertado)

    # O criterio primario nao se move com o peso: e a mesma perda no focal nas duas.
    assert decomposicao[mlm.CRITERIO_PRIMARIO]["media"] == 2.0
    assert mlm.CRITERIO_PRIMARIO == mlm.CATEGORIA_FOCAL


def test_agregar_pondera_pela_contagem_de_posicoes():
    um = {mlm.CATEGORIA_FOCAL: {"media": 1.0, "posicoes": 1},
          mlm.CATEGORIA_CONTEXTO: {"media": 0.0, "posicoes": 0},
          mlm.CATEGORIA_REFERENCIA: {"media": 2.0, "posicoes": 10}}
    outro = {mlm.CATEGORIA_FOCAL: {"media": 3.0, "posicoes": 1},
             mlm.CATEGORIA_CONTEXTO: {"media": 0.0, "posicoes": 0},
             mlm.CATEGORIA_REFERENCIA: {"media": 4.0, "posicoes": 10}}
    junto = mlm.agregar([um, outro])
    assert junto[mlm.CATEGORIA_FOCAL] == {"media": 2.0, "posicoes": 2}
    assert junto[mlm.CATEGORIA_REFERENCIA] == {"media": 3.0, "posicoes": 20}
    assert junto[mlm.CATEGORIA_CONTEXTO]["posicoes"] == 0


def test_pesos_iniciais_sao_declarados_para_as_tres():
    assert set(mlm.PESOS_INICIAIS) == set(mlm.CATEGORIAS)
    assert all(v >= 0 for v in mlm.PESOS_INICIAIS.values())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed, skipped = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    ran = len(tests) - failed - len(skipped)
    tail = f"  |  {len(skipped)} PULADO(S), sem cobertura: {', '.join(skipped)}" if skipped else ""
    print(f"\n{ran}/{len(tests) - len(skipped)} passaram{tail}")
    if skipped and os.environ.get("REQUIRE_NO_SKIP"):
        print("REQUIRE_NO_SKIP: teste pulado conta como falha")
        sys.exit(1)
    sys.exit(1 if failed else 0)
