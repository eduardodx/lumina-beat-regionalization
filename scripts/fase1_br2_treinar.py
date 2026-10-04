#!/usr/bin/env python3
"""Fase 1, BR v2: treina o braco E+F+BR2 (desenvolvimento exploratorio, posterior ao teste da Fase 1).

Especificacao, escrita antes de rodar: docs/fase1_br2_especificacao.md. Tudo igual ao passo 3
(scripts/fase1_bracos.py): release, embeddings, cinco execucoes, purgas, linhas, padronizacao no treino, regressao
logistica L2, grade de C, escolha pela macro AUROC da validation e regra de convergencia. Muda so o bloco ABraOM:

- estados `not_found`, `no_call`, `ac0`;
- faixas de AC das presentes: 1, 2, 3 a 9, 10 ou mais (particao de `present`);
- AF pontual = AC/AN e o limite inferior unilateral de 95% de Clopper-Pearson de AC/AN
  (`mosaic.regional_truth.af_lower_bound`), ambos em log10(x + 1e-6);
- qualidade: FILTER PASS, AN/2.342 e AN >= 80% de 2.342;
- excesso sustentado sobre o AF_joint do gnomAD, so com presenca, PASS e AN suficiente.

Antes de treinar confere AC/AN/AF por estado e interrompe em contradicao estrutural; a diferenca entre a AF publicada
e AC/AN so e relatada. E+F e E+F+BR nao sao retreinados: as leituras usam as predicoes do passo 3.

USO (notebook, no .venv do Mosaic)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_br2_treinar.py \\
        --entrega ~/mosaic-v1-2026-09-30 --bracos ~/artifacts/mosaic_v1/bracos_<...> \\
        --cache ~/artifacts/redesenho/g3_cache/M0 --cache ~/artifacts/redesenho/g7_cache/M0 \\
        --cache ~/artifacts/mosaic_v1/cache_M0_complemento --out-dir ~/artifacts/mosaic_v1/br2_<...>

SAIDAS (em --out-dir, que nao pode existir; gravadas em <out-dir>.tmp e renomeadas so depois do contrato):
fase1-e-f-br2/{predictions.parquet,system.yaml}, modelos/, selecao.json, contagens.json e fontes.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from scripts.inventario_mosaic_v1 import K  # noqa: E402

FalhaDaFase1 = bracos.FalhaDaFase1
BRACO, ID, BLOCOS = "E+F+BR2", "fase1-e-f-br2", ("e", "f", "br2")
ESPECIFICACAO = RAIZ / "docs" / "fase1_br2_especificacao.md"
AN_MAXIMO = bracos.AN_MAXIMO_DO_ABRAOM
AN_SUFICIENTE = 0.8 * AN_MAXIMO
#: (menor AC, maior AC ou None, nome) das faixas das presentes.
BANDAS = ((1, 1, "ac1"), (2, 2, "ac2"), (3, 9, "ac3a9"), (10, None, "ac10mais"))
ESTADOS = ("present", "ac0", "no_call", "not_found")


def carregar_release(raiz: Path) -> pd.DataFrame:
    """O release do passo 3 mais `abraom_ac`."""
    release = bracos.carregar_release(raiz)
    ac = pd.read_parquet(raiz / "variant-annotations.parquet", columns=["variant_id", "abraom_ac"])
    return release.merge(ac, on="variant_id", how="left", validate="one_to_one")


def conferir_contagens(frame: pd.DataFrame) -> dict[str, Any]:
    """Coerencia de AC, AN e AF com o estado de cada variante. Contradicao estrutural interrompe; o resto e relatado."""
    estado = frame["abraom_status"].fillna("<nulo>").astype(str)
    ac, an = pd.to_numeric(frame["abraom_ac"]), pd.to_numeric(frame["abraom_an"])
    af = pd.to_numeric(frame["abraom_af"])
    presente, zero = estado == "present", estado == "ac0"
    regras = {
        "present sem AC >= 1": presente & ~(ac >= 1),
        "present sem AN >= 1": presente & ~(an >= 1),
        "present com AC > AN": presente & (ac > an),
        f"AN acima de {AN_MAXIMO}": an > AN_MAXIMO,
        "ac0 com AC diferente de 0": zero & ~(ac == 0),
        "ac0 sem AN >= 1": zero & ~(an >= 1),
        "ac0 com AF diferente de 0": zero & ~(af == 0),
        "no_call com AN > 0": (estado == "no_call") & (an > 0),
        "not_found com AC, AN ou AF": (estado == "not_found") & (ac.notna() | an.notna() | af.notna()),
    }
    violacoes = {nome: int(mascara.sum()) for nome, mascara in regras.items() if int(mascara.sum())}
    estranhos = sorted(set(estado) - set(ESTADOS) - {"<nulo>"})
    if violacoes or estranhos:
        raise FalhaDaFase1(f"ABraOM incoerente: {violacoes or ''} {('estados ' + str(estranhos)) if estranhos else ''}")
    razao = (ac / an)[presente]
    diferenca = (af[presente] - razao).abs()
    relativa = diferenca / razao
    faixas = {nome: int((presente & (ac >= baixo) & (ac <= (alto if alto is not None else np.inf))).sum())
              for baixo, alto, nome in BANDAS}
    return {"por_estado": {k: int(v) for k, v in estado.value_counts().sort_index().items()},
            "faixas_de_ac_das_presentes": faixas,
            "an_das_presentes": {"min": float(an[presente].min()) if presente.any() else None,
                                 "max": float(an[presente].max()) if presente.any() else None,
                                 "fracao_suficiente": float((an[presente] >= AN_SUFICIENTE).mean()) if presente.any()
                                 else None},
            "af_publicada_contra_ac_an": {"maior_diferenca_absoluta": float(diferenca.max()) if len(diferenca) else 0.0,
                                         "maior_diferenca_relativa": float(relativa.max()) if len(relativa) else 0.0,
                                         "acima_de_1pct": int((relativa > 0.01).sum())},
            "filtro_das_presentes": {str(k): int(v) for k, v in
                                     frame.loc[presente, "abraom_filter"].fillna("<nulo>").value_counts().items()}}


def bloco_br2(frame: pd.DataFrame, limite_inferior: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> pd.DataFrame:
    """As 13 colunas do BR v2 (docs/fase1_br2_especificacao.md). `limite_inferior` e o `af_lower_bound` do Mosaic."""
    estado = frame["abraom_status"].fillna("").astype(str).to_numpy()
    presente = estado == "present"
    ac = pd.to_numeric(frame["abraom_ac"]).fillna(0).to_numpy(dtype=float)
    an = pd.to_numeric(frame["abraom_an"]).fillna(0).to_numpy(dtype=float)
    an_seguro = np.where(an > 0, an, 1.0)
    af = np.where(presente, ac / an_seguro, 0.0)
    inferior = np.where(presente, np.asarray(limite_inferior(np.where(presente, ac, 0.0), an_seguro), dtype=float), 0.0)
    passou = (frame["abraom_filter"].fillna("").astype(str) == "PASS").to_numpy()
    suficiente = an >= AN_SUFICIENTE
    estado_gnomad = frame["gnomad_status"].fillna("unknown")
    af_joint = (frame["gnomad_v4_af"].where(~estado_gnomad.isin(["not_found", "ac0"]), 0).fillna(0).clip(lower=0)
                .to_numpy(dtype=float))
    log_inferior = np.log10(inferior + 1e-6)
    colunas: dict[str, np.ndarray] = {"br2_not_found": estado == "not_found", "br2_no_call": estado == "no_call",
                                      "br2_ac0": estado == "ac0"}
    for baixo, alto, nome in BANDAS:
        colunas[f"br2_{nome}"] = presente & (ac >= baixo) & (ac <= (alto if alto is not None else np.inf))
    colunas.update({
        "br2_log_af": np.log10(af + 1e-6), "br2_log_af_inferior": log_inferior, "br2_pass": passou,
        "br2_an_frac": an / AN_MAXIMO, "br2_an_suficiente": suficiente,
        "br2_excesso_sustentado": np.where(presente & passou & suficiente,
                                           np.clip(log_inferior - np.log10(af_joint + 1e-6), 0, None), 0.0)})
    saida = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in colunas.items()}, index=frame.index)
    if not np.isfinite(saida.to_numpy()).all():
        raise FalhaDaFase1("bloco BR2 com valor nao finito")
    if (inferior > af + 1e-12).any():
        raise FalhaDaFase1("limite inferior acima da AF pontual: o af_lower_bound nao e o esperado")
    return saida


def ferramentas_do_mosaic() -> dict[str, Callable[..., Any]]:
    """As do passo 3 mais o limite inferior do Mosaic; os testes substituem esta funcao."""
    from mosaic.regional_truth import af_lower_bound

    return {**bracos.ferramentas_do_mosaic(), "af_lower_bound": af_lower_bound}


def sistema_br2(proveniencia: dict[str, Any]) -> dict[str, Any]:
    conteudo = bracos.sistema("E+F+BR", proveniencia)
    conteudo["id"] = ID
    conteudo["exposure"]["population_frequency"]["sources"] = [
        "gnomAD v4.1 joint: bloco de frequencia oficial do Mosaic (frequency_features)",
        "ABraOM WGS-1171 (variant-annotations), BR v2: estados, faixas de AC, AF = AC/AN, limite inferior de "
        "Clopper-Pearson, FILTER PASS, AN e excesso sustentado sobre o AF_joint"]
    conteudo["fase1"].update({"braco": BRACO, "blocos": list(BLOCOS),
                              "natureza": "desenvolvimento exploratorio posterior ao teste da Fase 1",
                              "especificacao_sha256": proveniencia["especificacao_sha256"]})
    return conteudo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--bracos", required=True, type=Path, help="pasta do passo 3 (confere release e linhas)")
    parser.add_argument("--cache", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    entrega, pasta3, destino = args.entrega.expanduser(), args.bracos.expanduser(), args.out_dir.expanduser()
    temporaria = destino.with_name(destino.name + ".tmp")
    if destino.exists() or temporaria.exists():
        return bracos._falhar([f"{destino} ou {temporaria} ja existe; o BR v2 grava sempre numa pasta nova"])
    if not ESPECIFICACAO.is_file():
        return bracos._falhar([f"sem a especificacao {ESPECIFICACAO}: ela vem antes de rodar"])
    try:
        ferramentas = ferramentas_do_mosaic()
    except ImportError as exc:
        return bracos._falhar([f"sem o pacote mosaic ou o sklearn ({exc}); rodar no .venv do Mosaic (uv run --project)"])

    inicio = time.perf_counter()
    raiz = entrega / bracos.RELEASE
    try:
        identidade = ferramentas["verificar_identidade"](raiz, config_dir=entrega / "config")
        protocolo_sha256 = sha256_do_arquivo(entrega / bracos.PROTOCOLO_DE_ESTUDOS)
        bracos.conferir_entrega(identidade, protocolo_sha256)
        leituras.conferir_fontes(json.loads((pasta3 / "fontes.json").read_text(encoding="utf-8")), raiz)
        release = carregar_release(raiz)
        por_run, contagens = bracos.linhas_por_execucao(release)
        if bracos.CONTAGENS_DO_GUIA_RUN0 is not None and contagens[0]["guia"] != bracos.CONTAGENS_DO_GUIA_RUN0:
            raise FalhaDaFase1(f"run 0 nao bate com o guia de splits: {contagens[0]['guia']}")
        elegiveis = release[release["sequence_eligible"]].reset_index(drop=True)
        conferencia_f = bracos.conferir_frequencia_oficial(elegiveis, ferramentas["frequencia_oficial"])
        contagens_br = conferir_contagens(elegiveis)
        br2 = bloco_br2(elegiveis, ferramentas["af_lower_bound"])
        matriz_e, fontes_dos_caches = bracos.carregar_leitura([p.expanduser() for p in args.cache], raiz,
                                                              elegiveis["variant_id"].astype(str).to_numpy())
        dados = bracos.montar_dados(elegiveis, matriz_e, por_run)
        dados.blocos["br2"], dados.colunas["br2"] = br2.to_numpy(np.float64), list(br2.columns)
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return bracos._falhar([f"{type(exc).__name__}: {exc}"])
    print(f"ABraOM conferido: {contagens_br['por_estado']} | faixas de AC {contagens_br['faixas_de_ac_das_presentes']} | "
          f"AF publicada x AC/AN {contagens_br['af_publicada_contra_ac_an']}", flush=True)

    resultados = []
    try:
        for run in range(K):
            t0 = time.perf_counter()
            r = bracos.treinar_braco(dados, run, BLOCOS, bracos.AJUSTADOR)
            resultados.append(r)
            iteracoes = ", ".join(f"{g['C']:g}:{g.get('n_iter')}" for g in r["grade"])
            print(f"[{BRACO} run {run}] C={r['C']:g} macro_validacao={r['macro_validacao']:.4f} (iteracoes {iteracoes}; "
                  f"{time.perf_counter() - t0:.1f} s)", flush=True)
        preds = bracos.predicoes(dados, resultados)
        leituras.conferir_linhas({BRACO: preds}, release)   # as mesmas linhas de validation e teste do passo 3
    except (FalhaDaFase1, ValueError) as exc:
        return bracos._falhar([f"{type(exc).__name__}: {exc}"])

    primeira = next(iter(fontes_dos_caches.values()))["identidade"]
    proveniencia = {
        "entrega": str(entrega), "passo3": str(pasta3),
        "identidade_do_release": {k: identidade.get(k) for k in ("release_id", "version", "release_identity_hash",
                                                                  "protocol_hash")},
        "study_protocol_sha256": protocolo_sha256,
        "arquivos_do_release": {nome: sha256_do_arquivo(raiz / nome) for nome in
                                ("clinical-variants.parquet", bracos.VISTA, "evaluation-panels.parquet",
                                 "variant-annotations.parquet")},
        "conferencia_do_bloco_f": conferencia_f, "caches": fontes_dos_caches,
        "r03": {"extracao": bracos.EXTRACAO, "checkpoint_sha256": primeira.get("checkpoint_sha256"),
                "versao_do_extrator": primeira.get("versao_do_extrator"), "janela_bp": primeira.get("janela_bp")},
        "especificacao_sha256": sha256_do_arquivo(ESPECIFICACAO), "script_sha256": sha256_do_arquivo(Path(__file__)),
        "revisao": bracos.revisao_do_repositorio(), "versoes": bracos.versoes(), "receita": bracos.RECEITA,
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    temporaria.mkdir(parents=True)
    try:
        (temporaria / "modelos").mkdir()
        pasta = temporaria / ID
        pasta.mkdir()
        preds.to_parquet(pasta / "predictions.parquet", index=False)
        bracos.escrever_sistema(pasta, sistema_br2(proveniencia))
        selecao = []
        for run, r in enumerate(resultados):
            np.savez(temporaria / "modelos" / f"{ID}_run{run}.npz", coef=r["coef"],
                     intercepto=np.float64(r["intercepto"]), media=r["media"], desvio=r["desvio"],
                     C=np.float64(r["C"]), colunas=np.array(r["colunas"]), braco=np.array(BRACO), run=np.int64(run))
            selecao.append({"run": run, "C": r["C"], "macro_validacao": r["macro_validacao"], "grade": r["grade"]})
        (temporaria / "selecao.json").write_text(json.dumps({"receita": bracos.RECEITA, "bracos": {BRACO: selecao}},
                                                            indent=2, ensure_ascii=False), encoding="utf-8")
        (temporaria / "contagens.json").write_text(json.dumps({"abraom": contagens_br, "execucoes": contagens,
                                                               "colunas_br2": list(br2.columns)}, indent=2,
                                                              ensure_ascii=False), encoding="utf-8")
        (temporaria / "fontes.json").write_text(json.dumps(proveniencia, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
        n = bracos.conferir_contrato(ferramentas, pasta, release, identidade, protocolo_sha256)
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return bracos._falhar([f"{type(exc).__name__}: {exc} (saida parcial deixada em {temporaria} para inspecao)"])
    temporaria.rename(destino)
    print(f"\nmacro da validation de {BRACO} (a mesma usada para escolher C): "
          + "  ".join(f"run{i} {r['macro_validacao']:.4f} (C={r['C']:g})" for i, r in enumerate(resultados)))
    print(f"PASSOU: {BRACO} no contrato do Mosaic ({n} linhas) | {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
