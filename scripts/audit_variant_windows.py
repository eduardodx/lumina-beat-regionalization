#!/usr/bin/env python3
"""Audita as janelas de sequencia da campanha: REF contra o FASTA, bordas e bases fora de ACGT.

Roda no notebook (pandas + pyarrow + pysam/pyfaidx; sem GPU). So le; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.2.

POR QUE EXISTE
--------------
Antes de gastar GPU com extracao, e preciso saber quantas variantes do conjunto sequer produzem janela valida, e
por que. Sao tres motivos, com politicas diferentes:

  `ref_mismatch`  a base do FASTA na posicao focal nao e o REF declarado. O release do Mosaic conferiu o REF
                  contra o GRCh38.p14, entao isto significa FASTA ou build errado -- e ERRO, nao estatistica:
                  o script sai com codigo 2.
  `non_acgt`      a janela tem base fora de ACGT (N). A politica declarada e DESCARTAR a variante; o soft-mask
                  (minusculas) e normalizado antes, entao nao conta aqui.
  `out_of_bounds` a janela nao cabe no cromossomo. Descartada tambem -- deslocar tiraria a variante do indice
                  focal declarado.

O numero que interessa para a declaracao e quantas variantes cada motivo tira, por papel e por painel.

COMO
----
Usa `eval/embedding_probe/windows.py` (portado da pesquisa), que implementa a convencao do Mosaic: offset focal
`L // 2 - 1`, validacao estrita sem fallback e nenhuma janela deslocada.

**A janela auditada e a janela declarada.** Se a tabela traz `focal_index` (plano do adapter, secao 5.1), e ELE
que decide onde a janela comeca; sem a coluna, vale o offset centrado do Mosaic (snapshots da cabeca). Auditar a
janela centrada de um plano deslocado e pior que nao auditar: a janela planejada pode ter `N` e o relatorio dizer
`ok`, porque as duas janelas cobrem trechos diferentes do cromossomo.

O QUE NAO PROVA
---------------
- Nao valida o conteudo biologico da janela, so o contrato: REF confere, cabe, e e ACGT.
- Nao decide o tamanho da janela: ele e declarado (`--window-bp`).
- Nao valida a RECEITA do plano (mistura, amostragem, comprimento dos spans), so a coerencia geometrica da linha.

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/audit_variant_windows.py \
        --variants ~/artifacts/redesenho/g2_final_janela4096/core_head_snapshot.parquet \
        --fasta ~/hg38/hg38.fa --window-bp 4096 \
        --out-dir ~/artifacts/redesenho/g3_janelas/treino4096
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedding_probe.windows import WindowError, build_window, focal_offset  # noqa: E402

COLUNAS = ("variant_id", "chrom", "pos_1based", "ref", "alt")

# Vocabulario do plano do adapter (`build_adapter_window_plan.py`). Opcionais: sem eles, a janela e a centrada.
COLUNA_FOCAL = "focal_index"
COLUNA_INICIO = "window_start"
COLUNA_SPANS = "spans"
TIPO_VARIANTE = "variante"

# Motivos proprios da auditoria do plano. Sao ERRO, nao estatistica: o plano discorda de si mesmo, entao a janela
# que o treinador construiria nao e a que o gerador declarou.
MOTIVO_INICIO_INCOERENTE = "window_start_incoerente"
MOTIVO_SPAN_FORA = "span_fora_da_janela"
MOTIVO_SPAN_FOCAL = "span_de_variante_nao_cobre_o_focal"
MOTIVOS_FATAIS = ("ref_mismatch", MOTIVO_INICIO_INCOERENTE, MOTIVO_SPAN_FORA, MOTIVO_SPAN_FOCAL)


def _opcional(row: Any, coluna: str) -> Any:
    """Valor da coluna, ou None se ausente/nulo. A tabela da cabeca nao tem as colunas do plano."""
    valor = getattr(row, coluna, None)
    if valor is None:
        return None
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    return valor


def focal_planejado(row: Any) -> int | None:
    """Indice focal declarado na linha, ou None (janela centrada)."""
    valor = _opcional(row, COLUNA_FOCAL)
    return None if valor is None else int(valor)


def checar_inicio(row: Any, focal_index: int) -> str | None:
    """`window_start` tem de ser `pos - 1 - focal`, senao a janela auditada nao e a planejada."""
    declarado = _opcional(row, COLUNA_INICIO)
    if declarado is None:
        return None
    esperado = int(row.pos_1based) - 1 - focal_index
    if int(declarado) != esperado:
        return f"window_start={int(declarado)} != pos-1-focal={esperado}"
    return None


def checar_spans(row: Any, *, focal_index: int, window_bp: int) -> tuple[str | None, str]:
    """Os spans cabem na janela e exatamente um -- o de variante -- cobre o focal."""
    bruto = _opcional(row, COLUNA_SPANS)
    if bruto is None:
        return None, ""
    spans = json.loads(bruto) if isinstance(bruto, str) else [list(s) for s in bruto]
    cobrindo = []
    for inicio, fim, tipo in spans:
        inicio, fim = int(inicio), int(fim)
        if not 0 <= inicio < fim <= window_bp:
            return MOTIVO_SPAN_FORA, f"span [{inicio},{fim}) fora de [0,{window_bp})"
        if inicio <= focal_index < fim:
            cobrindo.append((inicio, fim, str(tipo)))
    if len(cobrindo) != 1:
        return MOTIVO_SPAN_FOCAL, f"{len(cobrindo)} spans cobrem o focal {focal_index}, esperado 1"
    if cobrindo[0][2] != TIPO_VARIANTE:
        return MOTIVO_SPAN_FOCAL, f"quem cobre o focal e {cobrindo[0][2]!r}, esperado {TIPO_VARIANTE!r}"
    return None, ""


def audit_windows(
    rows: pd.DataFrame, fetch: Callable[[str, int, int], str], *, window_bp: int
) -> tuple[Counter, list[dict[str, Any]]]:
    """Constroi a janela DECLARADA de cada linha e agrega por motivo. Nao levanta: coleta."""
    motivos: Counter = Counter()
    falhas: list[dict[str, Any]] = []

    def registrar(row: Any, motivo: str, detalhe: str) -> None:
        motivos[motivo] += 1
        falhas.append({
            "variant_id": str(row.variant_id), "chrom": str(row.chrom),
            "pos_1based": int(row.pos_1based), "ref": str(row.ref), "alt": str(row.alt),
            "motivo": motivo, "detalhe": detalhe,
            **({"focal_index": focal_planejado(row)} if _opcional(row, COLUNA_FOCAL) is not None else {}),
            **({"role": row.role} if hasattr(row, "role") else {}),
            **({"primary_panel": row.primary_panel} if hasattr(row, "primary_panel") else {}),
            **({"binary_label": int(row.binary_label)} if hasattr(row, "binary_label") else {}),
        })

    for row in rows.itertuples(index=False):
        focal = focal_planejado(row)
        if focal is not None:
            detalhe = checar_inicio(row, focal)
            if detalhe:
                registrar(row, MOTIVO_INICIO_INCOERENTE, detalhe)
                continue
        try:
            janela = build_window(fetch, chrom=str(row.chrom), pos_1based=int(row.pos_1based),
                                  ref=str(row.ref), alt=str(row.alt), window_bp=window_bp,
                                  focal_index=focal)
        except WindowError as exc:
            registrar(row, exc.reason, exc.detail)
            continue
        motivo, detalhe = checar_spans(row, focal_index=janela.focal_index, window_bp=window_bp)
        if motivo:
            registrar(row, motivo, detalhe)
            continue
        motivos["ok"] += 1
    return motivos, falhas


def descrever_layout(rows: pd.DataFrame, *, window_bp: int) -> dict[str, Any]:
    """Diz, no relatorio, QUAL janela foi auditada -- sem isso um `ok` e ambiguo."""
    if COLUNA_FOCAL not in rows.columns:
        return {"layout": "centrado", "focal_index": focal_offset(window_bp),
                "origem": "convencao do Mosaic (L // 2 - 1); a tabela nao declara focal_index"}
    focais = rows[COLUNA_FOCAL].dropna().astype(int)
    return {
        "layout": "declarado pelo plano",
        "origem": f"coluna {COLUNA_FOCAL} da tabela auditada",
        "focal_index": {"min": int(focais.min()), "max": int(focais.max()),
                        "distintos": int(focais.nunique()), "centro_da_janela": focal_offset(window_bp)},
        "window_start_conferido": COLUNA_INICIO in rows.columns,
        "spans_conferidos": COLUNA_SPANS in rows.columns,
    }


def breakdown(falhas: list[dict[str, Any]], coluna: str) -> dict[str, dict[str, int]]:
    """Falhas por motivo e por uma coluna de contexto (papel, painel, classe)."""
    out: dict[str, dict[str, int]] = {}
    for falha in falhas:
        if coluna not in falha:
            continue
        out.setdefault(falha["motivo"], {})
        chave = str(falha[coluna])
        out[falha["motivo"]][chave] = out[falha["motivo"]].get(chave, 0) + 1
    return out


def abrir_fasta(caminho: Path):
    """Devolve um `fetch(chrom, start0, end0)`; aceita pysam ou pyfaidx, nesta ordem."""
    try:
        import pysam

        handle = pysam.FastaFile(str(caminho))
        return lambda chrom, start, end: handle.fetch(chrom, start, end), "pysam"
    except ImportError:
        pass
    try:
        from pyfaidx import Fasta

        handle = Fasta(str(caminho), as_raw=True, sequence_always_upper=False)
        return lambda chrom, start, end: str(handle[chrom][start:end]), "pyfaidx"
    except ImportError as exc:
        raise SystemExit(f"nem pysam nem pyfaidx disponiveis para ler {caminho}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--window-bp", type=int, default=4096)
    parser.add_argument("--limit", type=int, help="audita so as N primeiras linhas (sondagem rapida)")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    rows = pd.read_parquet(args.variants.expanduser())
    faltando = [c for c in COLUNAS if c not in rows.columns]
    if faltando:
        print(f"FALHOU: faltam colunas {faltando} em {args.variants}")
        return 2
    if args.limit:
        rows = rows.head(args.limit)

    fetch, leitor = abrir_fasta(args.fasta.expanduser())
    motivos, falhas = audit_windows(rows, fetch, window_bp=args.window_bp)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if falhas:
        pd.DataFrame(falhas).to_parquet(out_dir / "variantes_sem_janela.parquet", index=False)

    total = int(len(rows))
    relatorio: dict[str, Any] = {
        "entradas": {"variantes": str(args.variants), "fasta": str(args.fasta), "leitor": leitor,
                     "window_bp": args.window_bp},
        "janela_auditada": descrever_layout(rows, window_bp=args.window_bp),
        "total": total,
        "por_motivo": dict(sorted(motivos.items())),
        "fracao_descartada": round((total - motivos["ok"]) / total, 6) if total else None,
        "por_papel": breakdown(falhas, "role"),
        "por_painel": breakdown(falhas, "primary_panel"),
        "por_classe": breakdown(falhas, "binary_label"),
        "politica": {
            "non_acgt": "variante descartada (soft-mask normalizado antes; N nao e substituido)",
            "out_of_bounds": "variante descartada; janela nunca e deslocada",
            "ref_mismatch": "ERRO: o release conferiu REF contra GRCh38.p14, entao isto e FASTA ou build errado",
            "window_start_incoerente": "ERRO: o plano declara um inicio != pos-1-focal; a janela do treino nao seria a planejada",
            "span_fora_da_janela": "ERRO: span fora de [0, window_bp)",
            "span_de_variante_nao_cobre_o_focal": "ERRO: a receita exige um unico span cobrindo o focal",
        },
        "o_que_nao_prova": ["nao valida o conteudo biologico da janela, so o contrato"],
    }
    (out_dir / "auditoria_de_janelas.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("total", "por_motivo", "fracao_descartada", "por_painel")},
                     ensure_ascii=False, indent=2))

    fatais = {motivo: int(motivos[motivo]) for motivo in MOTIVOS_FATAIS if motivos.get(motivo)}
    if fatais:
        print(f"\nFALHOU: {fatais} -- erro de dado ou de plano, nao estatistica de descarte.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
