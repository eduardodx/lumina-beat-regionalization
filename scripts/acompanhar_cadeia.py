#!/usr/bin/env python3
"""Acompanha a cadeia MR_a2/MR_a3 e, rodando numa CELULA de notebook do JupyterLab, mantem o espaco ativo.

POR QUE
    No SageMaker Studio, o JupyterLab e considerado ocioso quando nao ha sessao ATIVA de kernel nem de terminal
    (https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-idle-shutdown.html; minimo de 60 min). Um processo
    com `nohup` nao conta: as duas quedas da cadeia (24/09) vieram ~1 h depois do ultimo uso. Uma celula que executa
    este script deixa o kernel OCUPADO enquanto a cadeia roda. E contorno, nao garantia: o ajuste certo e o tempo de
    ociosidade do dominio ou do perfil, que e do administrador.

O QUE FAZ
    A cada `--intervalo` segundos, enquanto houver processo da cadeia vivo, imprime a hora e a ultima linha de
    progresso do log mais recente. No fim imprime os `exit_` e sai com 0 se a cadeia terminou (`CADEIA COMPLETA`),
    2 se parou antes. So le.

USO (celula de notebook, depois de lancar a cadeia):
    !cd ~/testeArq/lumina-beat-regionalization && python3 scripts/acompanhar_cadeia.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROCESSOS_DA_CADEIA = "cadeia_mr_a2_a3.sh|extract_campaign_features.py|comparar_m0_mr_desenvolvimento.py|" \
                      "conferir_cabecas_salvas.py|conferir_reproducao_do_cache.py|conferir_codigo_do_cache.py"


def ultimo_progresso(linhas: list[str]) -> str:
    """A linha mais informativa do fim do log: fragmento, passo concluido ou falha. Pura."""
    for linha in reversed(linhas):
        texto = linha.strip()
        if texto.startswith(("fragmento", "exit_", "FALHOU", "CADEIA COMPLETA", "[extracao]", "[comparacao]",
                             "Traceback", "PASSOU")):
            return texto
    return "(sem progresso ainda)"


def resultado_final(linhas: list[str]) -> tuple[bool, list[str]]:
    """(completa, linhas de saida e falha). Pura."""
    relevantes = [l.strip() for l in linhas if l.strip().startswith(("exit_", "FALHOU", "CADEIA COMPLETA"))]
    return any(l.startswith("CADEIA COMPLETA") for l in relevantes), relevantes


def cadeia_viva() -> bool:
    return subprocess.run(["pgrep", "-f", PROCESSOS_DA_CADEIA], capture_output=True).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=Path("~/artifacts/redesenho"))
    parser.add_argument("--padrao", default="mr_a2_a3_*.log")
    parser.add_argument("--intervalo", type=int, default=300)
    parser.add_argument("--espera-inicial", type=int, default=120, help="segundos para a cadeia aparecer")
    args = parser.parse_args(argv)

    pasta = args.logs.expanduser()
    inicio = time.monotonic()
    while not cadeia_viva() and time.monotonic() - inicio < args.espera_inicial:
        time.sleep(10)
    while cadeia_viva():
        logs = sorted(pasta.glob(args.padrao))
        linhas = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines() if logs else []
        print(f"{time.strftime('%H:%M')}  {ultimo_progresso(linhas)}", flush=True)
        time.sleep(args.intervalo)
    logs = sorted(pasta.glob(args.padrao))
    if not logs:
        print("nenhum log da cadeia encontrado")
        return 2
    completa, relevantes = resultado_final(logs[-1].read_text(encoding="utf-8", errors="replace").splitlines())
    print(f"\n[{logs[-1].name}] cadeia {'COMPLETA' if completa else 'PAROU ANTES DO FIM'}")
    for linha in relevantes:
        print(f"  {linha}")
    return 0 if completa else 2


if __name__ == "__main__":
    sys.exit(main())
