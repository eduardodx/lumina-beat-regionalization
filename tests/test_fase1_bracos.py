"""Fase 1, passo 3 (bracos) sobre um release sintetico minimo e tres caches no formato do passo 2.

No Windows nao ha sklearn nem o pacote `mosaic`: o ajuste usa um substituto (ridge exato) e as checagens do Mosaic
usam copias locais. No .venv do Mosaic (notebook) os mesmos testes rodam com o sklearn e com as funcoes reais do
avaliador (`load_system`, `load_predictions`, `check_fold_roles`, `check_benchmark_reference`, `frequency_features`).
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from eval.campanha.recortes import COLUNAS, hash_da_tabela, hash_do_conteudo  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402

try:
    import sklearn  # noqa: F401
    TEM_SKLEARN = True
except ImportError:
    TEM_SKLEARN = False
try:
    from mosaic.comparator_eval import candidate as _candidate
    from mosaic.comparator_eval.frequency_arms import frequency_features as _frequencia_do_mosaic
    TEM_MOSAIC = True
except ImportError:
    TEM_MOSAIC = False

PAINEIS = ("missense", "splice", "noncoding", "plof", "synonymous")
N_POR_FOLD = 40
NAO_ELEGIVEIS = {"v0003", "v0042"}       # v0042: gold do fold 1, cai fora da validation do run 0
VIZINHA = "v0159"                       # fold 3, a 1 kb de v0080 (fold 2): purga interna no run 0
IDENTIDADE = {"versao_do_extrator": "campanha_r03_extracao_v2", "sistema": "M0", "checkpoint_sha256": "ck",
              "adapter_sha256": None, "fasta_sha256": "fa", "janela_bp": 4096, "indice_focal": 2047,
              "lote": {"variantes_por_lote": 8}, "extracoes": ["cabecas_172", "leitura_antiga_1344"],
              "codigo": {"arquivo": "sha"}, "ambiente": {"numpy": "x"}}


def frequencia_copiada(frame, *, with_abraom=False):
    """Copia literal de mosaic.comparator_eval.frequency_arms.frequency_features (f2e9a9f), para o Windows."""
    def log_af(values):
        return np.log10(values.fillna(0).clip(lower=0) + 1e-6)
    status = frame["gnomad_status"].fillna("unknown")
    global_af = frame["gnomad_v4_af"].where(~status.isin(["not_found", "ac0"]), 0)
    features = pd.DataFrame({"global": log_af(global_af), "popmax": log_af(frame["gnomad_v4_popmax_af"]),
                             "amr": log_af(frame["gnomad_v4_af_amr"]), "afr": log_af(frame["gnomad_v4_af_afr"]),
                             "nfe": log_af(frame["gnomad_v4_af_nfe"]),
                             "not_found": (status == "not_found").astype(float),
                             "ac0": (status == "ac0").astype(float)}, index=frame.index)
    if with_abraom:
        features["abraom"] = log_af(frame["abraom_af"])
        features["abraom_present"] = frame["present_abraom"].fillna(False).astype(float)
    return features


def ajustador_ridge(Z, y, grade):
    """Substituto sem sklearn: ridge exato pelo dual (n x n). So serve para testar o encanamento."""
    yc = y - y.mean()
    kernel = Z @ Z.T
    saida = []
    for c in grade:
        alfa = np.linalg.solve(kernel + np.eye(len(Z)) / c, yc)
        saida.append({"C": float(c), "coef": Z.T @ alfa, "intercepto": float(np.log(y.mean() / (1 - y.mean()))),
                      "n_iter": 1, "segundos": 0.0, "avisos": [], "convergiu": True})
    return saida


def _linhas_do_release() -> pd.DataFrame:
    rng = np.random.default_rng(20261004)
    linhas = []
    for fold in range(5):
        for j in range(N_POR_FOLD):
            i = fold * N_POR_FOLD + j
            gnomad = ("present", "not_found", "ac0", "present")[j % 4]
            abraom = ("present", "present", "ac0", "no_call", "not_found")[j % 5]
            af = float(rng.uniform(1e-5, 0.05))
            linhas.append({
                "variant_id": f"v{i:04d}", "chrom": "chr1" if fold % 2 == 0 else "chr2",
                "pos_1based": 1_000_000 * (fold + 1) + 10_000 * j, "ref": "A", "alt": "G",
                "binary_label": (j // 5) % 2, "label_tier": "consensus" if j % 4 == 3 else "gold",
                "primary_panel": PAINEIS[j % 5], "sequence_eligible": f"v{i:04d}" not in NAO_ELEGIVEIS,
                "overlap_cluster_id": f"c{i}", "core_fold": fold,
                # j == 0: purgada do treino no run fold+1; j == 1: purgada da validation no run fold-1
                "core_purged_runs": [(fold + 1) % 5] if j == 0 else [(fold - 1) % 5] if j == 1 else [],
                "gnomad_status": gnomad, "gnomad_v4_af": {"present": af, "not_found": np.nan, "ac0": 0.0}[gnomad],
                "gnomad_v4_popmax_af": af * 2 if gnomad == "present" else np.nan,
                "gnomad_v4_af_amr": af if gnomad == "present" else np.nan,
                "gnomad_v4_af_afr": af / 2 if gnomad == "present" else np.nan,
                "gnomad_v4_af_nfe": np.nan,
                "abraom_status": abraom,
                "abraom_af": {"present": af, "ac0": 0.0}.get(abraom, np.nan),
                "abraom_an": {"present": 2342.0, "ac0": 2000.0, "no_call": 0.0}.get(abraom, np.nan),
                "abraom_filter": {"present": "PASS" if j % 5 == 0 else "LowQual", "ac0": "PASS",
                                  "no_call": "."}.get(abraom),
                "present_abraom": abraom == "present"})
    # A vizinha (fold 3) fica a 1 kb de v0080 (fold 2, chr1:3.000.000); purgas coerentes com a regra do Mosaic.
    for linha in linhas:
        if linha["variant_id"] == VIZINHA:
            linha.update({"chrom": "chr1", "pos_1based": 3_001_000, "core_purged_runs": [1, 2]})
        if linha["variant_id"] == "v0198":
            linha.update({"abraom_status": None, "present_abraom": False})
    return pd.DataFrame(linhas)


def _gravar_release(entrega: Path, df: pd.DataFrame) -> None:
    raiz = entrega / bracos.RELEASE
    (raiz / "views" / "4kb").mkdir(parents=True)
    (entrega / "config").mkdir()
    (entrega / bracos.PROTOCOLO_DE_ESTUDOS).write_text("release_identity: sintetico\n", encoding="utf-8")
    df = df.assign(binary_label=df["binary_label"].astype("int8"), pos_1based=df["pos_1based"].astype("int64"))
    df[["variant_id", "chrom", "pos_1based", "ref", "alt", "binary_label", "label_tier"]].to_parquet(
        raiz / "clinical-variants.parquet", index=False)
    df[["variant_id", "sequence_eligible", "overlap_cluster_id", "core_fold", "core_purged_runs"]].to_parquet(
        raiz / bracos.VISTA, index=False)
    df[["variant_id", "primary_panel"]].to_parquet(raiz / "evaluation-panels.parquet", index=False)
    df[["variant_id", *bracos.COLUNAS_DO_GNOMAD, *bracos.COLUNAS_DO_ABRAOM]].to_parquet(
        raiz / "variant-annotations.parquet", index=False)


def _vetores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    e = rng.normal(0, 0.5, size=(len(df), 1344)).astype(np.float32)
    e[:, 0] += 1.5 * df["binary_label"].to_numpy()
    return dict(zip(df["variant_id"], e))


def _gravar_cache(pasta: Path, tabela: pd.DataFrame, vetores: dict[str, np.ndarray], papel: str,
                  fragmentos: int = 1) -> None:
    pasta.mkdir(parents=True)
    tabela = tabela.assign(papel=papel)[list(COLUNAS) + ["papel"]].reset_index(drop=True)
    tabela.to_parquet(pasta / "tabela.parquet", index=False)
    relida = pd.read_parquet(pasta / "tabela.parquet")
    ident = {**IDENTIDADE, "tabela_sha256_conteudo": hash_do_conteudo(relida),
             "tabela_sha256_composicao": hash_da_tabela(relida), "papeis": [papel], "revisao_do_codigo": papel}
    (pasta / "identidade.json").write_text(json.dumps(ident), encoding="utf-8")
    (pasta / "manifesto.json").write_text(json.dumps({"completo": True, "identidade": ident,
                                                      "variantes_na_tabela": len(tabela),
                                                      "variantes_no_cache": len(tabela), "faltando": 0}),
                                          encoding="utf-8")
    for k, parte in enumerate(np.array_split(np.arange(len(tabela)), fragmentos)):
        ids = tabela["variant_id"].to_numpy(str)[parte]
        np.savez(pasta / f"fragmento_{k:05d}.npz", variant_id=ids, papel=np.array([papel] * len(ids)),
                 cabecas_172=np.zeros((len(ids), 172), dtype=np.float32),
                 leitura_antiga_1344=np.stack([vetores[v] for v in ids]))


def _gravar_caches(pasta: Path, entrega: Path, vetores: dict[str, np.ndarray]) -> list[Path]:
    tabela = bracos.tabela_do_release(entrega / bracos.RELEASE)
    grupos = np.arange(len(tabela)) % 3
    caches = []
    for k, (nome, fragmentos) in enumerate((("g3", 2), ("g7", 1), ("complemento", 1))):
        caches.append(pasta / nome)
        _gravar_cache(caches[-1], tabela[grupos == k], vetores, nome, fragmentos)
    return caches


def _ferramentas(referencia: dict[str, str]) -> dict:
    """As do Mosaic quando o pacote existe; senao, copias locais das checagens que o passo 3 usa."""
    def verificar(raiz, config_dir=None):
        return {"release_id": "sintetico", "version": "v1", "release_identity_hash": referencia["release_identity_hash"],
                "protocol_hash": referencia["protocol_hash"]}

    if TEM_MOSAIC:
        return {"verificar_identidade": verificar, "frequencia_oficial": _frequencia_do_mosaic,
                "load_system": _candidate.load_system, "load_predictions": _candidate.load_predictions,
                "check_fold_roles": _candidate.check_fold_roles,
                "check_benchmark_reference": _candidate.check_benchmark_reference}

    def load_system(caminho):
        conteudo = json.loads(Path(caminho).read_text(encoding="utf-8"))
        assert {"id", "mode", "context_bp", "training_cutoff", "exposure"} <= set(conteudo)
        return conteudo

    def check_fold_roles(frame, preds, *, track, window):
        fold = preds["variant_id"].map(frame.set_index("variant_id")["core_fold"]).to_numpy()
        esperado = np.where(preds["role"] == "test", preds["run"], (preds["run"] + 1) % 5)
        if (fold != esperado).any():
            raise ValueError("predicoes fora do fold da execucao e do papel")

    def check_benchmark_reference(sistema, identidade, *, protocol_sha256=None):
        b = sistema["benchmark"]
        if (b["release_identity_hash"], b["protocol_hash"], b["study_protocol_sha256"]) != (
                identidade["release_identity_hash"], identidade["protocol_hash"], protocol_sha256):
            raise ValueError("benchmark divergente")

    return {"verificar_identidade": verificar, "frequencia_oficial": frequencia_copiada, "load_system": load_system,
            "load_predictions": lambda caminho, sistema: pd.read_parquet(caminho), "check_fold_roles": check_fold_roles,
            "check_benchmark_reference": check_benchmark_reference}


class Cenario:
    """Entrega, caches e referencia sinteticos num diretorio temporario."""

    def __init__(self, pasta: Path):
        self.pasta = pasta
        self.entrega = pasta / "entrega"
        self.df = _linhas_do_release()
        _gravar_release(self.entrega, self.df)
        self.vetores = _vetores(self.df)
        self.caches = _gravar_caches(pasta / "caches", self.entrega, self.vetores)
        sha = hashlib.sha256((self.entrega / bracos.PROTOCOLO_DE_ESTUDOS).read_bytes()).hexdigest()
        self.referencia = {"release_identity_hash": "r" * 64, "protocol_hash": "p" * 64, "study_protocol_sha256": sha}

    def argv(self, destino: Path) -> list[str]:
        saida = ["--entrega", str(self.entrega), "--out-dir", str(destino)]
        for cache in self.caches:
            saida += ["--cache", str(cache)]
        return saida

    def rodar(self, destino: Path, ferramentas: dict | None = None) -> int:
        with patch.object(bracos, "ferramentas_do_mosaic", return_value=ferramentas or _ferramentas(self.referencia)), \
                patch.object(bracos, "REFERENCIA_DO_BENCHMARK", self.referencia), \
                patch.object(bracos, "CONTAGENS_DO_GUIA_RUN0", None), \
                patch.object(bracos, "AJUSTADOR", bracos.ajustar_sklearn if TEM_SKLEARN else ajustador_ridge):
            return bracos.main(self.argv(destino))


class FeaturesTests(unittest.TestCase):
    def test_bloco_f_igual_ao_oficial_e_bloco_br(self):
        df = _linhas_do_release()
        conferencia = bracos.conferir_frequencia_oficial(df, _frequencia_do_mosaic if TEM_MOSAIC else frequencia_copiada)
        self.assertEqual(conferencia, {"f": 0.0, "br_log_af": 0.0, "br_present": 0.0})
        br = bracos.bloco_br(df).set_index(df["variant_id"])
        self.assertEqual(list(br.columns), ["br_log_af", "br_present", "br_ac0", "br_no_call", "br_not_found",
                                            "br_pass", "br_an"])
        self.assertFalse(br.isna().any().any())
        nao_achada = br.loc["v0004"]   # j=4: not_found
        self.assertEqual(nao_achada["br_log_af"], -6.0)
        self.assertEqual(nao_achada[["br_present", "br_ac0", "br_no_call", "br_pass", "br_an"]].sum(), 0)
        self.assertEqual(nao_achada["br_not_found"], 1.0)
        sem_chamada = br.loc["v0003"]  # j=3: no_call, AN 0, FILTER "."
        self.assertEqual((sem_chamada["br_no_call"], sem_chamada["br_an"], sem_chamada["br_pass"]), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(br.loc["v0002", "br_an"], 2000 / 2342)
        self.assertEqual(br.loc["v0198", ["br_present", "br_ac0", "br_no_call", "br_not_found"]].sum(), 0,
                         "estado nulo fica com os quatro indicadores em zero")
        self.assertEqual((br.loc["v0000", "br_pass"], br.loc["v0001", "br_pass"]), (1.0, 0.0))

    def test_bloco_f_diferente_do_oficial_reprova(self):
        df = _linhas_do_release()

        def outro(frame, *, with_abraom=False):
            saida = frequencia_copiada(frame, with_abraom=with_abraom)
            saida["global"] = saida["global"] + 1e-9
            return saida
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "difere do oficial"):
            bracos.conferir_frequencia_oficial(df, outro)

    def test_estado_desconhecido_e_an_impossivel_reprovam(self):
        df = _linhas_do_release()
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "abraom_status"):
            bracos.bloco_br(df.assign(abraom_status="outro"))
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "abraom_an"):
            bracos.bloco_br(df.assign(abraom_an=2343.0))


class LinhasTests(unittest.TestCase):
    def test_papeis_por_execucao(self):
        df = _linhas_do_release()
        df["sequence_eligible"] = df["sequence_eligible"].astype(bool)
        por_run, contagens = bracos.linhas_por_execucao(df)
        elegiveis = df[df["sequence_eligible"]].reset_index(drop=True)
        for run, idx in enumerate(por_run):
            linhas = {nome: elegiveis.iloc[v] for nome, v in idx.items()}
            purgada = lambda t: t["core_purged_runs"].map(lambda r: run in r)  # noqa: E731
            self.assertTrue((~linhas["treino"]["core_fold"].isin([run, (run + 1) % 5])).all())
            self.assertFalse(purgada(linhas["treino"]).any())
            self.assertTrue((linhas["validation"]["core_fold"] == (run + 1) % 5).all())
            self.assertTrue((linhas["validation"]["label_tier"] == "gold").all())
            self.assertFalse(purgada(linhas["validation"]).any())
            self.assertTrue((linhas["teste"]["core_fold"] == run).all())
            self.assertEqual(set(linhas["teste"]["label_tier"]), {"gold", "consensus"}, "teste com todos os tiers")
        self.assertEqual(contagens[0]["fora_da_elegibilidade"]["validation"], 1)   # v0042
        self.assertEqual(sum(c["efetivas"]["teste"] for c in contagens), len(elegiveis))

    def test_purga_interna_na_fronteira_de_4096(self):
        crom = np.array(["chr1", "chr1", "chr1", "chr2"])
        pos = np.array([10_000, 14_095, 14_096, 10_000])
        perto = bracos.perto_de(crom, pos, np.array([0]), np.array([1, 2, 3]))
        self.assertEqual(perto.tolist(), [True, False, False])

    def test_escolha_de_c(self):
        grade = [{"macro_validacao": v} for v in (0.6, 0.7, 0.7, None, 0.65)]
        self.assertEqual(bracos.escolher_c(grade), 1, "empate fica com o menor C")
        with self.assertRaises(bracos.CabecaSemSelecao):
            bracos.escolher_c([{"macro_validacao": None}])


class LeituraTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.c = Cenario(Path(temp.name))
        self.raiz = self.c.entrega / bracos.RELEASE
        tabela = bracos.tabela_do_release(self.raiz)
        self.ids = tabela["variant_id"].astype(str).to_numpy()

    def test_uniao_dos_tres_caches_alinhada(self):
        matriz, fontes = bracos.carregar_leitura(self.c.caches, self.raiz, self.ids)
        self.assertEqual(matriz.shape, (len(self.ids), 1344))
        for i in (0, 57, len(self.ids) - 1):
            np.testing.assert_array_equal(matriz[i], self.c.vetores[self.ids[i]])
        self.assertEqual(sum(f["lidas_no_release"] for f in fontes.values()), len(self.ids))

    def test_variante_sem_leitura_repetida_ou_identidade_divergente_reprova(self):
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "sem leitura_antiga_1344"):
            bracos.carregar_leitura(self.c.caches[:2], self.raiz, self.ids)
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "outro cache"):
            bracos.carregar_leitura([*self.c.caches, self.c.caches[1]], self.raiz, self.ids)
        ident = json.loads((self.c.caches[2] / "identidade.json").read_text(encoding="utf-8"))
        ident["checkpoint_sha256"] = "outro"
        (self.c.caches[2] / "identidade.json").write_text(json.dumps(ident), encoding="utf-8")
        with self.assertRaisesRegex(bracos.FalhaDaFase1, "identidade difere"):
            bracos.carregar_leitura(self.c.caches, self.raiz, self.ids)


class PontaAPontaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = Cenario(Path(cls.temp.name))
        cls.destino = Path(cls.temp.name) / "bracos"
        cls.codigo = cls.c.rodar(cls.destino)
        df = cls.c.df[cls.c.df["sequence_eligible"]].reset_index(drop=True)
        cls.elegiveis = df
        cls.preds = {cfg["id"]: pd.read_parquet(cls.destino / cfg["id"] / "predictions.parquet")
                     for cfg in bracos.BRACOS.values()} if cls.codigo == 0 else {}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_rodou_e_gravou_tudo_atomicamente(self):
        self.assertEqual(self.codigo, 0)
        self.assertFalse(self.destino.with_name("bracos.tmp").exists())
        for nome in ("selecao.json", "linhas.json", "fontes.json", "diagnosticos/s_fora_da_amostra.parquet"):
            self.assertTrue((self.destino / nome).exists(), nome)
        self.assertEqual(len(list((self.destino / "modelos").glob("*.npz"))), 6 * 5)
        self.assertEqual(self.c.rodar(self.destino), 2, "nunca grava por cima")

    def test_mesmas_linhas_e_papeis_coerentes_em_todos_os_bracos(self):
        chaves = {i: set(map(tuple, p[["variant_id", "run", "role"]].itertuples(index=False)))
                  for i, p in self.preds.items()}
        self.assertEqual(len(set(map(frozenset, chaves.values()))), 1)
        p = self.preds["fase1-e"].merge(self.c.df, on="variant_id")
        self.assertTrue(p["sequence_eligible"].all())
        self.assertTrue((p["study"] == "core_locus").all() and (p["window_bp"] == 4096).all())
        teste, val = p[p["role"] == "test"], p[p["role"] == "validation"]
        self.assertTrue((teste["core_fold"] == teste["run"]).all())
        self.assertEqual(sorted(teste["variant_id"]), sorted(self.elegiveis["variant_id"]))
        self.assertTrue((val["core_fold"] == (val["run"] + 1) % 5).all())
        self.assertTrue((val["label_tier"] == "gold").all())
        self.assertFalse(val.apply(lambda r: r["run"] in r["core_purged_runs"], axis=1).any())
        for preds in self.preds.values():
            self.assertTrue(np.isfinite(preds["score"]).all())

    def _matriz(self, blocos, ids, s=None):
        linhas = self.elegiveis.set_index("variant_id").loc[ids].reset_index()
        partes = {"f": bracos.bloco_f(linhas).to_numpy(), "br": bracos.bloco_br(linhas).to_numpy(),
                  "e": np.stack([self.c.vetores[v] for v in ids]).astype(np.float64),
                  "s": None if s is None else np.asarray(s, dtype=np.float64)[:, None]}
        return np.hstack([partes[b] for b in blocos])

    def test_modelos_gravados_reproduzem_as_predicoes(self):
        e = self.preds["fase1-e"].set_index(["variant_id", "run", "role"])["score"]
        for braco, cfg in bracos.BRACOS.items():
            preds = self.preds[cfg["id"]]
            for run in range(5):
                modelo = np.load(self.destino / "modelos" / f"{cfg['id']}_run{run}.npz")
                self.assertIn(float(modelo["C"]), bracos.GRADE_C)
                parte = preds[preds["run"] == run]
                ids = parte["variant_id"].tolist()
                s = None
                if "s" in cfg["blocos"]:
                    s = e.loc[list(zip(ids, parte["run"], parte["role"]))].to_numpy()
                X = self._matriz(cfg["blocos"], ids, s)
                esperado = ((X - modelo["media"]) / modelo["desvio"]) @ modelo["coef"] + float(modelo["intercepto"])
                np.testing.assert_allclose(parte["score"].to_numpy(), esperado, rtol=1e-9, atol=1e-9,
                                           err_msg=f"{braco} run {run}")

    def test_s_mais_f_usa_score_fora_da_amostra_com_purga_interna(self):
        s = pd.read_parquet(self.destino / "diagnosticos" / "s_fora_da_amostra.parquet")
        selecao = json.loads((self.destino / "selecao.json").read_text(encoding="utf-8"))["bracos"]
        for run in range(5):
            parte = s[s["run"] == run].merge(self.elegiveis[["variant_id", "core_fold"]], on="variant_id")
            self.assertFalse(parte["variant_id"].duplicated().any())
            self.assertTrue((parte["fold_interno"] == parte["core_fold"]).all())
            self.assertFalse(parte["fold_interno"].isin([run, (run + 1) % 5]).any())
            linhas = json.loads((self.destino / "linhas.json").read_text(encoding="utf-8"))
            self.assertEqual(len(parte), linhas["execucoes"][run]["efetivas"]["treino"])
            internos = selecao["S+F"][run]["s_interno"]
            self.assertEqual(len(internos), 3)
            e_c = selecao["E"][run]["C"]
            self.assertTrue(all(i["C"] == e_c for i in internos), "o S interno usa o C escolhido para E")
        purgas_run0 = {i["fold_interno"]: i["purgadas_internas"] for i in selecao["S+F"][0]["s_interno"]}
        self.assertEqual(purgas_run0, {2: 1, 3: 1, 4: 0})

    def test_system_yaml_no_contrato(self):
        for braco, cfg in bracos.BRACOS.items():
            conteudo = json.loads((self.destino / cfg["id"] / "system.yaml").read_text(encoding="utf-8"))
            self.assertEqual(bracos._floats(conteudo), [])
            self.assertEqual((conteudo["id"], conteudo["mode"], conteudo["context_bp"]),
                             (cfg["id"], "trained", cfg["context_bp"]))
            self.assertEqual(conteudo["benchmark"], self.c.referencia)
            usa_r03 = bool({"e", "s"} & set(cfg["blocos"]))
            self.assertEqual(conteudo["training_cutoff"], None if usa_r03 else "2026-06")
            self.assertEqual("backbone_r03" in conteudo["exposure"]["clinical_labels"], usa_r03)


class FalhaRapidaTests(unittest.TestCase):
    def test_entrega_diferente_nao_treina_nem_grava(self):
        with tempfile.TemporaryDirectory() as pasta:
            c = Cenario(Path(pasta))
            outra = dict(c.referencia, study_protocol_sha256="0" * 64)
            destino = Path(pasta) / "bracos"
            with patch.object(bracos, "ferramentas_do_mosaic", return_value=_ferramentas(c.referencia)), \
                    patch.object(bracos, "REFERENCIA_DO_BENCHMARK", outra), \
                    patch.object(bracos, "AJUSTADOR", side_effect=AssertionError("treinou")):
                codigo = bracos.main(c.argv(destino))
            self.assertEqual(codigo, 2)
            self.assertFalse(destino.exists() or destino.with_name("bracos.tmp").exists())

    def test_contagens_do_guia_sao_conferidas(self):
        with tempfile.TemporaryDirectory() as pasta:
            c = Cenario(Path(pasta))
            destino = Path(pasta) / "bracos"
            with patch.object(bracos, "ferramentas_do_mosaic", return_value=_ferramentas(c.referencia)), \
                    patch.object(bracos, "REFERENCIA_DO_BENCHMARK", c.referencia):
                self.assertEqual(bracos.main(c.argv(destino)), 2)   # guia do release real (194.666 ...)
            self.assertFalse(destino.exists())


@unittest.skipUnless(TEM_SKLEARN, "sem sklearn (Windows): roda no .venv do Mosaic")
class SklearnTests(unittest.TestCase):
    def test_ajuste_bate_com_newton_exato(self):
        rng = np.random.default_rng(3)
        Z = rng.normal(size=(400, 5))
        y = (Z @ np.array([1.0, -0.5, 0.0, 0.3, 0.0]) + rng.logistic(size=400) > 0).astype(int)
        for r in bracos.ajustar_sklearn(Z, y, bracos.GRADE_C):
            A = np.hstack([Z, np.ones((len(Z), 1))])
            w = np.zeros(6)
            reg = np.r_[np.full(5, 1.0 / r["C"]), 0.0]
            for _ in range(100):
                p = 1 / (1 + np.exp(-(A @ w)))
                passo = np.linalg.solve((A * (p * (1 - p))[:, None]).T @ A + np.diag(reg), A.T @ (p - y) + reg * w)
                w -= passo
                if np.max(np.abs(passo)) < 1e-12:
                    break
            np.testing.assert_allclose(r["coef"], w[:5], atol=1e-5, err_msg=f"C={r['C']}")
            self.assertAlmostEqual(r["intercepto"], w[5], places=5)
            self.assertTrue(r["convergiu"], r["avisos"])


if __name__ == "__main__":
    unittest.main()
