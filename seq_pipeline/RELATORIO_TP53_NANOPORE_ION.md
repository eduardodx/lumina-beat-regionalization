# Relatório — Sequenciamento de TP53 (Nanopore) e concordância com o painel Ion (BRCA Expanded)

**Frente:** análise dos dados de sequenciamento do laboratório de genética.
**Autor:** Gabriel · **Solicitante:** Eduardo · **Data:** 2026-08-24 · **Referência:** GRCh38/hg38.

> ⚠️ **Dados de paciente (LGPD).** Este relatório traz apenas resultados agregados e pseudônimos de
> amostra. Os arquivos brutos (BAM/VCF) permanecem no ambiente de trabalho, fora de repositórios públicos.

---

## 1. Resumo executivo

A partir dos arquivos brutos (FASTQ) de duas plataformas — **Ion Torrent** (painel Oncomine BRCA Expanded)
e **Oxford Nanopore** — construímos um pipeline que (1) **alinhou** os dados ao genoma de referência hg38,
(2) **identificou as variantes** do Nanopore e (3) **verificou** essas variantes contra o Ion. Foi feito
nas **6 amostras que possuem as duas plataformas** (353, 358, 361, 362, 370, 674).

**Principais achados:**
- As duas plataformas sequenciam **alvos diferentes** do mesmo paciente: o **Ion** é um painel que cobre
  BRCA1, BRCA2 **e TP53**; o **Nanopore** é um **amplicon focado em TP53**.
- Onde as duas se sobrepõem (TP53 exônico), a concordância é **100%** e não houve nenhum falso-positivo do
  Nanopore — o pipeline é confiável.
- **Nenhuma variante patogênica de TP53** foi encontrada nas 6 amostras — confirmado por **três fontes
  independentes** (catálogo ClinVar, consequência funcional/VEP e frequência populacional/gnomAD). As
  variantes são **polimorfismos germinativos benignos**; há **uma variante de significado incerto (VUS)**
  na amostra 674, registrada para eventual revisão.

---

## 2. Dados analisados

| Amostra | Ion — cobertura média (BRCA1 / BRCA2 / TP53) | Nanopore — TP53 (reads · profundidade) | Variantes chamadas (Nanopore, TP53) |
|---|---|---|---|
| 353 | 33× / 81× / 47× | 342 · 80× | 46 |
| 358 | 29× / 75× / 41× | 645 · 152× | 29 |
| 361 | 38× / 90× / 64× | 464 · 103× | 50 |
| 362 | 39× / 100× / 59× | 561 · 133× | 30 |
| 370 | 35× / 90× / 55× | 558 · 128× | 22 |
| 674 | 29× / 79× / 37× | 768 · 186× | 23 |

Ion: ~99% dos reads mapeados; profundidade alta nos amplicons-alvo. Nanopore: ~100% dos reads mapeados,
concentrados em TP53 (chr17:7.66–7.69 Mb), profundidade adequada para chamada de variantes.

---

## 3. Natureza dos dados (descoberta)

O conteúdo não estava documentado; foi **inferido dos próprios dados e confirmado por metadados**:

- **Ion Torrent = painel Oncomine BRCA Expanded** (nome no próprio arquivo). Cobre BRCA1 (chr17), BRCA2
  (chr13) e **TP53** (chr17), além de genes de reparo (HRR). Reads curtos (~120 pb), alta profundidade.
- **Nanopore = amplicon longo de TP53.** O cabeçalho dos FASTQ confirma: `sample_id=TP53_DDC`,
  `protocol_group_id=TP53_DDC_...`, química **R10.4.1 sup v5.2.0**. Reads longos (~8 kb), cobrindo o gene
  TP53 inteiro (íntrons inclusive), com breadth ~72% da janela.
- **Alvo em comum = TP53** (nas duas plataformas). BRCA1/BRCA2 aparecem só no Ion.

---

## 4. Metodologia

| Etapa | Ferramenta / versão | Observação |
|---|---|---|
| Caracterização (QC) | `seqkit`, `samtools` 1.24 | formato, plataforma, cobertura, pareamento de amostras |
| Alinhamento ao hg38 | `minimap2` (`-ax map-ont` Nanopore; `-ax sr` Ion) | referência `hg38.fa` |
| Chamada de variantes (Nanopore) | `Clair3` 2.0.2, modelo `r1041_e82_400bps_sup_v520` | restrito a TP53 (BED chr17:7660000-7695000) |
| Verificação no Ion | genotipagem direcionada (pysam) | conta o suporte do Ion em cada variante do Nanopore |
| Anotação clínica | `ClinVar` (release 2026-06-06) | classificação de significância (CLNSIG) |
| Anotação funcional | `VEP` (Ensembl REST) + `gnomAD` | consequência (missense/frameshift/intron/UTR…) + frequência populacional |

**Decisão metodológica importante (passo 3).** Em vez de chamar variantes no Ion de forma independente e
cruzar os VCFs, fizemos **genotipagem direcionada**: para cada variante do Nanopore, medimos diretamente o
suporte do Ion naquela posição. Motivos: (a) o caller do Ion (Clair3 modo Illumina) apresentou falha de
biblioteca no ambiente; e (b) o painel Ion cobre TP53 apenas nos éxons, então um cruzamento ingênuo
produziria falsas "discordâncias" onde o Ion simplesmente não tem cobertura. A genotipagem direcionada
separa três situações: **CONFIRMADA**, **NÃO-COBERTA** (fora do alcance do painel Ion) e **NÃO-CONFIRMADA**.

---

## 5. Resultados

### 5.1 Concordância Nanopore × Ion (por amostra)

| Amostra | Confirmadas pelo Ion | Fora da cobertura do Ion | SNVs onde o Ion cobre |
|---|---|---|---|
| 353 | 2 | 44 | 1/1 (100%) |
| 358 | 0 | 29 | — (Ion não cobre nenhuma) |
| 361 | 3 | 47 | 2/2 (100%) |
| 362 | 3 | 27 | 2/2 (100%) |
| 370 | 1 | 21 | 1/1 (100%) |
| 674 | 5 | 18 | 4/4 (100%) |

- **`NÃO-CONFIRMADA` = 0 em todas as amostras:** o Ion nunca cobriu bem uma posição e contradisse o
  Nanopore — ou seja, **nenhum falso-positivo detectável** do Nanopore.
- **Onde há sobreposição, concordância = 100%** (incl. até uma deleção de 16 pb na amostra 674).
- A maioria das variantes do Nanopore cai **fora** do painel Ion (o painel toca TP53 só nos éxons; o
  Nanopore vê o gene inteiro). Logo, **o Ion valida apenas parcialmente** o Nanopore — na 358, não há
  sobreposição alguma. Isso é uma característica de desenho dos ensaios, não uma falha de qualidade.

### 5.2 Significância clínica (TP53) — triangulada por três fontes

As 104 variantes únicas (só `PASS`) das 6 amostras foram avaliadas por **catálogo (ClinVar)**,
**consequência funcional (VEP)** e **frequência populacional (gnomAD)**:

- **Patogênicas / provavelmente patogênicas: 0** — confirmado pelos três eixos.
- A esmagadora maioria é **intrônica, UTR ou não-codificante**, com frequência populacional **alta**
  (gnomAD tipicamente 5–90%) — **polimorfismos germinativos comuns**. Inclui o **rs1042522 (p.Pro72Arg)**
  (gnomAD ~72%, benigno; rotulado *TP53 polymorphism*).
- **Nenhuma variante de alto impacto real.** Três chamadas de *frameshift* (chr17:7667260) apareceram na
  triagem, mas são **artefatos de homopolímero**: têm frequência populacional de **33–41%** (uma frameshift
  patogênica não teria essa frequência) e caem fora dos éxons codificantes canônicos. Descartadas pelo gnomAD.
- **1 variante de significado incerto (VUS):** chr17:7674889 A>C (missense) na **amostra 674** — rara
  (gnomAD 3×10⁻⁵), confirmada nas duas plataformas, ClinVar predominantemente *benign/likely_benign* com
  uma submissão *uncertain*. **Não é patogênica**, mas fica registrada para eventual revisão manual.

---

## 6. Conclusões

1. O pipeline reproduz corretamente o fluxo pedido: **Nanopore → hg38 → variantes → verificação no Ion**.
2. As variantes do Nanopore em TP53 são **confiáveis** (100% de concordância onde o Ion pode confirmar; nenhum falso-positivo).
3. **Nenhuma mutação patogênica de TP53** nas 6 amostras — confirmado por **três fontes independentes**
   (ClinVar + consequência funcional/VEP + frequência populacional/gnomAD). Apenas polimorfismos
   germinativos benignos; uma única **VUS** (amostra 674) registrada para revisão.

---

## 7. Limitações e decisões pendentes

- **Germinativo vs. somático.** O perfil de frequência alélica é **germinativo** (heterozigoto ~50%,
  homozigoto ~100%), e a análise foi feita para germinativo. **Se o objetivo for detectar mutação TP53
  somática (tumoral)** — tipicamente em baixa frequência alélica — o método precisa mudar (caller somático
  como ClairS/Mutect2 e limiares de FA baixos). **Esta é a principal decisão a confirmar.**
- **Sobreposição parcial das plataformas.** O Ion (painel focado) não serve como validador amplo do
  Nanopore (que cobre TP53 inteiro). Uma validação completa exigiria uma referência independente.
- **A concordância Nanopore×Ion é verificação pontual, não estatística.** A confirmação cruzada recai
  sobre as poucas posições exônicas cobertas pelo painel Ion (~10 SNVs no total das 6); é 100%, mas o n é
  pequeno — vale como checagem de qualidade, não como medida de sensibilidade/especificidade.
- **Indels em homopolímero.** Tanto Nanopore quanto Ion têm erro nesse contexto; indels foram tratados com
  cautela e a leitura forte é sobre os SNVs.
- **Amostras não analisadas:** `1005/1016/1036` (só Ion, painel BRCA) e `Amostra21/36_sarcoma` (só
  Nanopore) — podem ser processadas quando desejado.

---

## 8. Reprodutibilidade

Ambientes conda: `seqlab` (samtools, seqkit, minimap2, bcftools) e `clair3` (Clair3 2.0.2 + pysam).
Scripts (branch `seq-lab-pipeline`, diretório `seq_pipeline/`):
`characterize_seq_data.py` · `pilot_align_probe.py` · `compare_tp53_ont_vs_ion.py` · `annotate_tp53_clinvar.py` · `annotate_tp53_vep.py`.
Referência hg38; ClinVar release 2026-06-06; VEP Ensembl REST + gnomAD.
