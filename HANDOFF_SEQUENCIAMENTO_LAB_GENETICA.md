# HANDOFF — Frente NOVA: pipeline de sequenciamento (Ion Torrent + Nanopore) → variantes vs hg38

> **Para quem pega este documento num chat novo:** ele é **auto-contido**. Leia inteiro. É uma **frente
> NOVA e DESACOPLADA** da campanha de regionalização/Lumina (aquele é o `HANDOFF_R03_CONTINUACAO.md`; NÃO
> misture os dois — este aqui não usa o modelo Lumina, é bioinformática clássica de sequenciamento).
> Datado **2026-08-24**. Autor da frente: **Gabriel** (dev, TCC). Gestor: **Eduardo** (`eduardodx01`).
>
> **TL;DR:** o Eduardo entregou dados brutos de um laboratório de genética — saídas de duas plataformas
> de sequenciamento (**Ion Torrent** e **Oxford Nanopore**) — e pediu, em linguagem informal, para
> **montar a sequência, "parear", entender o que está acontecendo e identificar variantes contra o
> hg38**. Este handoff traduz isso num **pipeline de bioinformática concreto**, marca o que é
> **interpretação a validar com o Eduardo** vs. fato, e define o primeiro passo (inventário + QC).
> **Nada foi executado ainda** — este é o ponto de partida.
>
> **★ ACHADOS DA CARACTERIZAÇÃO (2026-08-24 — FATOS, prevalecem sobre as inferências das seções abaixo):**
> rodado `seq_pipeline/characterize_seq_data.py` sobre 17 arquivos / 11 amostras. (1) **Tudo é FASTQ**,
> nenhum BAM → **sem build pré-existente** (o risco hg19 sumiu; nós definimos hg38 no alinhamento).
> (2) **O painel Ion está explícito no nome dos arquivos: `Oncomine BRCA Expanded` (Thermo), chip Ion
> 530** — painel-alvo de **BRCA1/BRCA2**, com IonCode barcodes (germline/HRD). (3) **Ion** = ~120 bp/read,
> 494k–700k reads, 130–186 MB (9 amostras: 353, 358, 361, 362, 370, 674, 1005, 1016, 1036). (4) **Nanopore**
> = os `_final_clean.fastq` (long reads ~8.2 kb; `Amostra21/36_sarcoma` ~1.6 kb), **MAS profundidade
> baixíssima: 389–880 reads** (8 amostras: 353, 358, 361, 362, 370, 674 + Amostra21/36_sarcoma).
> (5) **6 amostras têm AS DUAS plataformas** (353, 358, 361, 362, 370, 674). Correções ao texto abaixo:
> os `_final_clean` são **Nanopore** (não Ion) e é tudo **FASTQ** (não BAM).
>
> **★★ QUADRO DEFINITIVO (2026-08-24 — alinhamento-piloto das 6 + respostas do Eduardo):**
> **(A)** O **Ion** (Oncomine BRCA Expanded) cobre **BRCA1, BRCA2 E TP53** (+ genes HRR em chr11/16/3/8),
> ~99% mapeado, 29–100× nos alvos. **(B)** O **Nanopore** = **amplicon longo de TP53** (chr17:7.66–7.69 Mb),
> profundidade **80–186×**, breadth ~72%, **ZERO em BRCA** — cada `_final_clean` tem só ~400–980 reads mas
> quase todos em TP53, então a cobertura de TP53 é ótima (minha leitura anterior de "profundidade
> baixíssima" estava medindo contra a régua errada). **(C) Alvo em comum = TP53** nas duas plataformas;
> BRCA1/2 só no Ion. **(D) "Parear" do Eduardo = ALINHAR ao hg38** (não comparar plataformas).
> **OBJETIVO do Eduardo, em 3 passos:** (1) alinhar Nanopore↔hg38 [**FEITO** — `seq_pipeline/pilot_align_probe.py`];
> (2) **chamar variantes do Nanopore em TP53** [próximo — Clair3]; (3) **ver se essas variantes também
> estão no Ion** (viável porque o Ion cobre TP53). Amostras avulsas a inventariar: só-Ion 1005/1016/1036;
> só-Nanopore `Amostra21/36_sarcoma` (avg ~1.6 kb — confirmar se também é TP53).
>
> **★★★ RESULTADO FINAL (2026-08-24 — pipeline completo nas 6 pareadas):** os 3 passos do Eduardo estão
> FEITOS. Envs conda: `seqlab` (samtools/seqkit/minimap2/bcftools); `clair3` (Clair3 2.0.2 + pysam).
> Pipeline: (1) alinhamento minimap2 `-ax map-ont`/`sr` (`pilot_align_probe.py`); (2) **Clair3** ONT modelo
> **`r1041_e82_400bps_sup_v520`** — o header do FASTQ confirmou química r10.4.1 sup v5.2.0 e
> **`sample_id=TP53_DDC`** (prova independente de que o Nanopore é ensaio de TP53); BED
> `~/seqlab/tp53.bed` = chr17:7660000-7695000; (3) **genotipagem direcionada** Nanopore→Ion
> (`compare_tp53_ont_vs_ion.py`), NÃO calling independente do Ion (o Clair3 `ilmn` quebra por lib
> `realigner` ausente no pacote, e o painel Ion cobre TP53 só nos éxons). **Achados:** concordância
> **100% onde o Ion cobre** e **0 falsos** (NAO_CONFIRMADA=0) nas 6; mas ~85–100% das variantes do
> Nanopore caem FORA do amplicon Ion (na 358, 100%) → o Ion valida só parcialmente (é painel focado; o
> Nanopore vê TP53 inteiro). **Anotação TRIANGULADA** (ClinVar `annotate_tp53_clinvar.py` + funcional VEP+gnomAD `annotate_tp53_vep.py`):
> **0 patogênicas/LP por 3 eixos** (catálogo + consequência + freq. populacional); 104 variantes, maioria
> intrônica/UTR comum; **1 VUS** (chr17:7674889 A>C na 674, missense rara gnomAD 3e-5). VEP-gotcha: o
> veredito PRECISA do gnomAD (AF≥1% = polimorfismo) senão frameshift-homopolímero e P72R viram falso-P/LP.
> Relatório final: `seq_pipeline/RELATORIO_TP53_NANOPORE_ION.md`. **Perfil GERMINATIVO**
> (AF~0.5/1.0); se o Eduardo quiser SOMÁTICO tumoral, refazer com caller somático (ClairS/Mutect2) + AF
> baixo. **Gotcha:** `Conflicting_classifications_of_pathogenicity` contém a substring "pathogenicity" —
> não é P/LP (corrigido no `clnsig_category`). Artefatos em `~/seqlab/` no notebook (BAMs/VCFs/TSVs), NÃO
> versionados. **Pendências:** decidir germinativo-vs-somático com o Eduardo; amostras avulsas
> (1005/1016/1036 só-Ion BRCA; Amostra21/36_sarcoma só-Nanopore); consolidar relatório pro Eduardo.

---

## 0. Contexto operacional (como trabalhamos)

- **Fluxo (herdado da outra frente):** o Windows local **não roda** nada pesado (sem torch/bioinfo/AWS)
  — serve para **editar código e escrever runbooks copiáveis**. O Gabriel **commita, dá `git pull` no
  notebook SageMaker e roda** (compute, S3). Nunca rodar pipeline/AWS a partir do Windows.
- **Repo:** por ora reusa o `lumina-beat-regionalization` (mesmo projeto), mas esta frente é
  independente. Sugestão: isolar tudo num diretório dedicado, ex. `seq_pipeline/` (scripts) +
  `~/artifacts/seqlab/` (dados/saídas no notebook), para não colidir com a campanha R03.
- **⚠️ Ambiente é o 1º obstáculo:** as ferramentas de sequenciamento (samtools, minimap2, bwa/bwa-mem2,
  bcftools, Clair3, DeepVariant, mosdepth, FastQC, NanoPlot, Sniffles2…) **NÃO estão** no `.venv` do
  Lumina (que é torch/mamba). Vai precisar de um **env conda bioinformático separado** (bioconda) ou
  containers. Ver §4.
- **Reusável do repo:** `~/hg38/hg38.fa` (a referência GRCh38 — o alvo do "identificar variantes a
  partir do hg38"); ClinVar (master + VCF `clinvar_20260606.vcf.gz`) para anotação; a convenção de
  chave canônica `GRCh38:chrom:pos:ref:alt` (normalização em `eval/clinvar/variant_utils.py`).

---

## 1. A missão (o pedido do Eduardo)

**Verbatim (parafraseado do que o Gabriel repassou):** a partir dos dados das máquinas Ion Torrent e
Nanopore — *"olhar se conseguimos montar a sequência, parear, encontrar o que está acontecendo, parear,
se conseguimos identificar variantes a partir do hg38"*.

**Tradução para pipeline (INTERPRETAÇÃO — validar com o Eduardo, ver §5):**
1. **"Montar a sequência"** → quase certamente **alinhamento/mapeamento dos reads ao hg38** (reference-
   based), **não** assembly *de novo* — porque o objetivo final é "variantes vs hg38". (Se algum dado
   Nanopore vier como sinal cru `fast5/pod5`, "montar" incluiria **basecalling** antes; mas as fotos
   mostram **FASTQ**, i.e. o basecalling já foi feito.)
2. **"Parear"** (dito 2×) → **AMBÍGUO, é a decisão #1.** Três leituras plausíveis:
   - (a) **Alinhar** reads à referência ("parear read↔genoma") — o sentido mais provável;
   - (b) **Parear as duas plataformas na MESMA amostra** (Ion × Nanopore) para validação cruzada de
     variantes — faz muito sentido, explicaria por que ele mandou as duas tecnologias juntas;
   - (c) **Tumor-normal** (contexto oncológico) — mas não há indício de amostras "normais" nas fotos.
3. **"Encontrar o que está acontecendo"** → **QC**: qualidade dos reads, cobertura, contaminação,
   adaptadores, quão bons são os dados de cada plataforma.
4. **"Identificar variantes a partir do hg38"** → **variant calling** contra GRCh38: SNVs/indels
   (e possivelmente SVs, forte no Nanopore) → VCF, depois anotado (ClinVar/VEP).

---

## 2. Os dados (do Google Drive do Eduardo — inventário pelas fotos)

**Fonte:** pasta `nanopore` compartilhada por `eduardodx01`
(`https://drive.google.com/drive/u/2/folders/1y666OovTeqs1KKacH7GvwnSWvmJ3RCui`), com subpastas:
- `FastQ_Ion Torrent`
- `FastQ_Sequenciamento por nanoporos`
- (e um `.DS_Store` — lixo de macOS, ignorar)

**Brutos Ion Torrent** (arquivos ~124–177 MB cada), nomes no padrão
`353_ALLS-R_2025_12_18_11_57_58_user_GSS5-0069-19-Chip_1…`:
- O padrão `user_GSS5-0069-XX-Chip_1` é o naming do **Ion GeneStudio S5 (GSS5)** — plataforma Ion
  Torrent. (Provável, pelo naming.) Amostras vistas: `353_ALLS`, `358_ADSG`, `361_ARSS`, `362_AAB`,
  `370_SSB`, `674_AHSL`, `1005_VBSG`, `1016_SCV`, `1036_DBB` (+ mais). Datas de run: 2025-12-18 e
  2026-06-10. Os prefixos alfabéticos (ALLS, ADSG…) parecem **códigos de amostra/paciente**; `ALLS`
  *pode* aludir a leucemia (ALL) — **especulativo**.
- **Formato a confirmar:** o S5 tipicamente exporta **BAM** (já alinhado por TMAP, muitas vezes a
  **hg19**) ou FASTQ. Precisamos ver a extensão/conteúdo real (ver §5, item 5).

**FASTQ "limpos"** (`*_final_clean.fastq`, ~6–19 MB): `353/358/361/362/370/674_final_clean.fastq` (os
números batem com os do Ion → provavelmente os FASTQ processados do Ion), + `Amostra21_sarcoma_final_
clean.fastq`, `Amostra36_Sarcoma_final_clean.fastq` (contexto **sarcoma** explícito).
- **Sinal importante:** FASTQ de **6–19 MB** é pequeno → sugere **painel-alvo (targeted panel)** de
  poucos genes, típico de diagnóstico onco-hematológico — **não** exoma/genoma inteiro. A confirmar.
- **`_final_clean` = já processado** (trimming/filtro) **por alguém antes de nós** — precisamos saber
  **o que "clean" fez** para não refazer nem assumir errado (ver §5, item 7).

**Contexto clínico inferido:** amostras oncológicas (sarcoma explícito; possível hematologia). Isso é
**dado de paciente** → ver privacidade em §6.

---

## 3. Pipeline proposto (por tecnologia — as duas diferem muito)

> Ion Torrent = reads curtos-médios (~200–400 bp), **erro característico de homopolímero** (indels
> espúrios), single-end. Nanopore = **long reads** (kb+), erro por-base mais alto (mas ótimo para SVs e
> fase). O caller e o alinhador **têm que ser específicos por plataforma** — não dá para tratar igual.

**Fase A — Inventário + QC (fazer PRIMEIRO, barato, decide o resto):**
- Catalogar: quais amostras têm Ion, quais têm Nanopore, **quais têm as duas** (isso decide se o
  "parear" cross-plataforma (2b) é viável). Conferir formato real (BAM vs FASTQ) do Ion.
- QC: **FastQC** (Ion) / **NanoPlot** ou **pycoQC** (Nanopore) — qualidade, distribuição de comprimento,
  adaptadores, nº de reads, estimativa de cobertura. Entender "o que está acontecendo" (item 3 do pedido).

**Fase B — Alinhamento ao hg38** (reusa `~/hg38/hg38.fa`):
- **Ion Torrent:** `bwa-mem2` (ou TMAP). Se os BAM já vierem alinhados a **hg19**, decidir: realinhar do
  FASTQ a hg38 (preferível) ou liftover (pior).
- **Nanopore:** `minimap2 -ax map-ont`.
- Pós: `samtools sort/index`; marcar duplicatas quando aplicável; `samtools flagstat` + **mosdepth**
  (cobertura por região — crítico se for painel: cobrimos os genes-alvo?).

**Fase C — Variant calling (vs hg38) → VCF:**
- **Ion Torrent:** um caller ciente de homopolímero — TVC (Torrent Variant Caller), ou GATK
  HaplotypeCaller, ou **DeepVariant** (modo apropriado).
- **Nanopore:** **Clair3** (SNV/indel) + **Sniffles2** (SVs estruturais).
- Normalizar os VCFs (left-align, `bcftools norm`) para a chave canônica do projeto.

**Fase D — "Parear" / comparar (se for a leitura 2b):**
- Mesma amostra em Ion e Nanopore → comparar VCFs (`bcftools isec`, **hap.py**) = concordância entre
  plataformas. É o resultado mais informativo se o objetivo é validação cruzada.

**Fase E — Anotação:**
- VEP / SnpEff / ANNOVAR; cruzar com **ClinVar** (já temos no projeto) para significado clínico.

---

## 4. Ambiente de execução (a montar)

- **Env bioconda dedicado** (NÃO tocar no `.venv` do Lumina). Esqueleto:
  `conda create -n seqlab -c bioconda -c conda-forge samtools bcftools minimap2 bwa-mem2 fastqc nanoplot
  mosdepth clair3 sniffles htslib` (+ DeepVariant via container, se usado). Validar por plataforma.
- **hg38:** reusar `~/hg38/hg38.fa` (+ gerar índices `.fai`/BWA/minimap2 uma vez).
- **Trânsito dos dados:** os arquivos estão no **Google Drive** do Eduardo. Precisam ir para o compute
  (notebook/instância) — via download + `aws s3 cp` para um bucket de trabalho, ou download direto no
  notebook. Volume total modesto (brutos ~130–180 MB × N amostras; clean poucos MB).
- **Compute:** pipeline de painel é leve (não precisa GPU). DeepVariant/Clair3 rodam melhor com GPU mas
  há modos CPU. Uma instância CPU decente basta para começar; dimensionar após o inventário.

---

## 5. Decisões abertas / perguntas para o Eduardo (bloqueiam o desenho fino)

1. **O que "parear" significa?** (a) alinhar ao hg38, (b) comparar Ion × Nanopore na mesma amostra, (c)
   tumor-normal? — **decisão #1**, muda o pipeline inteiro.
2. **Escopo de sequenciamento:** painel-alvo (qual painel/lista de genes?), exoma, ou genoma? (o tamanho
   dos FASTQ sugere **painel** — confirmar e obter o BED do painel, se houver.)
3. **Objetivo clínico/analítico:** diagnóstico (sarcoma / hemato)? variantes **somáticas** (tumor) ou
   **germinativas**? Isso muda o caller e os filtros.
4. **As mesmas amostras foram sequenciadas nas duas plataformas?** (viabiliza — ou não — a validação
   cruzada 2b.)
5. **Formato real dos arquivos Ion** (BAM alinhado / uBAM / FASTQ) e **a qual build** (hg19 vs hg38).
6. **Build de referência:** o pedido diz hg38 — confirmar (Ion historicamente hg19; se vier hg19,
   realinhar).
7. **O que o `_final_clean` fez** e **quem gerou** (trimming? filtro de qualidade? demultiplex?) — para
   não duplicar nem herdar um processamento desconhecido.
8. **Entregável esperado:** VCF anotado? relatório de QC + concordância? lista de variantes acionáveis?

---

## 6. Riscos / gotchas

1. **Homopolímero (Ion Torrent):** gera indels falsos — usar caller/filtros cientes disso; não comparar
   indels ingenuamente com Nanopore.
2. **Erro por-base (Nanopore):** precisa cobertura suficiente + caller adequado (Clair3); SNVs isolados
   de baixa cobertura são pouco confiáveis.
3. **Painel = cobertura localizada:** ótimo nos genes-alvo, **cego** no resto do genoma — não prometer
   achados genome-wide. Sem o BED do painel, a interpretação de "cobertura" fica incompleta.
4. **Build mismatch (hg19 vs hg38):** o maior risco silencioso. Coordenadas erradas se assumir a build
   errada. **Verificar antes de qualquer calling.**
5. **`_final_clean` opaco:** processamento anterior desconhecido pode ter viés (ex.: trimming agressivo,
   filtro que remove reads reais). Preferir, quando possível, partir do **bruto** e reproduzir a limpeza.
6. **PRIVACIDADE / LGPD (importante):** são **dados de paciente** de genética clínica. Não expor
   identificadores, não subir para buckets/serviços públicos, tratar os códigos de amostra como
   sensíveis. Confirmar com o Eduardo onde os dados podem residir e como podem ser compartilhados **antes**
   de mover qualquer arquivo.
7. **Não confiar em nome de arquivo:** confirmar plataforma/formato/build pelo **conteúdo** (header do
   BAM, primeiras linhas do FASTQ), não pelo nome — regra verify-over-doc herdada da outra frente.

---

## 7. Primeiros passos concretos (ordem sugerida)

1. **Alinhar expectativa com o Eduardo** nas 8 perguntas do §5 (sobretudo #1 "parear", #2 escopo, #3
   somático/germinativo, #5/#6 formato/build). Barato e evita retrabalho.
2. **Trazer 1–2 amostras** (idealmente uma que tenha Ion **e** Nanopore) para o compute e montar o **env
   bioconda**.
3. **Inventário + QC** dessas amostras (FastQC / NanoPlot; ver formato e build reais). Produz o primeiro
   "o que está acontecendo".
4. Só então **alinhar ao hg38** (bwa-mem2 / minimap2) e rodar um **calling piloto** (Clair3 / caller Ion)
   numa amostra — validar o fluxo ponta-a-ponta antes de escalar para todas.
5. Se 2b for o objetivo, **comparar Ion × Nanopore** na amostra piloto (concordância).

**O que NÃO fazer:** não misturar com a campanha R03/Lumina; não assumir painel/build/somático sem
confirmar; não mover dados de paciente sem alinhar privacidade; não tratar Ion e Nanopore com o mesmo
alinhador/caller.

---

*Fim do handoff. Estado em 2026-08-24: frente recém-aberta, nada executado. Dados no Drive do Eduardo,
ambiente bioinfo a montar, escopo a confirmar (§5). Próximo passo real: perguntas ao Eduardo + inventário/
QC de uma amostra piloto.*
