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
