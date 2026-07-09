# SageMaker Operations

## Canonical Training Entry Point

For bootstrapping the shared SageMaker Studio domain itself, see
[SageMaker Domain Provisioning](sagemaker-domain.md).

The canonical training launcher is config-driven:

```bash
uv run python scripts/sagemaker_train.py \
  --experiment my-experiment \
  --config configs/beat_v5/384w_8l_15ep_32k.yaml \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p6-b200.48xlarge \
  --spot \
  --detach
```

Key points:

- training data is expected under `s3://<bucket>/lumina-ssm/data/`
- experiment outputs are written under `s3://<bucket>/lumina-ssm/experiments/<experiment>/`
- checkpoints are written under `s3://<bucket>/lumina-ssm/checkpoints/<experiment>/`

Wrapper scripts such as `scripts/dispatch_beat_v1_b200.sh` are convenience or legacy launchers, not the canonical
interface.

## Quickstart H100/H200

Use este fluxo para iniciar um pretraining simples pelo launcher canonico. Evite os wrappers `dispatch_*` para novos runs, salvo quando o wrapper for explicitamente parte do experimento.

1. Instale as dependencias locais do launcher:

```bash
uv sync --extra sagemaker --extra tracking
```

2. Configure credenciais e variaveis, via `.env` ou shell:

```bash
export AWS_DEFAULT_REGION=us-east-2
export SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/<sagemaker-role>
export WANDB_API_KEY=<wandb-key>
```

3. Garanta que os dados estao no layout esperado em S3:

```bash
aws s3 sync data/ s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/
```

4. Escolha a instancia e confirme quota/capacidade na regiao alvo:

- H100: `ml.p5.48xlarge`
- H200: `ml.p5en.48xlarge`; `ml.p5e.48xlarge` tambem aparece nas quotas de SageMaker, mas os configs/scripts atuais deste repo usam `p5en` para H200

5. Antes de disparar, commit os arquivos que o container precisa. O pacote enviado ao SageMaker vem de `git archive HEAD`.

6. Lance o job com `--detach` para submeter e liberar o terminal.

H100:

```bash
uv run python scripts/sagemaker_train.py \
  --experiment beat-v5-h100-test \
  --config configs/beat_v5/384w_8l_15ep_32k.yaml \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p5.48xlarge \
  --spot \
  --detach
```

H200:

```bash
uv run python scripts/sagemaker_train.py \
  --experiment beat-v5-h200-test \
  --config configs/beat_v5/384w_8l_15ep_32k.yaml \
  --bucket ai4bio-lumina-experiments-v2 \
  --instance-type ml.p5en.48xlarge \
  --spot \
  --detach
```

7. Monitore pelo link impresso no terminal ou pelo console SageMaker. Os artefatos ficam em:

```text
s3://<bucket>/lumina-ssm/experiments/<experiment>/
s3://<bucket>/lumina-ssm/checkpoints/<experiment>/
```

Notas rapidas:

- `configs/beat_v5/384w_8l_15ep_32k.yaml` esta ajustado para H200/p5en com `batch_size: 7` e `grad_accum_steps: 4`.
- Em H100, se houver OOM, crie um novo config com `batch_size` menor e mantenha `grad_accum_steps` explicito para preservar comparabilidade.
- `--spot` reduz custo, mas pode sofrer interrupcao; o launcher configura checkpoint sync em `s3://<bucket>/lumina-ssm/checkpoints/<experiment>/`.
- Para `beat-v7`, o launcher ativa instalacao de `flash-attn` automaticamente no container.

## Source Packaging Rule

SageMaker source packaging uses `git archive HEAD`.

That means:

- only committed files are packaged into the training source bundle
- uncommitted config edits, new launcher scripts, and local-only docs will not reach the container

Commit launch-critical files before dispatching a job.

## Training Data Upload

Populate the standard S3 layout before dispatch:

```bash
aws s3 sync data/ s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/
```

## ClinVar Fine-Tuning On SageMaker

The dedicated ClinVar SageMaker entry point already exists:

```bash
CHECKPOINT_DIR=/opt/ml/input/data/training/checkpoints/<checkpoint-name> \
bash scripts/dispatch_clinvar_finetune_b200.sh
```

For remote checkpoint staging, `CHECKPOINT_DIR` may also be an S3 prefix under
`s3://<bucket>/lumina-ssm/data/checkpoints/...`.

The launcher resolves the Lumina pretraining checkpoint as:

```text
<checkpoint-dir>/best_checkpoint.pt
```

The ClinVar run writes:

- `metrics.json`
- `test_predictions.parquet`
- `best_model.pt`

to:

```text
s3://<bucket>/lumina-ssm/eval/clinvar/finetune/<experiment>/
```

## Environment Notes

- `SAGEMAKER_ROLE` and `WANDB_API_KEY` may be supplied through `.env` or the environment
- the default production instance types in this repo target A100 or B200 environments
- Linux/CUDA is the canonical environment for distributed training and published evaluation runs
