# Lora ClinVar no SageMaker

Use este guia para lancar um fine-tuning LoRA de ClinVar no SageMaker.

## O que voce precisa ter

- Um terminal no ambiente Lumina que ja tenha o launcher `dispatch_clinvar_finetune_b200.sh`.
- AWS configurado na conta correta.
- `SAGEMAKER_ROLE` com permissao para SageMaker e S3.
- Checkpoint Lumina em S3 contendo `best_checkpoint.pt`.
- Dados em `s3://<bucket>/lumina-ssm/data/`.
- Quota disponivel para a instancia escolhida.

Sem o launcher Lumina disponivel no terminal, nao da para iniciar o job: o SageMaker precisa do codigo de treino.

## Setup local

```bash
uv sync --extra sagemaker --extra tracking

export AWS_DEFAULT_REGION=us-east-2
export SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/<sagemaker-role>
export WANDB_API_KEY=<wandb-key>
```

## Escolha da GPU

- H100: `ml.p5.48xlarge`
- H200: `ml.p5en.48xlarge`

## Lancar em H100

```bash
CHECKPOINT_DIR=s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/checkpoints/<checkpoint-name> \
EXPERIMENT=clinvar-lora-h100-test \
INSTANCE_TYPE=ml.p5.48xlarge \
MODEL_VERSION=beat-v1 \
WANDB_ENABLED=true \
bash scripts/dispatch_clinvar_finetune_b200.sh
```

## Lancar em H200

```bash
CHECKPOINT_DIR=s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/checkpoints/<checkpoint-name> \
EXPERIMENT=clinvar-lora-h200-test \
INSTANCE_TYPE=ml.p5en.48xlarge \
MODEL_VERSION=beat-v1 \
WANDB_ENABLED=true \
bash scripts/dispatch_clinvar_finetune_b200.sh
```

## Parametros que voce pode ajustar

```bash
CONTEXT_SIZE=4096
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16
MAX_EPOCHS=30
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
HEAD_HIDDEN_DIM=128
HEAD_DROPOUT=0.3
LR_BACKBONE=1e-5
LR_HEAD=1e-3
PRECISION=bf16
```

Exemplo com LoRA maior:

```bash
CHECKPOINT_DIR=s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/checkpoints/<checkpoint-name> \
EXPERIMENT=clinvar-lora-r32-h200 \
INSTANCE_TYPE=ml.p5en.48xlarge \
MODEL_VERSION=beat-v1 \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.0 \
WANDB_ENABLED=true \
bash scripts/dispatch_clinvar_finetune_b200.sh
```

## Onde ficam os resultados

```text
s3://<bucket>/lumina-ssm/eval/clinvar/finetune/<experiment>/
```

Arquivos principais:

- `metrics.json`
- `test_predictions.parquet`
- `best_model.pt`


## Erro comum: `Mamba3 is not available`

Se aparecer este erro:

```text
ImportError: Mamba3 is not available in the installed mamba-ssm package
Command "/usr/local/bin/python train_lora.py", exit code: 1
```

significa que o job usou a imagem diretamente e nao rodou o bootstrap Lumina. A imagem `hf-training-fa2-b200` sozinha pode nao expor `mamba_ssm.Mamba3`.

Use o wrapper abaixo, porque ele roda o setup do ambiente antes do treino:

```bash
bash scripts/dispatch_clinvar_finetune_b200.sh
```

Nao lance `train_lora.py` diretamente com `/usr/local/bin/python`. O fluxo correto cria uma `.venv`, compila/instala `mamba-ssm` recente a partir de `state-spaces/mamba` e so depois inicia o fine-tuning.

Se for obrigado a usar uma imagem/script proprio, a imagem precisa ter um `mamba-ssm` que exponha `mamba_ssm.Mamba3`, ou o script precisa instalar isso antes de importar o modelo Lumina.

## Checklist antes de rodar

- Troque `<checkpoint-name>` pelo checkpoint correto.
- Use um `EXPERIMENT` unico para cada tentativa.
- Confirme que `WANDB_API_KEY` existe se `WANDB_ENABLED=true`.
- Confirme quota da instancia H100/H200 na regiao.
