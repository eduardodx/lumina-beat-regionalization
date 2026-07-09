# Configurations

This directory contains training config families for Lumina experiments.

## Current Focus

- `beat_v5/`: active documented family

## Other Families

- `beat_v4/`: predecessor compact family retained for reproducibility and ablations
- `beat_v3/`: scaled experimental family retained for reproducibility
- `beat_v2/`: prior compact consequence-aware family retained for reproducibility
- `beat_v1/`: earlier compact family retained for older checkpoints and launchers
- `bimamba/`, `bimamba3/`, `bimamba3_rc/`: baseline architecture families and earlier configuration references

## Layout Conventions

- `_base.yaml`: family defaults
- concrete `*.yaml`: runnable configs that inherit from the family base
- top-level `_base.yaml`: shared repository defaults

## Recommended Starting Point

For current config-driven SageMaker launches, start from:

- `beat_v5/384w_8l_15ep_32k.yaml`
