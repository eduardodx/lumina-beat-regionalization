from __future__ import annotations

import pytest

from eval.clinvar.config import FineTuneConfig


def test_config_accepts_dynamic_fusion_and_rejects_submitter_features() -> None:
    config = FineTuneConfig(
        init_finetuned_checkpoint_path="best_model.pt",
        fusion_mode="dynamic_lora",
        fusion_adapter_paths=["abraom.pt", "gnomad.pt"],
        fusion_adapter_names=["abraom", "gnomad"],
    )
    config.validate()

    leaked = FineTuneConfig(
        head_type="regime_a_plus_features",
        explicit_feature_columns=["log10_af_abraom", "has_brazilian_submitter"],
    )
    with pytest.raises(ValueError, match="submitter/provenance"):
        leaked.validate()
