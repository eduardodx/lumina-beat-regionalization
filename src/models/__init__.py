# --- tilelang/tvm workaround (SageMaker notebook conda env) ---------------------------------
# The notebook's conda env ships a tilelang/tvm build that aborts on import under Python 3.12
# (`tvm_ffi AttributeError: attribute '__dict__' of 'type' objects is not writable`). mamba_ssm
# imports tilelang eagerly for the Mamba3 MIMO kernel, so importing the model stack crashes here
# before any per-script shim can run. Poisoning sys.modules makes `import tilelang` raise
# ImportError, which mamba_ssm catches to fall back to Triton (the r1 checkpoint is SISO, so the
# MIMO/tilelang path is unused anyway). On the SageMaker training job tilelang is simply not
# installed (INSTALL_TILELANG=0), where this is a harmless no-op. Remove once the env is fixed.
import sys as _sys
import types as _types

if not isinstance(_sys.modules.get("tilelang"), _types.ModuleType):
    _sys.modules["tilelang"] = None
# --------------------------------------------------------------------------------------------

from src.models.beat_v2 import BeatV2Config, DNAFoundationBeatV2
from src.models.beat_v3 import BeatV3Config, DNAFoundationBeatV3
from src.models.beat_v4 import BeatV4Config, DNAFoundationBeatV4
from src.models.beat_v5 import BeatV5Config, DNAFoundationBeatV5
from src.models.beat_v6 import BeatV6Config, DNAFoundationBeatV6
from src.models.beat_v7 import BeatV7Config, DNAFoundationBeatV7
from src.models.beat_v10 import BeatV10Config, DNAFoundationBeatV10
from src.models.bimamba import BiMambaConfig, DNAFoundationBiMamba, default_use_mem_eff_path
from src.models.bimamba3 import BiMamba3Config, DNAFoundationBiMamba3
from src.models.bimamba3_rc import BiMamba3RCConfig, DNAFoundationBiMamba3RC
from src.models.registry import (
    DEFAULT_MODEL_KEY,
    REGISTERED_MODELS,
    ModelSpec,
    build_registered_model,
    get_model_spec,
    normalize_model_key,
    registered_model_keys,
    resolve_model_config,
    resolve_model_config_dict,
)

__all__ = [
    "DEFAULT_MODEL_KEY",
    "REGISTERED_MODELS",
    "BeatV2Config",
    "BeatV3Config",
    "BeatV4Config",
    "BeatV5Config",
    "BeatV6Config",
    "BeatV7Config",
    "BeatV10Config",
    "BiMamba3Config",
    "BiMamba3RCConfig",
    "BiMambaConfig",
    "DNAFoundationBeatV2",
    "DNAFoundationBeatV3",
    "DNAFoundationBeatV4",
    "DNAFoundationBeatV5",
    "DNAFoundationBeatV6",
    "DNAFoundationBeatV7",
    "DNAFoundationBeatV10",
    "DNAFoundationBiMamba",
    "DNAFoundationBiMamba3",
    "DNAFoundationBiMamba3RC",
    "ModelSpec",
    "build_registered_model",
    "default_use_mem_eff_path",
    "get_model_spec",
    "normalize_model_key",
    "registered_model_keys",
    "resolve_model_config",
    "resolve_model_config_dict",
]
