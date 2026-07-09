from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypeGuard, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

PrecisionMode = Literal["auto", "mxfp8", "bf16", "fp32"]
ModuleFilter = Callable[[str, nn.Module], bool]

BLACKWELL_MIN_COMPUTE_CAPABILITY = (10, 0)
TRANSFORMER_ENGINE_BACKEND = "transformer_engine"
MXFP8_RECIPE_CANDIDATES = (
    "MXFP8BlockScaling",
    "MXFP8CurrentScaling",
    "Float8CurrentScaling",
)
MXFP8_FORMAT_CANDIDATES = ("HYBRID", "E4M3", "E4M3_E5M2")


@dataclass(frozen=True)
class TransformerEngineRuntime:
    backend: str
    linear_cls: type[nn.Module]
    fp8_autocast: Callable[..., Any]
    recipe: Any


class LinearLikeModule(Protocol):
    weight: torch.Tensor
    bias: torch.Tensor | None
    in_features: int
    out_features: int

    def __call__(self, input: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class PrecisionPolicy:
    requested: PrecisionMode
    resolved: PrecisionMode
    use_autocast: bool
    autocast_device_type: str | None = None
    autocast_dtype: torch.dtype | None = None
    fallback_reason: str | None = None
    fp8_enabled: bool = False
    fp8_backend: str | None = None
    fp8_module_count: int = 0
    fp8_runtime: TransformerEngineRuntime | None = field(default=None, repr=False, compare=False)

    @property
    def uses_bf16_compute(self) -> bool:
        return (
            self.use_autocast
            and self.autocast_device_type == "cuda"
            and self.autocast_dtype == torch.bfloat16
        )


def configure_float32_precision(allow_tf32: bool) -> None:
    matmul_precision = "high" if allow_tf32 else "highest"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _call_with_supported_kwargs(callable_obj: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(*args, **kwargs)

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(*args, **filtered_kwargs)


def _build_mxfp8_recipe(recipe_module: Any) -> Any | None:
    format_enum = getattr(recipe_module, "Format", None)
    fp8_format = None
    if format_enum is not None:
        for name in MXFP8_FORMAT_CANDIDATES:
            if hasattr(format_enum, name):
                fp8_format = getattr(format_enum, name)
                break

    for recipe_name in MXFP8_RECIPE_CANDIDATES:
        recipe_cls = getattr(recipe_module, recipe_name, None)
        if recipe_cls is None:
            continue

        kwargs = {
            "margin": 0,
            "interval": 1,
            "amax_history_len": 16,
            "amax_compute_algo": "max",
            "fp8_format": fp8_format,
        }
        try:
            return _call_with_supported_kwargs(recipe_cls, **kwargs)
        except Exception:
            continue
    return None


def _load_transformer_engine_runtime() -> TransformerEngineRuntime | None:
    try:
        te_pytorch = importlib.import_module("transformer_engine.pytorch")
        te_recipe = importlib.import_module("transformer_engine.common.recipe")
    except ImportError:
        return None

    linear_cls = getattr(te_pytorch, "Linear", None)
    fp8_autocast = getattr(te_pytorch, "fp8_autocast", None)
    if linear_cls is None or fp8_autocast is None:
        return None

    recipe = _build_mxfp8_recipe(te_recipe)
    if recipe is None:
        return None

    return TransformerEngineRuntime(
        backend=TRANSFORMER_ENGINE_BACKEND,
        linear_cls=linear_cls,
        fp8_autocast=fp8_autocast,
        recipe=recipe,
    )


def _resolve_mxfp8_runtime(device: torch.device) -> tuple[TransformerEngineRuntime | None, str | None]:
    if device.type != "cuda":
        return None, f"non_cuda_device:{device.type}"

    if not torch.cuda.is_bf16_supported():
        return None, "cuda_bf16_unsupported"

    device_index = _device_index(device)
    major, minor = torch.cuda.get_device_capability(device_index)
    if (major, minor) < BLACKWELL_MIN_COMPUTE_CAPABILITY:
        return None, f"cuda_compute_capability_{major}_{minor}_below_blackwell"

    runtime = _load_transformer_engine_runtime()
    if runtime is None:
        return None, "transformer_engine_mxfp8_unavailable"
    return runtime, None


def resolve_precision_policy(device: torch.device, requested: PrecisionMode) -> PrecisionPolicy:
    if requested == "fp32":
        return PrecisionPolicy(requested=requested, resolved="fp32", use_autocast=False)

    if requested == "mxfp8":
        runtime, reason = _resolve_mxfp8_runtime(device)
        if runtime is None:
            raise RuntimeError(
                "mxfp8 precision was requested, but the active runtime does not support it "
                f"({reason or 'unknown_reason'})."
            )
        return PrecisionPolicy(
            requested=requested,
            resolved="mxfp8",
            use_autocast=True,
            autocast_device_type="cuda",
            autocast_dtype=torch.bfloat16,
            fp8_enabled=True,
            fp8_backend=runtime.backend,
            fp8_runtime=runtime,
        )

    if device.type != "cuda":
        if requested == "bf16":
            raise RuntimeError("bf16 precision requires CUDA with native bf16 support.")
        return PrecisionPolicy(
            requested=requested,
            resolved="fp32",
            use_autocast=False,
            fallback_reason=f"non_cuda_device:{device.type}",
        )

    bf16_supported = torch.cuda.is_bf16_supported()

    if requested == "bf16":
        if not bf16_supported:
            raise RuntimeError("bf16 precision was requested, but the active CUDA device does not support it.")
        return PrecisionPolicy(
            requested=requested,
            resolved="bf16",
            use_autocast=True,
            autocast_device_type="cuda",
            autocast_dtype=torch.bfloat16,
        )

    runtime, reason = _resolve_mxfp8_runtime(device)
    if runtime is not None:
        return PrecisionPolicy(
            requested=requested,
            resolved="mxfp8",
            use_autocast=True,
            autocast_device_type="cuda",
            autocast_dtype=torch.bfloat16,
            fp8_enabled=True,
            fp8_backend=runtime.backend,
            fp8_runtime=runtime,
        )

    if bf16_supported:
        return PrecisionPolicy(
            requested=requested,
            resolved="bf16",
            use_autocast=True,
            autocast_device_type="cuda",
            autocast_dtype=torch.bfloat16,
            fallback_reason=reason,
        )

    return PrecisionPolicy(
        requested=requested,
        resolved="fp32",
        use_autocast=False,
        fallback_reason=reason or "cuda_bf16_unsupported",
    )


class MXFP8Linear(nn.Linear):
    def __init__(self, base: nn.Linear, runtime: TransformerEngineRuntime) -> None:
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if self.bias is not None and base.bias is not None:
                self.bias.copy_(base.bias)

        self.weight.requires_grad = base.weight.requires_grad
        if self.bias is not None and base.bias is not None:
            self.bias.requires_grad = base.bias.requires_grad

        self.__dict__["_fp8_backend"] = runtime.backend
        self.__dict__["_fp8_impl"] = self._build_fp8_impl(runtime)

    def _build_fp8_impl(self, runtime: TransformerEngineRuntime) -> nn.Module | None:
        init_kwargs = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias is not None,
            "params_dtype": self.weight.dtype,
            "device": self.weight.device,
        }
        try:
            fp8_impl = _call_with_supported_kwargs(runtime.linear_cls, **init_kwargs)
        except Exception:
            return None

        if isinstance(fp8_impl, nn.Module):
            weight = cast(nn.Parameter, self.weight)
            bias = cast(nn.Parameter | None, self.bias)
            if hasattr(fp8_impl, "_parameters"):
                fp8_impl._parameters["weight"] = weight
                fp8_impl._parameters["bias"] = bias
            else:
                fp8_impl.weight = weight
                if bias is not None:
                    fp8_impl.bias = bias
            fp8_impl.to(device=self.weight.device)
            return fp8_impl
        return None

    @property
    def fp8_backend(self) -> str:
        return str(self.__dict__.get("_fp8_backend", TRANSFORMER_ENGINE_BACKEND))

    @property
    def is_fp8_linear(self) -> bool:
        return True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        fp8_impl = self.__dict__.get("_fp8_impl")
        if fp8_impl is None:
            return F.linear(input, self.weight, self.bias)
        output = fp8_impl(input)
        if isinstance(output, tuple):
            return output[0]
        return output


def is_linear_like_module(module: nn.Module) -> TypeGuard[LinearLikeModule]:
    required_attributes = ("weight", "bias", "in_features", "out_features")
    return isinstance(module, nn.Linear) or all(hasattr(module, attr) for attr in required_attributes)


def _replace_named_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def apply_fp8_linear_replacements(
    root: nn.Module,
    policy: PrecisionPolicy,
    module_filter: ModuleFilter,
) -> PrecisionPolicy:
    if not policy.fp8_enabled or policy.fp8_runtime is None:
        return policy

    targets: list[str] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, MXFP8Linear):
            continue
        if module_filter(name, module):
            targets.append(name)

    for name in targets:
        module = root.get_submodule(name)
        if not isinstance(module, nn.Linear):
            continue
        _replace_named_module(root, name, MXFP8Linear(module, policy.fp8_runtime))

    if not targets:
        if policy.requested == "mxfp8":
            raise RuntimeError("mxfp8 precision was requested, but no eligible linear modules were converted.")
        return replace(
            policy,
            resolved="bf16" if policy.uses_bf16_compute else "fp32",
            fallback_reason="no_eligible_fp8_linear_modules",
            fp8_enabled=False,
            fp8_backend=None,
            fp8_module_count=0,
            fp8_runtime=None,
        )

    return replace(policy, fp8_module_count=len(targets))


def precision_metadata(policy: PrecisionPolicy) -> dict[str, Any]:
    return {
        "requested_precision": policy.requested,
        "resolved_precision": policy.resolved,
        "fp8_enabled": policy.fp8_enabled,
        "fp8_backend": policy.fp8_backend,
        "fp8_module_count": policy.fp8_module_count,
        "fallback_reason": policy.fallback_reason,
        "autocast_dtype": (
            str(policy.autocast_dtype).replace("torch.", "") if policy.autocast_dtype is not None else None
        ),
    }


def precision_log_string(policy: PrecisionPolicy, *, allow_tf32: bool) -> str:
    pieces = [
        f"requested_precision={policy.requested}",
        f"resolved_precision={policy.resolved}",
        f"allow_tf32={allow_tf32}",
        f"fp8_enabled={policy.fp8_enabled}",
    ]
    if policy.fp8_backend is not None:
        pieces.append(f"fp8_backend={policy.fp8_backend}")
    if policy.fp8_enabled:
        pieces.append(f"fp8_module_count={policy.fp8_module_count}")
    if policy.fallback_reason is not None:
        pieces.append(f"fallback_reason={policy.fallback_reason}")
    return " ".join(pieces)


@contextmanager
def autocast_context(policy: PrecisionPolicy) -> Any:
    if not policy.use_autocast or policy.autocast_device_type is None or policy.autocast_dtype is None:
        with nullcontext():
            yield
        return

    with ExitStack() as stack:
        stack.enter_context(torch.autocast(device_type=policy.autocast_device_type, dtype=policy.autocast_dtype))
        if policy.fp8_enabled and policy.fp8_runtime is not None:
            stack.enter_context(
                _call_with_supported_kwargs(
                    policy.fp8_runtime.fp8_autocast,
                    enabled=True,
                    fp8_recipe=policy.fp8_runtime.recipe,
                )
            )
        yield
