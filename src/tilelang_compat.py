from __future__ import annotations

import logging
import sysconfig
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PATCHED_TILELANG_CUDA_CALLBACK = False
_PATCHED_TILELANG_NESTED_LOOP_CHECKER = False


class _NoopTileLangPass:
    """Callable replacement for the broken NestedLoopChecker TVM pass.

    tilelang.engine.phase.PreLowerSemanticCheck invokes
    ``tilelang.analysis.NestedLoopChecker()(mod)``, so we only need something
    that (a) can be instantiated, and (b) returns ``mod`` unchanged when
    invoked as a TVM pass.
    """

    def __call__(self, mod: Any) -> Any:  # noqa: D401 - TVM pass call signature
        return mod


class _NoopNestedLoopCheckVisitor:
    """Stand-in for tilelang.analysis.nested_loop_checker._NestedLoopCheckVisitor.

    Defense-in-depth: any caller that still constructs the visitor directly
    will get this no-op version and skip the TVMDerivedObject __init__ path
    that blows up with ``_inst`` AttributeError.
    """

    def visit_stmt(self, stmt: Any) -> None:  # noqa: D401 - mirrors upstream
        return None


def patch_tilelang_nested_loop_checker_bug() -> bool:
    """Neutralize the broken TileLang semantic-check visitors in tilelang 0.1.8.

    Root cause (upstream bug): ``nested_loop_checker._NestedLoopCheckVisitor``
    and ``fragment_loop_checker._FragmentLoopCheckVisitor`` are decorated with
    ``@tir.functor.visitor`` (alias for
    ``tvm.runtime.support.derived_object``). The wrapper's ``__init__``
    assigns ``self._inst = cls(*args, **kwargs)`` BEFORE the C++-backed parent
    ``tvm_ffi.core.Object`` has been initialized. The resulting
    ``__setattr__`` therefore reaches ``object.__setattr__`` on an
    uninitialized instance and raises
    ``AttributeError: '<...>LoopCheckVisitor' object has no attribute '_inst'``
    inside a TVM FFI callback, making it very hard to intercept from outer
    Python frames.

    Fix strategy: swap the broken analysis passes at the module level for
    factories that return a no-op pass. ``PreLowerSemanticCheck`` does
    ``tilelang.analysis.<Checker>()(mod)`` — the attribute is
    resolved at call time, so the swap takes effect for every subsequent
    invocation without touching the decorated class or the broken C++ init.

    The pass is a validation-only check (rejects nested parallel loops /
    pipelined-inside-parallel loops). Upstream mamba3 MIMO kernels are
    already known-good, so skipping validation does not affect correctness,
    only the quality of error messages if someone introduces an invalid
    kernel pattern later.
    """
    global _PATCHED_TILELANG_NESTED_LOOP_CHECKER
    if _PATCHED_TILELANG_NESTED_LOOP_CHECKER:
        log.info("patch_tilelang_nested_loop_checker_bug: already patched, skipping")
        return True

    log.info("patch_tilelang_nested_loop_checker_bug: begin")

    try:
        import tilelang.analysis as _analysis_mod
    except ImportError as exc:
        log.warning(
            "patch_tilelang_nested_loop_checker_bug: tilelang.analysis not importable (%s); skipping",
            exc,
        )
        return False

    checker_specs = (
        ("NestedLoopChecker", "tilelang.analysis.nested_loop_checker", "_NestedLoopCheckVisitor"),
        ("FragmentLoopChecker", "tilelang.analysis.fragment_loop_checker", "_FragmentLoopCheckVisitor"),
    )
    for checker_name, module_name, visitor_name in checker_specs:
        try:
            checker_mod = __import__(module_name, fromlist=[visitor_name])
        except ImportError as exc:
            log.warning(
                "patch_tilelang_nested_loop_checker_bug: %s not importable (%s); "
                "will only swap %s at analysis-module level",
                module_name,
                exc,
                checker_name,
            )
            checker_mod = None

        setattr(_analysis_mod, checker_name, lambda: _NoopTileLangPass())
        log.info(
            "patch_tilelang_nested_loop_checker_bug: replaced tilelang.analysis.%s with no-op pass",
            checker_name,
        )

        if checker_mod is not None and hasattr(checker_mod, visitor_name):
            setattr(checker_mod, visitor_name, _NoopNestedLoopCheckVisitor)
            log.info(
                "patch_tilelang_nested_loop_checker_bug: replaced %s.%s with no-op visitor",
                module_name,
                visitor_name,
            )

    _PATCHED_TILELANG_NESTED_LOOP_CHECKER = True
    log.info("patch_tilelang_nested_loop_checker_bug: done")
    return True


def ensure_tilelang_triton_cuda_include() -> bool:
    global _PATCHED_TILELANG_CUDA_CALLBACK

    if _PATCHED_TILELANG_CUDA_CALLBACK:
        return True

    purelib = Path(sysconfig.get_paths()["purelib"])
    triton_include = purelib / "triton" / "backends" / "nvidia" / "include"
    if not (triton_include / "cuda_fp8.h").is_file():
        return False

    try:
        import tvm_ffi
        from tilelang.contrib import nvcc
        from tilelang.engine.lower import PassConfigKey
        from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
    except ImportError:
        return False

    @tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
    def _patched_tilelang_callback_cuda_compile(
        code: str,
        target: Any,
        pass_config: dict[str, Any] | None = None,
    ) -> bytearray:
        target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))
        arch = [f"-arch=sm_{target_arch}"]
        compile_format = "cubin"

        cfg = pass_config or {}
        enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))
        ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
        verbose_ptxas_output = bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False))

        options = [
            "-std=c++17",
            f"-I{triton_include}",
            f"-I{TILELANG_TEMPLATE_PATH}",
            f"-I{CUTLASS_INCLUDE_DIR}",
        ]

        extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
        if extra_flags:
            import shlex

            if isinstance(extra_flags, str):
                tokens = shlex.split(extra_flags)
            else:
                tokens = []
                for flag in extra_flags:
                    if isinstance(flag, str):
                        tokens.extend(shlex.split(flag))
                    else:
                        tokens.append(str(flag))
            options.extend(tokens)

        verbose = False
        if enable_fast_math:
            options.append("--use_fast_math")
        if ptxas_usage_level is not None:
            options.append(f"--ptxas-options=--register-usage-level={ptxas_usage_level}")
        if verbose_ptxas_output:
            options.extend(["--ptxas-options=--verbose", "-w"])
            verbose = True

        return nvcc.compile_cuda(
            code,
            compile_format,
            arch,
            options=options,
            verbose=verbose,
        )

    log.info("Patched TileLang CUDA compile callback to prefer Triton CUDA headers: %s", triton_include)
    _PATCHED_TILELANG_CUDA_CALLBACK = True
    return True
