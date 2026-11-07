from __future__ import annotations

import importlib.metadata
import os
from typing import Any

DETERMINISM_ENV_VARS = (
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_BATCH_INVARIANT",
    "VLLM_USE_V1",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def configure_vllm_determinism_env() -> None:
    """Apply repo defaults before vLLM is imported."""

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")


def vllm_determinism_metadata() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "schema": "cot_loop_vllm_determinism.v1",
        "vllm_version": version,
        "env": {name: os.environ.get(name) for name in DETERMINISM_ENV_VARS},
    }


def assert_vllm_determinism_env(*, require_single_process_v1: bool = True) -> dict[str, Any]:
    metadata = vllm_determinism_metadata()
    env = metadata["env"]
    if require_single_process_v1 and env.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError(
            "VLLM_ENABLE_V1_MULTIPROCESSING must be 0 for deterministic "
            f"offline replay in this repo: {metadata}"
        )
    batch_invariant = env.get("VLLM_BATCH_INVARIANT")
    if batch_invariant not in {"0", "1"}:
        raise RuntimeError(
            "VLLM_BATCH_INVARIANT must be recorded explicitly as 0 or 1: "
            f"{metadata}"
        )
    return metadata


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def inspect_vllm_engine_config(
    llm: Any,
    *,
    expected_max_model_len: int | None = None,
    expected_max_num_seqs: int | None = None,
    expected_max_num_batched_tokens: int | None = None,
    expected_enable_prefix_caching: bool | None = None,
    expected_enforce_eager: bool | None = None,
    expected_attention_backend: str | None = None,
    require_no_speculative: bool = True,
    require_single_process_v1: bool = True,
) -> dict[str, Any]:
    engine = llm.llm_engine
    vllm_config = getattr(engine, "vllm_config", None)
    model_config = getattr(engine, "model_config", None) or getattr(vllm_config, "model_config", None)
    scheduler_config = getattr(engine, "scheduler_config", None) or getattr(vllm_config, "scheduler_config", None)
    cache_config = getattr(engine, "cache_config", None) or getattr(vllm_config, "cache_config", None)
    attention_config = getattr(engine, "attention_config", None) or getattr(vllm_config, "attention_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    missing = [
        name
        for name, value in (
            ("model_config", model_config),
            ("scheduler_config", scheduler_config),
            ("cache_config", cache_config),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(f"Could not inspect vLLM engine config fields {missing}; engine={type(engine)!r}")

    actual = {
        "max_model_len": int(getattr(model_config, "max_model_len")),
        "max_num_seqs": int(getattr(scheduler_config, "max_num_seqs")),
        "max_num_batched_tokens": _optional_int(
            getattr(scheduler_config, "max_num_batched_tokens", None)
        ),
        "enable_prefix_caching": bool(getattr(cache_config, "enable_prefix_caching", True)),
        "enforce_eager": bool(getattr(model_config, "enforce_eager", False)),
        "attention_backend": None,
        "speculative_config_set": speculative_config is not None,
        "speculative_config_repr": None if speculative_config is None else repr(speculative_config),
        "determinism": assert_vllm_determinism_env(
            require_single_process_v1=require_single_process_v1
        ),
    }
    if attention_config is not None:
        backend = getattr(attention_config, "backend", None)
        if backend is not None:
            actual["attention_backend"] = getattr(backend, "name", str(backend))
    env = actual["determinism"]["env"]
    actual["vllm_enable_v1_multiprocessing"] = env.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    actual["vllm_batch_invariant"] = env.get("VLLM_BATCH_INVARIANT")

    if expected_max_model_len is not None and actual["max_model_len"] != int(expected_max_model_len):
        raise RuntimeError(f"vLLM max_model_len mismatch: {actual['max_model_len']} != {expected_max_model_len}")
    if expected_max_num_seqs is not None and actual["max_num_seqs"] != int(expected_max_num_seqs):
        raise RuntimeError(f"vLLM max_num_seqs mismatch: {actual['max_num_seqs']} != {expected_max_num_seqs}")
    if (
        expected_max_num_batched_tokens is not None
        and actual["max_num_batched_tokens"] != int(expected_max_num_batched_tokens)
    ):
        raise RuntimeError(
            "vLLM max_num_batched_tokens mismatch: "
            f"{actual['max_num_batched_tokens']} != {expected_max_num_batched_tokens}"
        )
    if (
        expected_enable_prefix_caching is not None
        and actual["enable_prefix_caching"] != bool(expected_enable_prefix_caching)
    ):
        raise RuntimeError(
            "vLLM enable_prefix_caching mismatch: "
            f"{actual['enable_prefix_caching']} != {bool(expected_enable_prefix_caching)}"
        )
    if expected_enforce_eager is not None and actual["enforce_eager"] != bool(expected_enforce_eager):
        raise RuntimeError(f"vLLM enforce_eager mismatch: {actual['enforce_eager']} != {bool(expected_enforce_eager)}")
    if expected_attention_backend is not None:
        expected_backend = str(expected_attention_backend).upper()
        actual_backend = (
            None
            if actual["attention_backend"] is None
            else str(actual["attention_backend"]).upper()
        )
        if actual_backend != expected_backend:
            raise RuntimeError(
                "vLLM attention_backend mismatch: "
                f"{actual['attention_backend']} != {expected_attention_backend}"
            )
    if require_no_speculative and speculative_config is not None:
        raise RuntimeError(f"vLLM speculative_config must be None: {speculative_config!r}")
    return actual
