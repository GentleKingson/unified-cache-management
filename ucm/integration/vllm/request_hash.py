"""Request-aware block hashing shared by UCM vLLM connectors."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

try:
    from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
except ImportError:
    generate_block_hash_extra_keys = None

if TYPE_CHECKING:
    from vllm.v1.request import Request


UCM_CACHE_KEY_SCHEMA = "v2-multimodal"


def build_cache_key_namespace(
    vllm_config: Any,
    rank_id: int,
    namespace: str = "",
) -> bytes:
    """Build the static metadata prepended to every UCM request hash."""
    meta = (
        f"{UCM_CACHE_KEY_SCHEMA}:"
        f"{vllm_config.model_config.model}:"
        f"{vllm_config.parallel_config.tensor_parallel_size}:"
        f"{vllm_config.model_config.dtype}:"
        f"{rank_id}:{namespace}"
    )
    return meta.encode("utf-8")


def _request_has_semantic_extra_keys(request: "Request") -> bool:
    return (
        bool(getattr(request, "mm_features", None))
        or getattr(request, "lora_request", None) is not None
        or getattr(request, "cache_salt", None) is not None
        or getattr(request, "prompt_embeds", None) is not None
    )


def request_hashing_supported(request: "Request") -> bool:
    """Whether this vLLM version can safely hash the request semantics."""
    return (
        generate_block_hash_extra_keys is not None
        or not _request_has_semantic_extra_keys(request)
    )


def generate_block_extra_keys(
    request: "Request",
    start_token_idx: int,
    end_token_idx: int,
    start_mm_idx: int,
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate vLLM-compatible semantic keys for one request block."""
    if generate_block_hash_extra_keys is None:
        return None, start_mm_idx
    return generate_block_hash_extra_keys(
        request,
        start_token_idx,
        end_token_idx,
        start_mm_idx,
    )


def generate_request_block_hashes(
    request: "Request",
    block_size: int,
    parent_hash: bytes,
    hasher: Callable[[object], bytes],
) -> list[bytes]:
    """Hash all complete request blocks, including semantic request inputs.

    Older vLLM versions may not expose the extra-key helper. Such versions
    may still hash plain-text requests, but semantic requests must fail closed
    rather than reuse an unsafe token-only cache key.
    """
    if not request_hashing_supported(request):
        return []

    token_ids = request.all_token_ids
    hashes: list[bytes] = []
    start_mm_idx = 0
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        if len(block_token_ids) < block_size:
            break

        extra_keys, start_mm_idx = generate_block_extra_keys(
            request,
            start,
            end,
            start_mm_idx,
        )
        hash_value = hasher((parent_hash, tuple(block_token_ids), extra_keys))
        parent_hash = hash_value
        hashes.append(hash_value)

    return hashes
