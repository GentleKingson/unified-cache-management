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
UCM_MM_EXTRA_KEY_SCHEMA = b"ucm-mm-position-v1"


class _RequestWithoutMM:
    """Delegate request attributes while hiding multimodal features."""

    def __init__(self, request: Any):
        self._request = request
        self.mm_features: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)


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
    mm_features = getattr(request, "mm_features", None) or ()
    if any(getattr(feature, "identifier", None) is None for feature in mm_features):
        return False
    return (
        generate_block_hash_extra_keys is not None
        or not _request_has_semantic_extra_keys(request)
    )


def _generate_ucm_mm_extra_keys(
    request: "Request",
    start_token_idx: int,
    end_token_idx: int,
    start_mm_idx: int,
) -> tuple[list[tuple[Any, ...]], int]:
    """Generate UCM-owned, position-aware keys for overlapping MM inputs."""
    extra_keys: list[tuple[Any, ...]] = []
    mm_features = getattr(request, "mm_features", None) or ()
    if not mm_features:
        return extra_keys, start_mm_idx

    # vLLM Request keeps MM features sorted by mm_position.offset. This lets
    # the returned index resume the scan for the next token block.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # Match vLLM's -1 convention for resuming from the final MM input.
    if start_mm_idx < 0:
        if -start_mm_idx > len(mm_features):
            raise IndexError("start_mm_idx is outside request.mm_features")
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        if mm_feature.identifier is None:
            raise ValueError("multimodal feature identifier must not be None")

        position = mm_feature.mm_position
        offset = position.offset
        length = position.length
        if start_token_idx >= offset + length:
            curr_mm_idx += 1
            continue
        if offset >= end_token_idx:
            break

        extra_keys.append(
            (
                UCM_MM_EXTRA_KEY_SCHEMA,
                mm_feature.identifier,
                offset - start_token_idx,
                length,
            )
        )
        if offset + length <= end_token_idx:
            curr_mm_idx += 1
        else:
            break

    return extra_keys, curr_mm_idx


def generate_block_extra_keys(
    request: "Request",
    start_token_idx: int,
    end_token_idx: int,
    start_mm_idx: int,
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate normalized semantic keys for one request block."""
    if generate_block_hash_extra_keys is None:
        return None, start_mm_idx

    ucm_mm_keys, new_start_mm_idx = _generate_ucm_mm_extra_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    non_mm_keys, _ = generate_block_hash_extra_keys(
        _RequestWithoutMM(request),
        start_token_idx,
        end_token_idx,
        start_mm_idx,
    )
    # UCM MM keys always precede vLLM's non-MM keys. The local schema isolates
    # identifier-only MM entries without invalidating text-only keys already
    # generated under the global v2 schema.
    combined = tuple(ucm_mm_keys) + tuple(non_mm_keys or ())
    return combined or None, new_start_mm_idx


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
    block_inputs: list[tuple[tuple[int, ...], tuple[Any, ...] | None]] = []
    start_mm_idx = 0
    try:
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
            block_inputs.append((tuple(block_token_ids), extra_keys))
    except Exception:
        # A request must never expose a partially hashed semantic prefix.
        return []

    hashes: list[bytes] = []
    for block_token_ids, extra_keys in block_inputs:
        hash_value = hasher((parent_hash, tuple(block_token_ids), extra_keys))
        parent_hash = hash_value
        hashes.append(hash_value)

    return hashes
