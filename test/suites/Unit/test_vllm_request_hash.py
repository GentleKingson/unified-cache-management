import hashlib
import importlib
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ucm.integration.vllm import request_hash
from ucm.integration.vllm.request_hash import (
    UCM_CACHE_KEY_SCHEMA,
    UCM_MM_EXTRA_KEY_SCHEMA,
    build_cache_key_namespace,
    generate_block_extra_keys,
    generate_request_block_hashes,
    request_hashing_supported,
)
from ucm.integration.vllm.ucm_connector import (
    KVConnectorRole,
    RequestHasher,
    UCMConnector,
    UCMCPConnector,
    UCMDirectConnector,
    UCMLayerWiseConnector,
)


def _hasher(value: object) -> bytes:
    return hashlib.sha256(pickle.dumps(value)).digest()


def _mm_feature(
    identifier: bytes | None,
    offset: int = 1,
    length: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        identifier=identifier,
        mm_position=SimpleNamespace(offset=offset, length=length),
    )


def _request(
    *,
    image_id: bytes | None = None,
    mm_offset: int = 1,
    mm_length: int = 4,
    mm_features: list[SimpleNamespace] | None = None,
    cache_salt: str | None = None,
    lora_name: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
    token_ids: list[int] | None = None,
) -> SimpleNamespace:
    if mm_features is None:
        mm_features = (
            [] if image_id is None else [_mm_feature(image_id, mm_offset, mm_length)]
        )
    tokens = list(range(12)) if token_ids is None else token_ids
    return SimpleNamespace(
        request_id="request-a",
        all_token_ids=tokens,
        num_tokens=len(tokens),
        max_tokens=1,
        mm_features=mm_features,
        cache_salt=cache_salt,
        lora_request=(
            SimpleNamespace(lora_name=lora_name) if lora_name is not None else None
        ),
        prompt_embeds=prompt_embeds,
        _prompt_embeds_per_block_hashes={},
    )


def _prompt_embed_key(
    prompt_embeds: torch.Tensor,
    start_token_idx: int,
    end_token_idx: int,
) -> bytes:
    block = prompt_embeds[start_token_idx:end_token_idx]
    return hashlib.sha256(block.detach().cpu().contiguous().numpy().tobytes()).digest()


def _fake_vllm_extra_keys(
    request: Any,
    start_token_idx: int,
    end_token_idx: int,
    start_mm_idx: int,
) -> tuple[tuple[Any, ...] | None, int]:
    """Model vLLM 0.18's non-MM helper behavior for focused unit tests."""
    assert request.mm_features == []
    extra_keys: list[Any] = []
    if request.lora_request is not None:
        extra_keys.append(request.lora_request.lora_name)
    if start_token_idx == 0 and request.cache_salt:
        extra_keys.append(request.cache_salt)
    if request.prompt_embeds is not None:
        block_range = (start_token_idx, end_token_idx)
        embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
        if embeds_hash is None:
            embeds_hash = _prompt_embed_key(
                request.prompt_embeds, start_token_idx, end_token_idx
            )
            request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
        extra_keys.append(embeds_hash)
    return (tuple(extra_keys) or None), start_mm_idx


@pytest.fixture(autouse=True)
def _install_vllm_extra_key_helper(monkeypatch):
    monkeypatch.setattr(
        request_hash,
        "generate_block_hash_extra_keys",
        _fake_vllm_extra_keys,
    )


def _hashes(request: SimpleNamespace, block_size: int = 4) -> list[bytes]:
    return generate_request_block_hashes(request, block_size, b"seed", _hasher)


def test_same_tokens_image_identifier_and_position_produce_stable_hashes():
    assert _hashes(_request(image_id=b"image-a")) == _hashes(
        _request(image_id=b"image-a")
    )


def test_same_tokens_and_different_image_identifier_produce_different_hashes():
    image_a = _request(image_id=b"image-a")
    image_b = _request(image_id=b"image-b")

    assert image_a.all_token_ids == image_b.all_token_ids
    assert _hashes(image_a) != _hashes(image_b)


def test_same_identifier_at_different_block_positions_changes_hashes():
    offset_one = _request(image_id=b"image-a", mm_offset=1, mm_length=2)
    offset_two = _request(image_id=b"image-a", mm_offset=2, mm_length=2)

    assert offset_one.all_token_ids == offset_two.all_token_ids
    assert _hashes(offset_one) != _hashes(offset_two)


def test_same_identifier_and_offset_with_different_length_changes_hashes():
    short = _request(image_id=b"image-a", mm_offset=1, mm_length=2)
    long = _request(image_id=b"image-a", mm_offset=1, mm_length=3)

    assert short.all_token_ids == long.all_token_ids
    assert _hashes(short) != _hashes(long)


def test_multiblock_mm_keys_use_block_relative_offsets_and_parent_chain():
    request = _request(image_id=b"image-a", mm_offset=1, mm_length=8)
    mm_idx = 0
    extras_by_block = []
    for start in range(0, 12, 4):
        extras, mm_idx = generate_block_extra_keys(request, start, start + 4, mm_idx)
        extras_by_block.append(extras)

    assert extras_by_block == [
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", 1, 8),),
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", -3, 8),),
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", -7, 8),),
    ]

    hash_inputs = []

    def recording_hasher(value: object) -> bytes:
        hash_inputs.append(value)
        return _hasher(value)

    hashes = generate_request_block_hashes(request, 4, b"seed", recording_hasher)

    assert [value[2] for value in hash_inputs] == extras_by_block
    assert hash_inputs[0][0] == b"seed"
    assert hash_inputs[1][0] == hashes[0]
    assert hash_inputs[2][0] == hashes[1]


def test_multiple_mm_inputs_in_one_block_preserve_request_order():
    request = _request(
        mm_features=[
            _mm_feature(b"image-a", offset=0, length=1),
            _mm_feature(b"image-b", offset=2, length=1),
        ]
    )

    extras, next_mm_idx = generate_block_extra_keys(request, 0, 4, 0)

    assert extras == (
        (UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", 0, 1),
        (UCM_MM_EXTRA_KEY_SCHEMA, b"image-b", 2, 1),
    )
    assert next_mm_idx == 2

    changed_identifier = _request(
        mm_features=[
            _mm_feature(b"image-a", offset=0, length=1),
            _mm_feature(b"image-c", offset=2, length=1),
        ]
    )
    changed_offset = _request(
        mm_features=[
            _mm_feature(b"image-a", offset=0, length=1),
            _mm_feature(b"image-b", offset=3, length=1),
        ]
    )
    assert _hashes(request) != _hashes(changed_identifier)
    assert _hashes(request) != _hashes(changed_offset)


def test_mm_inputs_touching_block_boundaries_do_not_overlap():
    request = _request(
        mm_features=[
            _mm_feature(b"ends-at-start", offset=2, length=2),
            _mm_feature(b"starts-at-end", offset=8, length=2),
        ]
    )

    extras, next_mm_idx = generate_block_extra_keys(request, 4, 8, 0)

    assert extras is None
    assert next_mm_idx == 1


def test_start_mm_idx_minus_one_resumes_from_last_mm_input():
    request = _request(
        mm_features=[
            _mm_feature(b"image-a", offset=0, length=1),
            _mm_feature(b"image-b", offset=5, length=4),
        ]
    )

    extras, next_mm_idx = generate_block_extra_keys(request, 4, 8, -1)

    assert extras == ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-b", 1, 4),)
    assert next_mm_idx == 1


def test_local_mm_schema_prevents_reusing_identifier_only_hashes():
    parent_hash = b"seed"
    block_tokens = tuple(range(4))
    image_identifier = b"image-a"
    legacy_extra_keys = (image_identifier,)
    new_extra_keys = ((UCM_MM_EXTRA_KEY_SCHEMA, image_identifier, 1, 4),)

    legacy_hash = _hasher((parent_hash, block_tokens, legacy_extra_keys))
    new_hash = _hasher((parent_hash, block_tokens, new_extra_keys))

    assert legacy_hash != new_hash


def test_plain_text_hash_keeps_exact_v2_request_hasher_semantics():
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="model", dtype="float16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    hasher = RequestHasher(config, rank_id=1, namespace="deployment-a")
    parent_hash = hasher(b"fixed-parent")
    token_ids = [11, 12, 13, 14]

    actual = generate_request_block_hashes(
        _request(token_ids=token_ids),
        block_size=4,
        parent_hash=parent_hash,
        hasher=hasher,
    )
    expected = [hasher((parent_hash, tuple(token_ids), None))]

    assert UCM_CACHE_KEY_SCHEMA == "v2-multimodal"
    assert actual == expected


def test_empty_mm_features_keep_none_extra_key():
    request = _request(mm_features=[])

    extras, next_mm_idx = generate_block_extra_keys(request, 0, 4, 0)

    assert extras is None
    assert next_mm_idx == 0


@pytest.mark.parametrize(
    ("request_a", "request_b"),
    [
        (_request(cache_salt="tenant-a"), _request(cache_salt="tenant-b")),
        (_request(lora_name="adapter-a"), _request(lora_name="adapter-b")),
        (
            _request(prompt_embeds=torch.zeros(12, 2)),
            _request(prompt_embeds=torch.ones(12, 2)),
        ),
        (
            _request(image_id=b"image-a", lora_name="adapter-a"),
            _request(image_id=b"image-a", lora_name="adapter-b"),
        ),
        (
            _request(image_id=b"image-a", cache_salt="tenant-a"),
            _request(image_id=b"image-a", cache_salt="tenant-b"),
        ),
        (
            _request(image_id=b"image-a", prompt_embeds=torch.zeros(12, 2)),
            _request(image_id=b"image-a", prompt_embeds=torch.ones(12, 2)),
        ),
    ],
)
def test_semantic_request_inputs_and_mm_combinations_change_hashes(
    request_a, request_b
):
    assert request_a.all_token_ids == request_b.all_token_ids
    assert _hashes(request_a) != _hashes(request_b)


def test_mm_keys_precede_preserved_non_mm_helper_keys_without_duplication():
    prompt_embeds = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    cases = [
        (_request(image_id=b"image-a", lora_name="adapter-a"), "adapter-a"),
        (_request(image_id=b"image-a", cache_salt="tenant-a"), "tenant-a"),
        (
            _request(image_id=b"image-a", prompt_embeds=prompt_embeds),
            _prompt_embed_key(prompt_embeds, 0, 4),
        ),
    ]

    for request, non_mm_key in cases:
        extras, _ = generate_block_extra_keys(request, 0, 4, 0)

        assert extras == (
            (UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", 1, 4),
            non_mm_key,
        )
        assert b"image-a" not in extras


def test_proxy_preserves_future_non_mm_semantics_and_original_mm_features(
    monkeypatch,
):
    request = _request(image_id=b"image-a")
    request.future_semantic_key = b"future-key"
    original_mm_features = request.mm_features

    def future_helper(proxy, _start, _end, start_mm_idx):
        assert proxy.mm_features == []
        return (proxy.future_semantic_key,), start_mm_idx

    monkeypatch.setattr(request_hash, "generate_block_hash_extra_keys", future_helper)

    extras, _ = generate_block_extra_keys(request, 0, 4, 0)

    assert extras == (
        (UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", 1, 4),
        b"future-key",
    )
    assert request.mm_features is original_mm_features
    assert request.mm_features == [_mm_feature(b"image-a")]


def test_plain_text_hashes_are_stable_and_drop_partial_blocks():
    request = _request(token_ids=list(range(10)))

    assert _hashes(request) == _hashes(_request(token_ids=list(range(10))))
    assert len(_hashes(request)) == 2


@pytest.mark.parametrize(
    "semantic_request",
    [
        _request(image_id=b"image-a"),
        _request(cache_salt="tenant-a"),
        _request(lora_name="adapter-a"),
        _request(prompt_embeds=torch.zeros(12, 2)),
        _request(image_id=b"image-a", cache_salt="tenant-a"),
        _request(image_id=b"image-a", lora_name="adapter-a"),
        _request(image_id=b"image-a", prompt_embeds=torch.zeros(12, 2)),
    ],
)
def test_missing_vllm_helper_fails_closed_for_semantic_requests(
    monkeypatch, semantic_request
):
    monkeypatch.setattr(request_hash, "generate_block_hash_extra_keys", None)

    assert _hashes(semantic_request) == []


def test_missing_vllm_helper_keeps_plain_text_available(monkeypatch):
    monkeypatch.setattr(request_hash, "generate_block_hash_extra_keys", None)

    assert len(_hashes(_request())) == 3


def test_none_mm_identifier_fails_closed_before_helper_call(monkeypatch):
    helper_called = False

    def recording_helper(*_args):
        nonlocal helper_called
        helper_called = True
        return None, 0

    monkeypatch.setattr(
        request_hash, "generate_block_hash_extra_keys", recording_helper
    )
    request = _request(mm_features=[_mm_feature(None)])

    assert request_hashing_supported(request) is False
    assert _hashes(request) == []
    assert helper_called is False


def test_helper_exception_discards_all_blocks_before_hashing(monkeypatch):
    helper_starts = []
    hasher_inputs = []

    def failing_helper(_request, start, _end, start_mm_idx):
        helper_starts.append(start)
        if start == 4:
            raise RuntimeError("unsafe helper failure")
        return None, start_mm_idx

    def recording_hasher(value):
        hasher_inputs.append(value)
        return _hasher(value)

    monkeypatch.setattr(request_hash, "generate_block_hash_extra_keys", failing_helper)

    assert (
        generate_request_block_hashes(
            _request(image_id=b"image-a", mm_length=8),
            4,
            b"seed",
            recording_hasher,
        )
        == []
    )
    assert helper_starts == [0, 4]
    assert hasher_inputs == []


@pytest.mark.parametrize("helper_state", ["missing", "raises"])
def test_direct_connector_does_not_lookup_or_create_metadata_when_unsafe(
    monkeypatch, helper_state
):
    if helper_state == "missing":
        monkeypatch.setattr(request_hash, "generate_block_hash_extra_keys", None)
    else:

        def failing_helper(*_args):
            raise RuntimeError("unsafe helper failure")

        monkeypatch.setattr(
            request_hash, "generate_block_hash_extra_keys", failing_helper
        )

    store = SimpleNamespace(lookup_on_prefix_calls=0)

    def lookup_on_prefix(_block_ids):
        store.lookup_on_prefix_calls += 1
        return -1

    store.lookup_on_prefix = lookup_on_prefix
    connector = UCMDirectConnector.__new__(UCMDirectConnector)
    connector.block_size = 4
    connector.hash_block_size = 4
    connector.cp_world_size = 1
    connector._seed = b"seed"
    connector.request_hasher = _hasher
    connector.persist_token_threshold = 0
    connector.enable_record_traces = False
    connector.requests_meta = {}
    connector._other_rank_hashers = []
    connector.store = store

    assert connector.get_num_new_matched_tokens(
        _request(image_id=b"image-a"), num_computed_tokens=0
    ) == (0, False)
    assert store.lookup_on_prefix_calls == 0
    assert connector.requests_meta == {}


def test_direct_connector_shutdown_drains_pending_dump_tasks():
    wait_calls = []
    connector = UCMDirectConnector.__new__(UCMDirectConnector)
    connector._pending_dump_tasks = [
        SimpleNamespace(
            task="pending-dump",
            request_ids={"request-a"},
            event_handle=0,
            wait_for_save_start_ms=0.0,
        )
    ]
    connector.enable_event_sync = False
    connector.store = SimpleNamespace(wait=wait_calls.append)

    connector.shutdown()

    assert wait_calls == ["pending-dump"]
    assert connector._pending_dump_tasks == []


def test_outer_connector_shutdown_delegates_to_inner_connector():
    shutdown_calls = []
    connector = UCMConnector.__new__(UCMConnector)
    connector.connector = SimpleNamespace(shutdown=lambda: shutdown_calls.append(True))

    connector.shutdown()

    assert shutdown_calls == [True]


def test_cache_key_namespace_includes_schema_rank_and_operator_namespace():
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="model", dtype="float16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )

    rank0 = build_cache_key_namespace(config, 0, "deployment-a")
    rank1 = build_cache_key_namespace(config, 1, "deployment-a")
    other_namespace = build_cache_key_namespace(config, 0, "deployment-b")

    assert rank0.startswith(f"{UCM_CACHE_KEY_SCHEMA}:".encode())
    assert rank0 != rank1
    assert rank0 != other_namespace


def test_other_scheduler_rank_hashers_use_the_operator_namespace():
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="model", dtype="float16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=3),
    )
    connector = UCMDirectConnector.__new__(UCMDirectConnector)
    connector.is_mla = False
    connector.launch_config = {"request_hash_namespace": "deployment-a"}

    hashers = connector._make_other_rank_hashers(config)

    assert len(hashers) == 2
    assert all(hasher.meta_bytes.endswith(b":deployment-a") for hasher in hashers)


def test_cp_scheduler_and_worker_hashers_share_namespace_and_rank_keys(monkeypatch):
    vllm_distributed = importlib.import_module("vllm.distributed")
    prefetched_block_ids = []

    def fake_layerwise_init(self, vllm_config, _role, _kv_cache_config=None):
        self._vllm_config = vllm_config
        self.launch_config = {
            "use_layerwise": True,
            "request_hash_namespace": "deployment-a",
        }
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_rank = vllm_config.test_tp_rank
        self.is_mla = False
        self.block_size = 4
        self._other_rank_hashers = []

    monkeypatch.setattr(UCMLayerWiseConnector, "__init__", fake_layerwise_init)
    monkeypatch.setattr(
        UCMCPConnector,
        "_create_store",
        lambda _self, _layout: SimpleNamespace(prefetch=prefetched_block_ids.append),
    )
    monkeypatch.setattr(
        vllm_distributed,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
        raising=False,
    )
    monkeypatch.setattr(
        vllm_distributed,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
        raising=False,
    )

    def config(tp_rank):
        return SimpleNamespace(
            test_tp_rank=tp_rank,
            model_config=SimpleNamespace(model="model", dtype="float16"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=4,
                prefill_context_parallel_size=2,
                decode_context_parallel_size=1,
            ),
        )

    scheduler_config = config(tp_rank=0)
    worker_config = config(tp_rank=2)
    scheduler = UCMCPConnector(scheduler_config, KVConnectorRole.SCHEDULER)
    worker = UCMCPConnector(worker_config, KVConnectorRole.WORKER)

    assert scheduler.request_hasher.meta_bytes.endswith(b":0:deployment-a")
    assert worker.request_hasher.meta_bytes.endswith(b":1:deployment-a")
    assert len(scheduler._other_rank_hashers) == 1
    assert (
        scheduler._other_rank_hashers[0].meta_bytes == worker.request_hasher.meta_bytes
    )

    rank0_block_id = scheduler.request_hasher((b"seed", (1, 2, 3, 4), None))
    expected_worker_block_id = worker.request_hasher(rank0_block_id)
    assert scheduler._other_rank_hashers[0](rank0_block_id) == expected_worker_block_id

    scheduler._prefetch_other_rank_hashes([rank0_block_id])

    assert prefetched_block_ids == [[expected_worker_block_id]]
    assert scheduler_config.parallel_config.tensor_parallel_size == 4
    assert worker_config.parallel_config.tensor_parallel_size == 4
