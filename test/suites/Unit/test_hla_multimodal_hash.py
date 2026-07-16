import hashlib
import math
import pickle
from types import SimpleNamespace

from ucm.integration.vllm import request_hash
from ucm.integration.vllm.hla_connector import (
    GroupInfo,
    KVCacheGroupManager,
    UCMHybridLinearAttentionConnector,
)
from ucm.integration.vllm.request_hash import (
    UCM_MM_EXTRA_KEY_SCHEMA,
    generate_block_extra_keys,
    generate_request_block_hashes,
)


def _hasher(value: object) -> bytes:
    return hashlib.sha256(pickle.dumps(value)).digest()


def _request(
    image_id: bytes | None,
    num_tokens: int = 12,
    *,
    offset: int = 1,
    length: int = 5,
):
    return SimpleNamespace(
        all_token_ids=list(range(num_tokens)),
        num_tokens=num_tokens,
        mm_features=[
            SimpleNamespace(
                identifier=image_id,
                mm_position=SimpleNamespace(offset=offset, length=length),
            )
        ],
        cache_salt=None,
        lora_request=None,
        prompt_embeds=None,
        _prompt_embeds_per_block_hashes={},
    )


def _manager() -> KVCacheGroupManager:
    manager = KVCacheGroupManager.__new__(KVCacheGroupManager)
    manager.request_hasher = _hasher
    manager.groups_by_id = [
        GroupInfo(0, 2, ("fa.0",), b"fa-small"),
        GroupInfo(1, 4, ("fa.1",), b"fa-large"),
        GroupInfo(2, 4, ("mamba.0",), b"mamba", is_mamba_align=True),
    ]
    manager.full_attn_groups = manager.groups_by_id[:2]
    manager.state_groups = manager.groups_by_id[2:]
    manager.lcm_block_size = math.lcm(
        *(group.block_size for group in manager.groups_by_id)
    )
    return manager


class _LookupSpy:
    def __init__(self):
        self.lookup_on_prefix_calls = 0
        self.lookup_calls = 0
        self.prefetch_calls = 0

    def lookup_on_prefix(self, _block_ids):
        self.lookup_on_prefix_calls += 1
        raise AssertionError("unsafe request must not run prefix lookup")

    def lookup(self, _block_ids):
        self.lookup_calls += 1
        raise AssertionError("unsafe request must not run state lookup")

    def prefetch(self, _block_ids):
        self.prefetch_calls += 1
        raise AssertionError("unsafe request must not prefetch")

    @property
    def total_calls(self) -> int:
        return self.lookup_on_prefix_calls + self.lookup_calls + self.prefetch_calls


def _connector(manager) -> tuple[UCMHybridLinearAttentionConnector, _LookupSpy]:
    connector = UCMHybridLinearAttentionConnector.__new__(
        UCMHybridLinearAttentionConnector
    )
    connector.group_manager = manager
    connector.persist_token_threshold = 0
    connector.requests_meta = {}
    connector.store = _LookupSpy()
    return connector, connector.store


def _mamba_state_hash(
    manager: KVCacheGroupManager, group_hashes: list[list[bytes]]
) -> bytes | None:
    return manager.compute_mamba_align_state_hash(
        manager.state_groups[0], 4, group_hashes
    )


def test_each_group_hashes_at_its_own_block_size():
    manager = _manager()
    request = _request(b"image-a")

    hashes = manager.compute_all_group_block_ids(request)

    assert [len(group_hashes) for group_hashes in hashes] == [6, 3, 3]
    assert hashes[0] == generate_request_block_hashes(request, 2, b"fa-small", _hasher)
    assert hashes[1] == generate_request_block_hashes(request, 4, b"fa-large", _hasher)
    assert hashes[2] == [b""] * 3
    assert manager.lcm_block_size == 4
    assert all(
        manager.lcm_block_size % group.block_size == 0 for group in manager.groups_by_id
    )


def test_different_images_change_primary_and_mamba_state_hashes():
    manager = _manager()
    image_a_hashes = manager.compute_all_group_block_ids(_request(b"image-a"))
    image_b_hashes = manager.compute_all_group_block_ids(_request(b"image-b"))

    assert image_a_hashes[0] != image_b_hashes[0]
    assert image_a_hashes[1] != image_b_hashes[1]
    assert _mamba_state_hash(manager, image_a_hashes) != _mamba_state_hash(
        manager, image_b_hashes
    )


def test_different_mm_offsets_change_every_full_attention_and_mamba_hash():
    manager = _manager()
    offset_1_hashes = manager.compute_all_group_block_ids(
        _request(b"image-a", offset=1)
    )
    offset_2_hashes = manager.compute_all_group_block_ids(
        _request(b"image-a", offset=2)
    )

    for group in manager.full_attn_groups:
        assert offset_1_hashes[group.group_id] != offset_2_hashes[group.group_id]
    assert _mamba_state_hash(manager, offset_1_hashes) != _mamba_state_hash(
        manager, offset_2_hashes
    )


def test_different_mm_lengths_change_every_full_attention_and_mamba_hash():
    manager = _manager()
    length_5_hashes = manager.compute_all_group_block_ids(
        _request(b"image-a", length=5)
    )
    length_4_hashes = manager.compute_all_group_block_ids(
        _request(b"image-a", length=4)
    )

    for group in manager.full_attn_groups:
        assert length_5_hashes[group.group_id] != length_4_hashes[group.group_id]
    assert _mamba_state_hash(manager, length_5_hashes) != _mamba_state_hash(
        manager, length_4_hashes
    )


def test_partial_blocks_are_not_hashed_for_any_group():
    manager = _manager()

    hashes = manager.compute_all_group_block_ids(_request(b"image-a", num_tokens=11))

    assert [len(group_hashes) for group_hashes in hashes] == [5, 2, 2]


def test_mm_position_key_uses_relative_offset_for_each_group_block_size():
    request = _request(b"image-a")

    mm_idx = 0
    size_2_extras = []
    for start in range(0, 6, 2):
        extras, mm_idx = generate_block_extra_keys(request, start, start + 2, mm_idx)
        size_2_extras.append(extras)

    mm_idx = 0
    size_4_extras = []
    for start in range(0, 8, 4):
        extras, mm_idx = generate_block_extra_keys(request, start, start + 4, mm_idx)
        size_4_extras.append(extras)

    assert size_2_extras == [
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", 1, 5),),
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", -1, 5),),
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", -3, 5),),
    ]
    assert size_4_extras == [
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", 1, 5),),
        ((UCM_MM_EXTRA_KEY_SCHEMA, b"image-a", -3, 5),),
    ]


def test_hla_does_not_create_persistence_metadata_without_extra_key_helper(
    monkeypatch,
):
    monkeypatch.setattr(request_hash, "generate_block_hash_extra_keys", None)
    connector, store = _connector(SimpleNamespace(lcm_block_size=4))
    request = _request(b"image-a")
    request.request_id = "request-a"

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.requests_meta == {}
    assert store.total_calls == 0


def test_hla_does_not_lookup_or_create_metadata_without_mm_identifier():
    connector, store = _connector(SimpleNamespace(lcm_block_size=4))
    request = _request(None)
    request.request_id = "request-a"

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.requests_meta == {}
    assert store.total_calls == 0


def test_hla_does_not_lookup_or_create_metadata_when_extra_key_helper_raises(
    monkeypatch,
):
    def fail_extra_key_generation(*_args, **_kwargs):
        raise RuntimeError("helper cannot safely represent this request")

    monkeypatch.setattr(
        request_hash,
        "generate_block_hash_extra_keys",
        fail_extra_key_generation,
    )
    connector, store = _connector(_manager())
    request = _request(b"image-a")
    request.request_id = "request-a"

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.requests_meta == {}
    assert store.total_calls == 0
