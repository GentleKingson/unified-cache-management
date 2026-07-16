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
from ucm.integration.vllm.request_hash import generate_block_extra_keys


def _hasher(value: object) -> bytes:
    return hashlib.sha256(pickle.dumps(value)).digest()


def _request(image_id: bytes, num_tokens: int = 12):
    return SimpleNamespace(
        all_token_ids=list(range(num_tokens)),
        num_tokens=num_tokens,
        mm_features=[
            SimpleNamespace(
                identifier=image_id,
                mm_position=SimpleNamespace(offset=1, length=5),
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


def test_each_group_hashes_at_its_own_block_size():
    manager = _manager()

    hashes = manager.compute_all_group_block_ids(_request(b"image-a"))

    assert [len(group_hashes) for group_hashes in hashes] == [6, 3, 3]
    assert manager.lcm_block_size == 4
    assert all(
        manager.lcm_block_size % group.block_size == 0
        for group in manager.groups_by_id
    )


def test_different_images_change_primary_and_mamba_state_hashes():
    manager = _manager()
    image_a_hashes = manager.compute_all_group_block_ids(_request(b"image-a"))
    image_b_hashes = manager.compute_all_group_block_ids(_request(b"image-b"))

    assert image_a_hashes[0] != image_b_hashes[0]
    assert image_a_hashes[1] != image_b_hashes[1]
    assert manager.compute_mamba_align_state_hash(
        manager.state_groups[0], 4, image_a_hashes
    ) != manager.compute_mamba_align_state_hash(
        manager.state_groups[0], 4, image_b_hashes
    )


def test_partial_blocks_are_not_hashed_for_any_group():
    manager = _manager()

    hashes = manager.compute_all_group_block_ids(_request(b"image-a", num_tokens=11))

    assert [len(group_hashes) for group_hashes in hashes] == [5, 2, 2]


def test_multimodal_identifier_is_present_in_every_overlapping_block():
    request = _request(b"image-a")
    mm_idx = 0
    block_extras = []
    for start in range(0, 6, 2):
        extras, mm_idx = generate_block_extra_keys(request, start, start + 2, mm_idx)
        block_extras.append(extras)

    assert all(b"image-a" in extras for extras in block_extras)


def test_hla_does_not_create_persistence_metadata_without_extra_key_helper(
    monkeypatch,
):
    monkeypatch.setattr(request_hash, "generate_block_hash_extra_keys", None)
    connector = UCMHybridLinearAttentionConnector.__new__(
        UCMHybridLinearAttentionConnector
    )
    connector.group_manager = SimpleNamespace(lcm_block_size=4)
    connector.persist_token_threshold = 0
    connector.requests_meta = {}
    request = _request(b"image-a")
    request.request_id = "request-a"

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.requests_meta == {}
