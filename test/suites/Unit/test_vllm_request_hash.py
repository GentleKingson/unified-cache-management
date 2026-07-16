import hashlib
import pickle
from types import SimpleNamespace

import pytest
import torch

from ucm.integration.vllm import request_hash
from ucm.integration.vllm.request_hash import (
    UCM_CACHE_KEY_SCHEMA,
    build_cache_key_namespace,
    generate_request_block_hashes,
)
from ucm.integration.vllm.ucm_connector import UCMDirectConnector


def _hasher(value: object) -> bytes:
    return hashlib.sha256(pickle.dumps(value)).digest()


def _request(
    *,
    image_id: bytes | None = None,
    cache_salt: str | None = None,
    lora_name: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
    token_ids: list[int] | None = None,
):
    mm_features = []
    if image_id is not None:
        mm_features.append(
            SimpleNamespace(
                identifier=image_id,
                mm_position=SimpleNamespace(offset=1, length=4),
            )
        )
    tokens = token_ids or list(range(12))
    return SimpleNamespace(
        all_token_ids=tokens,
        num_tokens=len(tokens),
        mm_features=mm_features,
        cache_salt=cache_salt,
        lora_request=(
            SimpleNamespace(lora_name=lora_name) if lora_name is not None else None
        ),
        prompt_embeds=prompt_embeds,
        _prompt_embeds_per_block_hashes={},
    )


def _hashes(request, block_size: int = 4) -> list[bytes]:
    return generate_request_block_hashes(request, block_size, b"seed", _hasher)


def test_same_tokens_and_image_identifier_produce_same_hashes():
    assert _hashes(_request(image_id=b"image-a")) == _hashes(
        _request(image_id=b"image-a")
    )


def test_same_tokens_and_different_image_identifier_produce_different_hashes():
    image_a = _request(image_id=b"image-a")
    image_b = _request(image_id=b"image-b")

    assert image_a.all_token_ids == image_b.all_token_ids
    assert _hashes(image_a) != _hashes(image_b)


@pytest.mark.parametrize(
    ("request_a", "request_b"),
    [
        (_request(cache_salt="tenant-a"), _request(cache_salt="tenant-b")),
        (_request(lora_name="adapter-a"), _request(lora_name="adapter-b")),
        (
            _request(prompt_embeds=torch.zeros(12, 2)),
            _request(prompt_embeds=torch.ones(12, 2)),
        ),
    ],
)
def test_semantic_request_inputs_change_hashes(request_a, request_b):
    assert request_a.all_token_ids == request_b.all_token_ids
    assert _hashes(request_a) != _hashes(request_b)


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
