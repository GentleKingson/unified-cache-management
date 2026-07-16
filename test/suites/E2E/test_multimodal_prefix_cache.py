import base64
import os
import re
import struct
import zlib

import pytest
import requests
from common.online_inference_utils import VLLMServerManager


def _solid_png_data_url(rgb: tuple[int, int, int], size: int = 64) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _chat(server_url: str, model: str, image_url: str) -> tuple[str, int]:
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {
                            "type": "text",
                            "text": (
                                "Identify the dominant color in this image. "
                                "Answer with exactly RED or BLUE."
                            ),
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8,
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    return (
        payload["choices"][0]["message"]["content"].strip().upper(),
        payload["usage"]["prompt_tokens"],
    )


def _ucm_hit_tokens(server_url: str) -> float:
    response = requests.get(f"{server_url}/metrics", timeout=30)
    response.raise_for_status()
    total = 0.0
    pattern = re.compile(
        r"^ucm:ucm_hit_tokens_total(?:\{[^}]*\})?\s+([-+eE0-9.]+)$"
    )
    for line in response.text.splitlines():
        match = pattern.match(line)
        if match:
            total += float(match.group(1))
    return total


@pytest.mark.stage(2)
@pytest.mark.platform("npu")
@pytest.mark.feature("multimodal_prefix_cache")
def test_multimodal_posix_cache_isolated_across_server_restart(tmp_path):
    model_path = os.getenv("UCM_MM_MODEL_PATH")
    if not model_path:
        pytest.skip("UCM_MM_MODEL_PATH is required for the multimodal hardware test")

    served_model_name = "ucm-mm-test"
    storage_path = tmp_path / "ucm-mm-posix"
    ucm_config = {
        "enable_metrics": True,
        "use_layerwise": True,
        "request_hash_namespace": "e2e-qwen36-hla-mm-v2",
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": {
                    "store_pipeline": "Cache|Posix",
                    "storage_backends": str(storage_path),
                    "io_direct": False,
                    "cache_buffer_capacity_gb": 2,
                },
            }
        ],
    }
    server_args = {
        "model_path": model_path,
        "served_model_name": served_model_name,
        "port": int(os.getenv("UCM_MM_TEST_PORT", "8000")),
        "ucm_config": ucm_config,
        "enable_prefix_caching": False,
        "max_model_len": 4096,
        "max_num_batched_tokens": 4096,
        "additional_args": ["--enforce-eager"],
        "startup_timeout": 600,
    }
    red_image = _solid_png_data_url((255, 0, 0))
    blue_image = _solid_png_data_url((0, 0, 255))

    with VLLMServerManager(**server_args) as server:
        red_output, red_prompt_tokens = _chat(
            server.url, served_model_name, red_image
        )
        assert "RED" in red_output

    with VLLMServerManager(**server_args) as server:
        before = _ucm_hit_tokens(server.url)
        blue_output, blue_prompt_tokens = _chat(
            server.url, served_model_name, blue_image
        )
        after_first_blue = _ucm_hit_tokens(server.url)
        second_blue_output, _ = _chat(server.url, served_model_name, blue_image)
        after_second_blue = _ucm_hit_tokens(server.url)

    first_blue_hits = after_first_blue - before
    second_blue_hits = after_second_blue - after_first_blue
    assert "BLUE" in blue_output
    assert "BLUE" in second_blue_output
    assert red_prompt_tokens == blue_prompt_tokens
    assert first_blue_hits < blue_prompt_tokens
    assert second_blue_hits > first_blue_hits
