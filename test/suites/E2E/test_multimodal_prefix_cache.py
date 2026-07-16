import base64
import os
import re
import struct
import time
import zlib
from pathlib import Path

import pytest
import requests
from common.online_inference_utils import VLLMServerManager

_CACHE_DUMP_TIMEOUT_SECONDS = 120.0
_CACHE_QUIET_SECONDS = 2.0


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
    pattern = re.compile(r"^ucm:ucm_hit_tokens_total(?:\{[^}]*\})?\s+([-+eE0-9.]+)$")
    for line in response.text.splitlines():
        match = pattern.match(line)
        if match:
            total += float(match.group(1))
    return total


def _cache_file_state(
    storage_path: Path,
) -> tuple[tuple[Path, ...], tuple[tuple[str, int, int], ...], tuple[Path, ...]]:
    if not storage_path.exists():
        return (), (), ()

    committed_files = []
    committed_snapshot = []
    active_files = []
    for path in storage_path.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
            if path.name.endswith(".tmp"):
                active_files.append(path)
            elif stat.st_size > 0:
                committed_files.append(path)
                committed_snapshot.append(
                    (
                        str(path.relative_to(storage_path)),
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
        except FileNotFoundError:
            # PosixStore can rename an active file between discovery and stat.
            continue

    return (
        tuple(sorted(committed_files)),
        tuple(sorted(committed_snapshot)),
        tuple(sorted(active_files)),
    )


def _wait_for_cache_quiescence(storage_path: Path) -> tuple[Path, ...]:
    deadline = time.monotonic() + _CACHE_DUMP_TIMEOUT_SECONDS
    last_snapshot: tuple[tuple[str, int, int], ...] | None = None
    quiet_since: float | None = None
    while True:
        cache_files, snapshot, active_files = _cache_file_state(storage_path)
        now = time.monotonic()
        is_committed_and_idle = bool(cache_files) and not active_files

        if is_committed_and_idle and snapshot == last_snapshot:
            if quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= _CACHE_QUIET_SECONDS:
                return cache_files
        else:
            last_snapshot = snapshot
            quiet_since = now if is_committed_and_idle else None

        remaining = deadline - now
        if remaining <= 0:
            raise AssertionError(
                "Timed out waiting for nonempty committed Posix cache files to "
                "become quiescent; "
                f"committed={snapshot}, "
                "active="
                f"{[str(path.relative_to(storage_path)) for path in active_files]}"
            )
        time.sleep(min(0.1, remaining))


def _require_npu() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch is required for the NPU test")
    pytest.importorskip("torch_npu", reason="torch_npu is required for the NPU test")
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        pytest.skip("An available NPU is required for the multimodal hardware test")


@pytest.mark.stage(2)
@pytest.mark.platform("npu")
@pytest.mark.feature("multimodal_prefix_cache")
def test_multimodal_posix_cache_isolated_across_server_restart(tmp_path):
    model_path = os.getenv("UCM_MM_MODEL_PATH")
    if not model_path:
        pytest.skip("UCM_MM_MODEL_PATH is required for the multimodal hardware test")
    _require_npu()

    served_model_name = "ucm-mm-test"
    storage_path = tmp_path / "ucm-mm-posix"
    ucm_config = {
        "enable_metrics": True,
        "use_layerwise": True,
        "request_hash_namespace": "e2e-multimodal-posix-restart-v1",
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
        before_first_red = _ucm_hit_tokens(server.url)
        red_output, red_prompt_tokens = _chat(server.url, served_model_name, red_image)
        after_first_red = _ucm_hit_tokens(server.url)
        second_red_output, second_red_prompt_tokens = _chat(
            server.url, served_model_name, red_image
        )
        after_second_red = _ucm_hit_tokens(server.url)

        assert "RED" in red_output
        assert "RED" in second_red_output
        assert red_prompt_tokens == second_red_prompt_tokens

        first_red_hits = after_first_red - before_first_red
        second_red_hits = after_second_red - after_first_red
        assert second_red_hits > 0, (
            "The second red request did not reuse the cache written by the first "
            f"request: first_delta={first_red_hits}, second_delta={second_red_hits}"
        )

    # HLA's per-forward wait drains layerwise dumps, and graceful shutdown
    # drains any generic pending store tasks before this post-exit check.
    persisted_files = _wait_for_cache_quiescence(storage_path)
    assert persisted_files, "Service A left no nonempty committed Posix cache files"

    with VLLMServerManager(**server_args) as server:
        before_restart_red = _ucm_hit_tokens(server.url)
        restart_red_output, restart_red_prompt_tokens = _chat(
            server.url, served_model_name, red_image
        )
        after_restart_red = _ucm_hit_tokens(server.url)

        restart_red_hits = after_restart_red - before_restart_red
        assert "RED" in restart_red_output
        assert restart_red_prompt_tokens == red_prompt_tokens
        assert restart_red_hits > 0, (
            "Service B did not reuse Service A's persisted red-image cache: "
            f"delta={restart_red_hits}"
        )

        blue_output, blue_prompt_tokens = _chat(
            server.url, served_model_name, blue_image
        )
        after_first_blue = _ucm_hit_tokens(server.url)
        second_blue_output, _ = _chat(server.url, served_model_name, blue_image)
        after_second_blue = _ucm_hit_tokens(server.url)

    first_blue_hits = after_first_blue - after_restart_red
    second_blue_hits = after_second_blue - after_first_blue
    assert "BLUE" in blue_output
    assert "BLUE" in second_blue_output
    assert red_prompt_tokens == blue_prompt_tokens
    assert first_blue_hits < restart_red_hits, (
        "The first blue request reused as much of the persisted prefix as the red "
        "request, so multimodal cache isolation was not demonstrated: "
        f"red_delta={restart_red_hits}, blue_delta={first_blue_hits}"
    )
    assert first_blue_hits < blue_prompt_tokens
    assert second_blue_hits > first_blue_hits, (
        "The second blue request did not improve on the isolated first-blue lookup: "
        f"first_delta={first_blue_hits}, second_delta={second_blue_hits}"
    )
