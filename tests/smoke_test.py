"""Smoke test for the inference API.

Hits a running service (BASE_URL env, default http://localhost:8000):
  1. GET  /health   -> 200, status == "ok"
  2. POST /predict  -> 200, valid class name, confidence in [0, 1], top_k sorted

The test image is generated in memory, so no dataset or weights are needed —
run the server with ALLOW_UNTRAINED=1 in CI.

    python tests/smoke_test.py
"""
from __future__ import annotations

import io
import os
import sys
import time

import httpx
from PIL import Image

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
STARTUP_TIMEOUT_S = int(os.environ.get("STARTUP_TIMEOUT_S", "90"))


def wait_for_server(client: httpx.Client) -> dict:
    deadline = time.time() + STARTUP_TIMEOUT_S
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = client.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                return r.json()
        except httpx.TransportError as e:
            last_err = e
        time.sleep(2)
    raise SystemExit(f"Server did not become healthy within {STARTUP_TIMEOUT_S}s: {last_err}")


def make_test_jpeg() -> bytes:
    img = Image.new("RGB", (320, 240), color=(40, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def main() -> None:
    with httpx.Client(timeout=30) as client:
        health = wait_for_server(client)
        assert health["status"] == "ok", health
        assert health["num_classes"] == 28, health
        print(f"[ok] /health: model={health['model_name']} device={health['device']} "
              f"weights_loaded={health['weights_loaded']}")

        r = client.post(
            f"{BASE_URL}/predict",
            files={"file": ("test.jpg", make_test_jpeg(), "image/jpeg")},
        )
        assert r.status_code == 200, f"/predict returned {r.status_code}: {r.text}"
        body = r.json()
        assert body["predicted_class"], body
        assert 0.0 <= body["confidence"] <= 1.0, body
        confs = [e["confidence"] for e in body["top_k"]]
        assert confs == sorted(confs, reverse=True), f"top_k not sorted: {confs}"
        assert body["top_k"][0]["class_name"] == body["predicted_class"], body
        print(f"[ok] /predict: class={body['predicted_class']} "
              f"confidence={body['confidence']:.4f} latency={body['latency_ms']}ms")

        # invalid upload must be rejected cleanly
        r = client.post(f"{BASE_URL}/predict",
                        files={"file": ("junk.txt", b"not an image", "text/plain")})
        assert r.status_code == 400, f"expected 400 for junk upload, got {r.status_code}"
        print("[ok] /predict rejects non-image uploads with 400")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
