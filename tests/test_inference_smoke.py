"""Smoke tests for the upgraded inference engine.

Verify imports work, helper functions behave correctly, and the FastAPI
route shape matches the new GroundingResponse without needing a GPU or
the actual model weights. Lets us catch Python errors before deploy.
"""

import base64
import io
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _png_b64(size=(64, 64)) -> str:
    img = Image.new("RGB", size, (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_inference_imports_and_helpers():
    """inference.py imports cleanly and the math helpers are sane."""
    import inference

    cx, cy = inference._parse_bbox_center("[10, 20, 30, 40]")
    assert (cx, cy) == (20.0, 30.0)

    cx, cy = inference._parse_bbox_center("noise (1.0, 2.0, 3.0, 5.0) tail")
    assert (cx, cy) == (2.0, 3.5)

    assert inference._parse_bbox_center("garbage") == (None, None)

    assert inference._median_pairwise_distance([]) == 0.0
    assert inference._median_pairwise_distance([(0, 0)]) == 0.0
    pts = [(0, 0), (3, 4), (6, 8)]
    assert abs(inference._median_pairwise_distance(pts) - 5.0) < 0.01

    assert inference._confidence_from_agreement(0.0, n_passes=4) == 0.99
    assert inference._confidence_from_agreement(2.0, n_passes=4) == 0.99
    assert inference._confidence_from_agreement(150.0, n_passes=4) == 0.4
    mid = inference._confidence_from_agreement(50.0, n_passes=4)
    assert 0.4 < mid < 0.99
    assert inference._confidence_from_agreement(0.0, n_passes=1) == 0.75

    centroid, voters = inference._largest_cluster_centroid(
        [(100, 100), (102, 101), (98, 99), (500, 500)]
    )
    assert len(voters) == 3
    assert 95 < centroid[0] < 105 and 95 < centroid[1] < 105

    centroid, voters = inference._largest_cluster_centroid([])
    assert centroid == (0.0, 0.0) and voters == []


def test_server_route_with_stub_engine():
    """FastAPI route accepts the new request and returns the new response.

    We monkeypatch the module's `engine` global to a stub so the route
    returns deterministic data without loading any model.
    """
    import inference
    import server
    from fastapi.testclient import TestClient

    class StubEngine:
        def predict(self, image, instruction, mode="fast"):
            return inference.GroundingResult(
                x=0.5, y=0.5, confidence=0.92,
                latency_ms=42, mode=mode, n_passes=2,
                agreement_px=3.5, raw_response="[10,10,30,30]",
            )

    server.engine = StubEngine()

    client = TestClient(server.app)

    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "fast" in body["modes_supported"] and "accurate" in body["modes_supported"]

    payload = {
        "image": _png_b64(),
        "instruction": "click submit",
        "mode": "accurate",
    }
    res = client.post("/v1/ground", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    for key in ("x", "y", "confidence", "latency_ms", "mode", "n_passes", "agreement_px"):
        assert key in body, f"missing {key} in {body}"
    assert body["mode"] == "accurate"

    res = client.post("/v1/ground", json={"image": _png_b64(), "instruction": ""})
    assert res.status_code == 400

    res = client.post("/v1/ground", json={"image": "not-base64$$$", "instruction": "x"})
    assert res.status_code == 400


if __name__ == "__main__":
    test_inference_imports_and_helpers()
    print("inference helpers: OK")
    try:
        test_server_route_with_stub_engine()
        print("server routes: OK")
    except Exception as e:
        print(f"server route test failed: {e}")
        raise
    print("ALL SMOKE TESTS PASSED")
