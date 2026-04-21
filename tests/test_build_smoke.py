"""Build-time smoke for the vLLM container image.

ACR builds on a CPU-only worker, so we can't actually load the model
here. But we CAN verify:
  - vllm imports cleanly (catches missing libs, mismatched pins)
  - inference.py imports under the deferred-imports model
  - helpers compute the right values
  - server.py FastAPI app instantiates

Anything past this is a runtime concern caught by /health polling
right after deploy.
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def main() -> int:
    print("=== build smoke ===")
    print(f"python: {sys.version}")

    import vllm
    print(f"vllm: {vllm.__version__}")

    import transformers
    print(f"transformers: {transformers.__version__}")

    import torch
    print(f"torch: {torch.__version__}")

    import inference
    assert hasattr(inference, "GroundingEngine"), "GroundingEngine missing"
    assert hasattr(inference, "_smart_resize"), "_smart_resize missing"
    assert hasattr(inference, "CoarseResult"), "CoarseResult missing"

    cx, cy = inference._parse_bbox_center("[100, 200, 300, 400]")
    assert (cx, cy) == (200.0, 300.0), f"parse failed: {cx},{cy}"

    new_h, new_w = inference._smart_resize(1080, 1920, max_pixels=800_000)
    assert (new_h * new_w) <= 800_000
    assert new_h % 28 == 0 and new_w % 28 == 0

    win = inference._compute_crop_window((500, 500), (1000, 1000))
    assert win[0] >= 0 and win[2] <= 1000

    import server
    assert hasattr(server, "app"), "FastAPI app missing"
    # Confirm streaming endpoint is registered
    routes = {r.path for r in server.app.routes}
    assert "/v1/ground" in routes, f"/v1/ground not in {routes}"
    assert "/v1/ground/stream" in routes, f"/v1/ground/stream not in {routes}"

    print("BUILD SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
