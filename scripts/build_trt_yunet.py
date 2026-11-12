"""YuNet (OpenCV model zoo, face_detection_yunet_2023mar.onnx, MIT) as a TensorRT engine for the face pass
(vibe_stream.face.TRTYuNet): the ONNX graph as published, input 1 x 3 x 640 x 640 BGR, fp32. Writes the engine and
<engine>.json (TensorRT version, platform, size).

Usage: python scripts/build_trt_yunet.py --onnx models/face_detection_yunet_2023mar.onnx --engine out/trt/yunet_640.engine
"""
import argparse
import json
import sys
from pathlib import Path

import _paths  # noqa: F401
from vibe_stream.trt_engine import build_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--engine", default="out/trt/yunet_640.engine")
    a = ap.parse_args()
    engine = Path(a.engine)
    engine.parent.mkdir(parents=True, exist_ok=True)
    version = build_onnx(a.onnx, engine, workspace_gb=1.0)
    engine.with_suffix(".json").write_text(json.dumps(dict(tensorrt=version, platform=sys.platform, size=640,
                                                           onnx=str(a.onnx)), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
