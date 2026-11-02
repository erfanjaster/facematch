#!/usr/bin/env python3
"""Download the ONNX models used by the embedding backends.

Usage::

    python scripts/download_models.py                 # yunet + sface (default backend)
    python scripts/download_models.py --pack buffalo_l  # + scrfd/arcface (high-accuracy backend)
    python scripts/download_models.py --dest /somewhere
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
DEFAULT_MODELS = {
    "face_detection_yunet_2023mar.onnx": ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
BUFFALO_L = {
    "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "members": ["det_10g.onnx", "w600k_r50.onnx"],
}


def fetch(url: str) -> bytes:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "facematch-bootstrap"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="models", type=Path)
    ap.add_argument("--pack", choices=["", "buffalo_l"], default="")
    args = ap.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    for name, url in DEFAULT_MODELS.items():
        target = dest / name
        if target.exists() and target.stat().st_size > 10_000:
            print(f"= {name} already present")
            continue
        data = fetch(url)
        target.write_bytes(data)
        print(f"+ {name} ({len(data) / 1e6:.1f} MB)")

    if args.pack == "buffalo_l":
        for member in BUFFALO_L["members"]:
            target = dest / member
            if target.exists() and target.stat().st_size > 10_000:
                print(f"= {member} already present")
                continue
        else:
            data = fetch(BUFFALO_L["url"])
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for member in BUFFALO_L["members"]:
                    (dest / member).write_bytes(zf.read(member))
                    print(f"+ {member} ({(dest / member).stat().st_size / 1e6:.1f} MB)")

    print(f"\nmodels ready in {dest.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
