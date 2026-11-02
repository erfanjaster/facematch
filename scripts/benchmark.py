#!/usr/bin/env python3
"""Standalone latency benchmark for the vector-search layer.

Usage::

    python scripts/benchmark.py --sizes 1000 10000 100000 --iterations 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facematch.config import get_settings  # noqa: E402
from facematch.service import FaceMatchService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    ap.add_argument("--iterations", type=int, default=300)
    args = ap.parse_args()

    svc = FaceMatchService(get_settings())
    report = svc.benchmark(args.sizes, args.iterations)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
