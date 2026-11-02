#!/usr/bin/env python3
"""Seed the database from a directory tree (one sub-folder per person) and
then run the bundled query images so you can eyeball match quality.

Usage::

    python scripts/seed_demo_db.py                       # demo_images + queries/
    python scripts/seed_demo_db.py --root my_photos --query my_query.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facematch.config import get_settings  # noqa: E402
from facematch.service import FaceMatchService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="demo_images", type=Path)
    ap.add_argument("--query", action="append", type=Path, default=None)
    args = ap.parse_args()

    svc = FaceMatchService(get_settings())
    svc.start()
    report = svc.ingest_directory(str(args.root))
    print(f"enrolled: {report['faces']} faces / {report['persons']} persons")
    for skip in report["skipped"]:
        print(f"  skipped: {skip}")

    queries = args.query or sorted(Path("queries").glob("*.jpg")) if Path("queries").exists() else []
    for q in queries:
        if not q.exists():
            continue
        result = svc.search(q.read_bytes())
        print(f"\nquery: {q}")
        for face in result.faces:
            if face.matched:
                best = face.candidates[0]
                print(
                    f"  face #{face.face_index + 1}: MATCH -> {best['person_name']} "
                    f"(sim {best['similarity']:.3f}, search {result.timings_ms['search']:.1f} ms)"
                )
            else:
                top = face.candidates[0] if face.candidates else None
                print(
                    f"  face #{face.face_index + 1}: no match"
                    + (f" (closest {top['person_name']} @ {top['similarity']:.3f})" if top else "")
                )
        print(f"  timings: {result.timings_ms}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
