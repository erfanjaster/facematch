#!/usr/bin/env bash
# One-shot bootstrap: venv + deps + models. Idempotent — safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
VENV=${VENV:-.venv}

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating virtualenv in $VENV"
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip wheel >/dev/null
fi

echo ">> installing dependencies"
"$VENV/bin/python" -m pip install -e ".[faiss,arcface]" >/dev/null

echo ">> downloading ONNX models (add --pack buffalo_l later for arcface)"
"$VENV/bin/python" scripts/download_models.py

cat <<'EOF'

Done. Next steps:
  source .venv/bin/activate
  python scripts/seed_demo_db.py     # optional: load the demo gallery
  python -m facematch                # serve UI + API on :8000
EOF
