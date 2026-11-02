PY ?= python3
VENV ?= .venv

.PHONY: bootstrap venv models run test bench seed clean docker

bootstrap: venv models install

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip wheel

install:
	$(VENV)/bin/pip install -e ".[dev,faiss,arcface]"

models:
	$(VENV)/bin/python scripts/download_models.py

run:
	$(VENV)/bin/python -m facematch

test:
	$(VENV)/bin/python -m pytest tests -q

bench:
	$(VENV)/bin/python scripts/benchmark.py

seed:
	$(VENV)/bin/python scripts/seed_demo_db.py

clean:
	rm -rf data models .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

docker:
	docker build -t facematch:0.1.0 .
	docker run --rm -p 8000:8000 facematch:0.1.0
