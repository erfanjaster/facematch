# ---- build stage -----------------------------------------------------------
FROM python:3.11-slim AS build
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt faiss-cpu onnxruntime \
    && pip install --no-cache-dir --prefix=/install .

# ---- runtime stage -----------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app
COPY --from=build /install /usr/local
COPY src ./src
COPY web ./web
COPY scripts ./scripts
COPY demo_images ./demo_images
COPY queries ./queries
ENV FACEMATCH_MODELS_DIR=/app/models \
    FACEMATCH_DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1
RUN python scripts/download_models.py --dest /app/models
EXPOSE 8000
VOLUME ["/app/data"]
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["python", "-m", "facematch"]
