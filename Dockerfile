# HTTP contract from src/serve.py and src/serve_vllm.py.
# Model weights and organizer data are never copied into the image; they are mounted.
#   api       -> FastAPI client for an external vLLM backend (small CPU image)
#   extractor -> Transformers inference; CPU wheels unless TORCH_INDEX_URL selects CUDA

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models/merged \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /models/merged /cache/huggingface \
    && chown -R app:app /models /cache

FROM base AS api

COPY requirements-cpu.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-cpu.txt

COPY src ./src

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["python", "-m", "uvicorn", "serve_vllm:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

FROM api AS extractor

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

USER 0:0
COPY requirements-serve.txt ./
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" torch \
    && python -m pip install -r requirements-serve.txt
USER 10001:10001

CMD ["python", "-m", "uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
