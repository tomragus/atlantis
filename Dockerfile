# Ideagram Studio as one container: FastAPI serves the API and the built
# frontend together, so there is a single URL and no CORS in production.
#
# The repo layout is mirrored inside the image deliberately -- main.py finds
# test-images/ via Path(__file__).parent.parent, and the built frontend via
# Path(__file__).parent.parent / "frontend" / "out", so backend/, frontend/,
# and test-images/ have to sit beside each other inside the image exactly as
# they do in a checkout.

# --- stage 1: build the frontend ------------------------------------------
FROM node:22-slim AS web

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Empty on purpose: every API call in lib.ts becomes a same-origin relative
# path (nullish coalescing only falls back on null/undefined, not ""), since
# the built frontend and the API are served from the same origin once
# deployed.
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# --- stage 2: the application ----------------------------------------------
FROM python:3.12-slim AS app

# Where model weights get baked in, plus a promise never to reach for the
# network at runtime to fetch them.
ENV HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only torch, from PyTorch's own index -- the default PyPI wheel drags in
# the entire CUDA stack, gigabytes of GPU libraries for a Cloud Run instance
# that has no GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./backend/requirements.txt
# torch is already satisfied above (it's unpinned in requirements.txt), so
# this cannot pull the CUDA build back in.
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY test-images/ ./test-images/
COPY --from=web /build/frontend/out ./frontend/out

# Bakes all four algorithms' weights into the image at build time rather
# than fetching them on every cold start, through the app's own loader
# (embedding_tagger._load) so the model ids can't drift from HF_MODEL_IDS.
# Offline mode is lifted just for this one step.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 PYTHONPATH=/app/backend python -c "\
from algorithms import ALGORITHMS; \
from embedding_tagger import _load; \
[_load(algorithm) for algorithm in ALGORITHMS]"

WORKDIR /app/backend
EXPOSE 8080
# Cloud Run supplies $PORT; the default keeps a plain `docker run` working.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
