# syntax=docker/dockerfile:1.7
#
# Production image for Hugging Face Spaces (or any container host).
# Builds the React frontend, then bakes it into a Python image that
# serves both the static UI and the FastAPI backend from one process.
#
# Local development is unaffected — this file is only used by the
# container host. `./dev.sh` continues to run backend + Vite separately.

# ---------- stage 1: build the React frontend ----------
FROM node:22-slim AS frontend-build

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
# Empty VITE_BACKEND_URL is a real value here, not "unset" — the app
# code treats this as "same-origin, use relative URLs".
ENV VITE_BACKEND_URL=""
RUN npm run build


# ---------- stage 2: Python runtime ----------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tools needed by some wheels (chromadb, tokenizers fallback).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, deterministic installs.
RUN pip install uv

WORKDIR /app

# Install CPU-only torch first so sentence-transformers doesn't pull
# the 2GB CUDA wheels (HF free tier has no GPU).
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Backend deps (cache layer — only re-runs when pyproject changes).
COPY backend/pyproject.toml ./backend/
RUN cd backend && uv pip install --system .

# Backend source.
COPY backend/ ./backend/

# Built frontend from stage 1.
COPY --from=frontend-build /build/dist ./frontend/dist

# HF Spaces mounts /data as the persistent disk (if attached).
# Without it, chroma writes still work but evaporate on Space restart —
# acceptable for a demo. Local dev keeps its ./data/chroma default.
ENV CHROMA_PERSIST_DIR=/data/chroma

# HF Spaces sets PORT=7860 and routes traffic there.
EXPOSE 7860

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
