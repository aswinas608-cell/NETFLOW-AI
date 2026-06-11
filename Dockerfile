FROM python:3.11-slim

# ── Environment ────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies (own layer — cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────────────────────
COPY . .

# ── Runtime directories ─────────────────────────────────────────────────────
# /tmp is always writable on HF Spaces (free tier has no persistent disk)
RUN mkdir -p /tmp/rag_data static

# ── Hugging Face Spaces: port 7860 is required ──────────────────────────────
EXPOSE 7860

# ── Start server ────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "75"]
