FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 to match HF Spaces runtime
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy requirements and install
COPY --chown=user:user network_predictor/backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY --chown=user:user network_predictor/backend/ /app/backend/
COPY --chown=user:user network_predictor/data/ /app/data/
COPY --chown=user:user network_predictor/frontend/ /app/frontend/
COPY --chown=user:user network_predictor/models/ /app/models/

# Set env variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    MODEL_PATH=/app/models/traffic_lstm.pth

EXPOSE 7860

# Start server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
