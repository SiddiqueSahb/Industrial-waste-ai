# WaRP-C inference service — CPU-only image for Cloud Run / any container host.
#
# Build:  docker build -t warp-api .
# Run:    docker run -p 8080:8080 warp-api
#
# Weights resolution order inside the container:
#   1. Baked in at build time (models/classification/*.pth copied below, if present)
#   2. Downloaded at startup from MODEL_URI (gs:// or https://)
#   3. ALLOW_UNTRAINED=1 serves random weights (CI smoke tests only)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    MODEL_NAME=convnext_tiny

WORKDIR /app

# CPU-only torch first (the default PyPI wheels pull in the full CUDA stack)
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r api/requirements.txt

# Application code — only what inference needs
COPY configs/ configs/
COPY src/__init__.py src/
COPY src/config/ src/config/
COPY src/classification/__init__.py \
     src/classification/models.py \
     src/classification/inference.py \
     src/classification/augmentations.py \
     src/classification/
COPY api/ api/

# Model weights: copies real checkpoints when building locally; in a fresh
# clone the directory only contains .gitkeep and weights come from MODEL_URI.
COPY models/ models/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
# Cloud Run injects PORT; default to 8080 elsewhere
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
