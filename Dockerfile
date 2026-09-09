# Production Dockerfile for Multimodal Conformal Risk Control (Google Cloud Run ready)
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch (CPU wheel for Cloud Run serving or CUDA if GPU runtime selected)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    peft \
    gradio \
    matplotlib \
    seaborn \
    pillow \
    qwen-vl-utils

# Copy project files
COPY src/ /app/src/
COPY app/ /app/app/
COPY results/ /app/results/
COPY checkpoints/ /app/checkpoints/

ENV PYTHONPATH=/app/src
ENV PORT=8080

EXPOSE 8080

CMD ["python", "app/app.py"]
