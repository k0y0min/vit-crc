# Container for Multimodal Conformal Risk Control evaluation & replication
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

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY results/ /app/results/
COPY checkpoints/ /app/checkpoints/

ENV PYTHONPATH=/app/src

CMD ["python", "src/aggregate_2b_results.py"]
