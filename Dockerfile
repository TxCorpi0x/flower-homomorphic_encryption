# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FL_SIMULATION=0

# System deps for building scientific stack, TenSEAL (cmake), and healthchecks (netcat)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install CPU-only PyTorch (faster/more reliable builds)
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.2.2 torchvision==0.17.2 --extra-index-url https://pypi.org/simple

# Copy requirements and install
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Default exposed ports for non-simulation server
EXPOSE 8081 8082 8083 8084

# Default command prints help; compose will override
CMD ["python", "-c", "print('Container ready. Use docker-compose services to run server/clients.')"]
