# ── Build stage ──────────────────────────────────────────────────────────────
# nvidia/cuda base gives us CUDA 12.1 + cuDNN 8 runtime so that
# sentence-transformers, fastembed and docling can use the GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python 3.11 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        build-essential \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip

WORKDIR /build

# Copy dependency declaration first (cache-friendly)
COPY pyproject.toml .
COPY backend.egg-info/ backend.egg-info/

# Install all project dependencies into a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.1 support *before* the rest so the
# GPU-capable wheel is picked up by downstream packages
RUN pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# Install the backend package and its runtime dependencies
COPY . .
RUN pip install --no-cache-dir ".[dev]" || pip install --no-cache-dir .

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Let torch / fastembed / onnxruntime discover the GPU
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        # Runtime libs needed by OpenCV / docling image handling
        libgl1 \
        libglib2.0-0 \
        # PDF rendering
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from the builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application source
COPY src/ ./src/
COPY pyproject.toml .

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --no-create-home appuser \
 && chown -R appuser:appgroup /app
USER appuser

EXPOSE 5008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:5008/health')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5008"]
