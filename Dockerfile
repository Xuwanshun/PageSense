# ════════════════════════════════════════════════════════════════════════════
# Multi-stage Dockerfile for RAG Agent for PDF Reading
#
# HOW MULTI-STAGE BUILDS WORK
# ─────────────────────────────
# A Dockerfile can have multiple FROM statements. Each one starts a new
# "stage". You can copy artifacts from one stage to the next.
# Docker caches each stage independently.
#
# WHY THIS MATTERS FOR YOUR PROJECT
# ──────────────────────────────────
# PaddlePaddle (~3.5 GB installed) takes ~15-20 minutes to install and
# download models. With multi-stage:
#   - Change a Python source file → only Stage 4 rebuilds (~5 seconds)
#   - Change requirements.txt     → Stage 2 + 3 rebuild (~15 min)
#   - First ever build             → All stages build (~20 min)
#
# Stage layout:
#   Stage 1 (base)    → OS + system libraries for Paddle
#   Stage 2 (deps)    → pip install (heavy, ~3.5 GB)
#   Stage 3 (models)  → Download Paddle ML models (~1.5 GB)
#   Stage 4 (final)   → Copy source code + configure runtime
#
# IMPORTANT: The model download in Stage 3 is baked into the image.
# This means the container starts quickly without needing internet access.
# The first build is slow; every subsequent build is fast.
# ════════════════════════════════════════════════════════════════════════════

# ── Stage 1: base ────────────────────────────────────────────────────────────
# python:3.11-slim is a minimal Debian-based image with Python 3.11.
# We use "slim" (not "alpine") because PaddlePaddle requires glibc, which
# Alpine does not have. Slim keeps the image smaller than the full image
# while staying compatible.
FROM python:3.11-slim AS base

# Minimal system libraries — only what PaddlePaddle and the health check need.
#
#   libgomp1  — OpenMP, required by PaddlePaddle for parallel CPU inference.
#   curl      — used by the Docker HEALTHCHECK (GET /health).
#
# WHY so few packages compared to typical OpenCV setups:
#   On Debian Trixie (python:3.11-slim), installing libgl1 triggers a full
#   Mesa + LLVM graphics stack — libllvm19 alone is 23 MB and LLVM's
#   post-install scripts are memory-intensive enough to OOM-kill Docker builds
#   on machines with constrained memory (Docker Desktop on MacBook Air).
#
#   We avoid libgl1 entirely by using opencv-python-headless (installed via
#   pip in Stage 2). The headless variant ships without OpenGL/X11 support,
#   which is exactly what we need — PaddleOCR never opens a window, it only
#   processes images in memory.
#
#   libglib2.0-0, libsm6, libxrender1, libxext6 are only needed when OpenCV
#   opens a display window (cv2.imshow / cv2.waitKey). We never call those.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
# All subsequent COPY and RUN commands use this as their base path.
WORKDIR /app

# ── Stage 2: deps ────────────────────────────────────────────────────────────
# Install Python dependencies BEFORE copying source code.
# This is the key caching trick: if you change source code but NOT
# requirements.txt, Docker reuses this cached layer (skips ~15 min reinstall).
FROM base AS deps

# Copy only the requirements file first — not the whole project.
COPY requirements.txt .

# Install dependencies.
#   --no-cache-dir  → don't store pip's download cache in the image (~500MB saved)
#   --upgrade pip   → use latest pip to avoid installation bugs
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 3: models ───────────────────────────────────────────────────────────
# Pre-download PaddleOCR and PaddleX models into the image.
# This is the slowest step (~5-10 min) but it only runs when deps change.
#
# WHY bake models into the image (vs. download at runtime):
#   - Container starts predictably (no 5-min download delay on first request)
#   - Works without internet access in restricted AWS VPCs
#   - Models are version-pinned (same model as in development)
#
# The models are stored at /app/paddle_models inside the image.
# In production, this path is replaced by an EFS volume mount so models
# are shared across container restarts without re-downloading.
FROM deps AS models

# Tell Paddle to use our controlled directory (not the default CWD/.paddlex)
ENV PADDLE_PDX_CACHE_HOME=/app/paddle_models
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Create the model cache directory
RUN mkdir -p /app/paddle_models/temp

# Pre-download PaddleOCR and PaddleX models.
# WHY a separate script file instead of RUN python -c "...":
#   Docker's Dockerfile parser reads line-by-line. A multi-line Python
#   string inside RUN python -c "..." confuses it — it tries to parse
#   lines like `import os` as Dockerfile instructions and fails.
#   Using a script file avoids this entirely and is easier to read.
COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py \
    || echo "Model pre-download encountered issues — models will download on first use."
# Note: the || echo means the build continues even if this fails.
# Models will then download lazily on first container startup.

# ── Stage 4: final ────────────────────────────────────────────────────────────
# Copy application source code. This is the stage that rebuilds on
# every code change — it is intentionally fast (just a file copy).
FROM models AS final

# ── Security: run as non-root user ───────────────────────────────────────────
# Running as root in a container is a security risk. If the container
# is ever compromised, the attacker would have root. A dedicated user
# limits the blast radius.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

# Copy application source code (files not in .dockerignore)
COPY . /app/

# Create data directories that the app writes to at runtime.
# These are typically replaced by volume mounts (Docker) or EFS mounts (ECS).
RUN mkdir -p /app/data/raw /app/data/processed /app/data/embedded \
    && chown -R appuser:appgroup /app/data /app/paddle_models

# Switch to non-root user
USER appuser

# ── Runtime environment variables ────────────────────────────────────────────
# These are DEFAULTS — they can be overridden at runtime by ECS task
# definition environment variables or docker-compose environment section.

# Critical: without PYTHONUNBUFFERED=1, Python buffers stdout and your
# logs may NEVER appear in CloudWatch Logs (they sit in Python's buffer
# until the process exits, which in ECS it never does gracefully).
ENV PYTHONUNBUFFERED=1

# Skip writing .pyc bytecode files (containers don't benefit from them)
ENV PYTHONDONTWRITEBYTECODE=1

# Paddle model cache (points to the directory we baked models into above)
ENV PADDLE_PDX_CACHE_HOME=/app/paddle_models
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# App configuration defaults for the container environment
ENV RAW_DOCUMENTS_DIR=/app/data/raw
ENV PROCESSED_DOCUMENTS_DIR=/app/data/processed
ENV VECTORSTORE_DIR=/app/data/embedded
ENV PADDLE_CACHE_DIR=/app/paddle_models

# Use JSON logging in the container so CloudWatch can parse structured logs
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO

# Start in API server mode (the CLI mode is for local use without Docker)
ENV APP_MODE=api

# ── Port ──────────────────────────────────────────────────────────────────────
# Tell Docker (and ECS) that this container listens on port 8000.
# This is documentation only — it does not actually open the port.
# You still need to map the port in docker run / docker-compose / ECS task.
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker and ECS both use this to verify the container is alive.
# If /health returns non-200 three times, ECS replaces the container.
#
# interval=30s  — check every 30 seconds
# timeout=10s   — the request must complete within 10 seconds
# retries=3     — fail after 3 consecutive failures
# start_period=120s — wait 120s before starting checks (Paddle needs time to load)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Start the FastAPI server. This is the command ECS runs when the
# container starts. The --serve flag triggers _run_server() in main.py,
# which starts uvicorn on 0.0.0.0:8000.
CMD ["python", "main.py", "--serve"]
