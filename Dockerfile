# syntax=docker/dockerfile:1.6

FROM python:3.11-slim AS base

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (but not data thanks to .dockerignore)
COPY src ./src
COPY app ./app
COPY data ./data
# Create folders
RUN mkdir -p /workspace/models /workspace/logs

# Optionally train at build-time if no model is present and TRAIN_IF_MISSING=true
# BuildKit mount lets us bind the dataset from build context at build-time without copying it into layers.
ARG TRAIN_IF_MISSING=true
ENV TRAIN_IF_MISSING=${TRAIN_IF_MISSING}

# Optionally run tests after build/training. Controlled by build-arg RUN_TESTS (true/false)
ARG RUN_TESTS=false
ENV RUN_TESTS=${RUN_TESTS}

# This step will be skipped if TRAIN_IF_MISSING=false or if a model already exists in models/
# Note: Ensure BuildKit is enabled: DOCKER_BUILDKIT=1
RUN --mount=type=bind,source=data,target=/workspace/data,ro \
    /bin/sh -lc 'set -e; \
    if [ "$TRAIN_IF_MISSING" = "true" ] && [ ! -f /workspace/models/skin_cnn_resnet18.pt ]; then \
        if [ -d /workspace/data ] && find /workspace/data -mindepth 1 -print -quit | grep -q .; then \
            echo "Training model at build-time..."; \
            python -m src.train; \
        else \
            echo "No dataset available during build; skipping training."; \
        fi; \
    else \
        echo "Skipping training during build."; \
    fi'

# If requested, and tests are provided via a bind mount, install testing deps and run pytest.
# This is useful for CI or validation builds where you want tests to execute as part of the image build.
RUN --mount=type=bind,source=tests,target=/workspace/tests,ro \
    /bin/sh -lc 'set -e; \
    if [ "${RUN_TESTS}" = "true" ] && [ -d /workspace/tests ]; then \
        echo "Installing test dependencies..."; \
        pip install --no-cache-dir pytest pytest-cov pytest-asyncio httpx; \
        echo "Running test suite..."; \
        # Fail fast and show concise output
        python -m pytest /workspace/tests -q --maxfail=1; \
    else \
        echo "Skipping tests during build."; \
    fi'

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
