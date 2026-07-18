# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime

ARG DOUZERO_GIT_SHA=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DOUZERO_GIT_SHA=${DOUZERO_GIT_SHA}

WORKDIR /workspace/DouZero

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3-venv \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

COPY . /workspace/DouZero

RUN python -m pip install --upgrade pip "setuptools>=77.0.3,<82" wheel \
    && python -m pip install -e . \
    && if [ -f requirements-dev.txt ]; then \
         python -m pip install -r requirements-dev.txt; \
       else \
         python -m pip install pytest pytest-timeout; \
       fi

# A Docker build normally has no access to the host GPU, so only validate
# imports and package metadata here.
RUN python - <<'PY'
import sys

import git
import numpy
import rlcard
import torch
import yaml

import douzero

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("DouZero:", douzero.__file__)
print("Dependency import test passed")
PY

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python", "-m", "pytest", "-q"]
