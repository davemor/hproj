FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN python -m pip install uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

# Install RAPIDS first
RUN uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.12.*" "dask-cudf-cu12==25.12.*" "cuml-cu12==25.12.*" \
    "cugraph-cu12==25.12.*" "nx-cugraph-cu12==25.12.*" "cuxfilter-cu12==25.12.*" \
    "cucim-cu12==25.12.*" "pylibraft-cu12==25.12.*" "raft-dask-cu12==25.12.*" \
    "cuvs-cu12==25.12.*"

# Install PyTorch
RUN uv pip install torch

# CRITICAL: Force nvjitlink 12.9+ LAST - RAPIDS/torch combo needs this version
# for the __nvJitLinkGetErrorLogSize_12_9 symbol
RUN uv pip install --force-reinstall nvidia-nvjitlink-cu12==12.9.86

COPY . .
RUN uv pip install -e .

CMD ["bash"]
