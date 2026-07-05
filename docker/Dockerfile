FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

WORKDIR /workspace

# System dependencies (Ubuntu 24.04 ships Python 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-full \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — install PyTorch with CUDA 13.0 support first
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.10.0+cu130 \
    torchvision==0.25.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# Copy project metadata and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --break-system-packages \
    "hydra-core>=1.3" \
    "omegaconf>=2.3" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "Pillow>=9.0" \
    "opencv-python>=4.7" \
    "matplotlib"

# Copy source code and install project
COPY . .
RUN pip install --no-cache-dir --break-system-packages -e ".[dev]"

ENTRYPOINT ["python"]
CMD ["train.py"]
