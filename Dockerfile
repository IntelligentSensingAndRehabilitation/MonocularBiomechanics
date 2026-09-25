FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# MuJoCo / OpenGL configuration
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Gradio configuration
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    ffmpeg \
    libgl1 \
    libegl1 \
    libgles2 \
    libosmesa6-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency files first so Docker can cache dependency installation
COPY pyproject.toml uv.lock ./

# Install Python + project dependencies
RUN uv sync --frozen --extra cuda

# Expose pip-installed NVIDIA CUDA libraries to TensorFlow
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/curand/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib"

RUN ln -sf \
    /app/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
    /app/.venv/bin/ptxas

# Copy source code
COPY . .

EXPOSE 7860

CMD ["uv", "run", "python", "main.py"]