# Dockerfile for VGGT-SLAM with NVIDIA CUDA support for RunPod
# Uses Gradio to provide API interface

# Force x86_64 platform for CUDA/PyTorch compatibility
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libboost-all-dev \
    cmake \
    gcc \
    g++ \
    unzip \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA support (before other requirements)
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies (excluding torch/torchvision since already installed)
RUN grep -v "^torch" /app/requirements.txt | pip install -r /dev/stdin

# Clone and install Salad (image retrieval)
RUN git clone https://github.com/Dominic101/salad.git /app/salad \
    && pip install -e /app/salad

# Clone and install VGGT (Facebook's Visual Geometry Transformer)
RUN git clone https://github.com/facebookresearch/vggt.git /app/vggt \
    && pip install -e /app/vggt

# Copy the VGGT-SLAM project
COPY . /app/

# Install VGGT-SLAM in editable mode
RUN pip install -e /app

# Pre-download the VGGT model weights during build (optional but recommended)
# This prevents downloading on first request
RUN python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt', map_location='cpu')" || true

# Create directories for temporary files
RUN mkdir -p /app/temp_images /app/outputs

# Expose Gradio default port
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Make startup script executable
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command - use the startup script
CMD ["/app/start.sh"]
