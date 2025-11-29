#!/bin/bash
# Startup script for VGGT-SLAM container on RunPod
# This script initializes the environment and starts the Gradio server

set -e

echo "=========================================="
echo "VGGT-SLAM Container Starting..."
echo "=========================================="

# Display GPU info
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Display Python and PyTorch info
echo ""
echo "Python version:"
python --version

echo ""
echo "PyTorch CUDA availability:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None; print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# Create necessary directories
mkdir -p /app/temp_images
mkdir -p /app/outputs

# Set environment variables for Gradio
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo ""
echo "=========================================="
echo "Starting Gradio server on ${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"
echo "=========================================="

# Start the Gradio application
cd /app
exec python app.py
