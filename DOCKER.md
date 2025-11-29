# Docker Deployment for VGGT-SLAM

This guide explains how to build and run VGGT-SLAM in a Docker container with NVIDIA CUDA support, suitable for deployment on RunPod or similar GPU cloud platforms.

## Prerequisites

- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- For local testing: `docker-compose` with GPU support

## Quick Start

### Build the Docker Image

```bash
docker build -t vggt-slam:latest .
```

### Run Locally with GPU

```bash
docker run --gpus all -p 7860:7860 vggt-slam:latest
```

Or using docker-compose:

```bash
docker-compose up
```

The Gradio interface will be available at `http://localhost:7860`

## RunPod Deployment

### Option 1: Build and Push to Docker Hub

1. Build and tag the image:
```bash
docker build -t yourusername/vggt-slam:latest .
docker push yourusername/vggt-slam:latest
```

2. In RunPod, create a new pod with:
   - **Container Image**: `yourusername/vggt-slam:latest`
   - **Expose HTTP Ports**: `7860`
   - **GPU Type**: Any NVIDIA GPU with 8GB+ VRAM (RTX 3080, A10, A100, etc.)

### Option 2: Build Directly on RunPod

1. Create a RunPod pod with a GPU
2. Clone this repository
3. Build the image:
```bash
cd VGGT-SLAM
docker build -t vggt-slam:latest .
docker run --gpus all -p 7860:7860 vggt-slam:latest
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | `7860` | Server port |
| `GRADIO_SHARE` | `false` | Create public Gradio link |

## API Usage

The Gradio interface provides an API endpoint. You can interact with it programmatically:

### Python Client

```python
from gradio_client import Client

client = Client("http://localhost:7860")

# Run SLAM on a zip file of images
result = client.predict(
    image_zip="path/to/images.zip",  # ZIP file containing RGB images
    use_sim3=False,                   # Use Sim(3) instead of SL(4)
    submap_size=16,                   # Frames per submap
    max_loops=1,                      # Max loop closures per submap
    min_disparity=50.0,               # Min optical flow disparity
    conf_threshold=25.0,              # Confidence threshold
    api_name="/predict"
)

# result contains (glb_path, status_message)
glb_file, message = result
print(message)
```

### cURL

```bash
# Upload and process images
curl -X POST "http://localhost:7860/api/predict" \
  -F "data=@images.zip" \
  -F "data=false" \
  -F "data=16" \
  -F "data=1" \
  -F "data=50.0" \
  -F "data=25.0"
```

## Resource Requirements

- **GPU Memory**: 8GB+ VRAM (16GB+ recommended for large scenes)
- **RAM**: 16GB+ recommended
- **Disk**: 10GB+ for container and model weights
- **Network**: Initial model download is ~4GB

## Troubleshooting

### Out of Memory

Reduce `submap_size` or process fewer images at once.

### Slow First Request

The VGGT model (~4GB) is downloaded on first run. Subsequent requests will be faster as the model is cached.

### No GPU Detected

Ensure NVIDIA drivers and Container Toolkit are installed:
```bash
nvidia-smi  # Should show GPU info
docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi  # Test Docker GPU access
```

## Model Information

- **VGGT-1B**: Facebook's Visual Geometry Transformer (~4GB)
- **DINOv2+SALAD**: For loop closure detection
- Models are downloaded from HuggingFace on first run

## License

See the main repository LICENSE and [VGGT license](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) for model usage terms.
