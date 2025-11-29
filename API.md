# VGGT-SLAM API Documentation

The VGGT-SLAM service exposes a Gradio-based API for 3D scene reconstruction from RGB images.

## Base URL

```
http://<host>:7860
```

For RunPod deployments, use the provided proxy URL (e.g., `https://<pod-id>-7860.proxy.runpod.net`).

## Endpoints

### Health Check

```
GET /
```

Returns the Gradio web interface. Use this to verify the service is running.

### API Info

```
GET /info
```

Returns API metadata and available endpoints.

### Predict (Main SLAM Endpoint)

```
POST /api/predict
```

Runs VGGT-SLAM on uploaded images and returns a 3D reconstruction.

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_zip` | File (.zip) | Yes | - | ZIP file containing RGB images (.jpg, .jpeg, .png) |
| `use_sim3` | boolean | No | `false` | Use Sim(3) optimization instead of SL(4). Sim(3) has less drift but may not align as well. |
| `submap_size` | integer | No | `16` | Number of frames per submap (range: 4-32) |
| `max_loops` | integer | No | `1` | Maximum loop closures to add per new submap (range: 0-5) |
| `min_disparity` | float | No | `50.0` | Minimum optical flow disparity between keyframes (range: 0-100) |
| `conf_threshold` | float | No | `25.0` | Confidence threshold percentile for point filtering. Higher = fewer points (range: 0-100) |

## Output

Returns a tuple of two values:

| Index | Type | Description |
|-------|------|-------------|
| 0 | File path (string) | Path to the generated .glb 3D model file |
| 1 | string | Status message with submap and loop closure counts |

## Client Examples

### Python (using gradio_client)

```python
from gradio_client import Client

# Connect to the API
client = Client("http://localhost:7860")

# Run SLAM with default parameters
result = client.predict(
    image_zip="path/to/images.zip",
    api_name="/predict"
)

glb_path, message = result
print(f"Status: {message}")
print(f"3D Model: {glb_path}")
```

### Python (with custom parameters)

```python
from gradio_client import Client

client = Client("http://localhost:7860")

result = client.predict(
    image_zip="path/to/images.zip",
    use_sim3=False,           # Use SL(4) optimization
    submap_size=16,           # 16 frames per submap
    max_loops=1,              # 1 loop closure per submap
    min_disparity=50.0,       # Keyframe selection threshold
    conf_threshold=25.0,      # Point confidence filter
    api_name="/predict"
)

glb_path, message = result
```

### Python (using requests)

```python
import requests

url = "http://localhost:7860/api/predict"

# Prepare the file and parameters
with open("images.zip", "rb") as f:
    files = {"data": f}
    data = {
        "data": [
            None,  # File will be uploaded separately
            False,  # use_sim3
            16,     # submap_size
            1,      # max_loops
            50.0,   # min_disparity
            25.0    # conf_threshold
        ]
    }

    response = requests.post(url, files=files, json=data)
    result = response.json()
```

### cURL

```bash
# Basic request
curl -X POST "http://localhost:7860/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@images.zip"

# With parameters (using Gradio's format)
curl -X POST "http://localhost:7860/run/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"path": "images.zip", "meta": {"_type": "gradio.FileData"}},
      false,
      16,
      1,
      50.0,
      25.0
    ]
  }'
```

### JavaScript/TypeScript

```typescript
import { Client } from "@gradio/client";

async function runSLAM(zipFile: File) {
  const client = await Client.connect("http://localhost:7860");

  const result = await client.predict("/predict", {
    image_zip: zipFile,
    use_sim3: false,
    submap_size: 16,
    max_loops: 1,
    min_disparity: 50.0,
    conf_threshold: 25.0,
  });

  const [glbPath, message] = result.data;
  console.log("Status:", message);
  console.log("3D Model:", glbPath);

  return result;
}
```

### JavaScript (fetch)

```javascript
async function runSLAM(zipFile) {
  const formData = new FormData();
  formData.append("files", zipFile);

  const response = await fetch("http://localhost:7860/api/predict", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  return result;
}
```

## Downloading the 3D Model

The API returns a file path on the server. To download the .glb file:

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://localhost:7860")

result = client.predict(
    image_zip="path/to/images.zip",
    api_name="/predict"
)

glb_path, message = result

# Download the file
import urllib.request
urllib.request.urlretrieve(
    f"http://localhost:7860/file={glb_path}",
    "output.glb"
)
```

### Using gradio_client file handling

```python
from gradio_client import Client

client = Client("http://localhost:7860")

# The client automatically downloads files to a temp directory
result = client.predict(
    image_zip="path/to/images.zip",
    api_name="/predict"
)

glb_local_path = result[0]  # Local path to downloaded .glb file
print(f"Downloaded to: {glb_local_path}")
```

## Input Image Requirements

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)

### Image Guidelines
- RGB images only (depth images are automatically filtered out)
- Images should be from a continuous trajectory
- Minimum ~20 images recommended for meaningful reconstruction
- Higher resolution = better quality but slower processing

### ZIP Structure

The ZIP file can contain images directly or in subdirectories:

```
images.zip
├── frame_0001.jpg
├── frame_0002.jpg
├── frame_0003.jpg
└── ...
```

Or nested:

```
images.zip
└── my_scene/
    ├── rgb/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
```

Images are automatically sorted by numeric values in filenames.

## Error Handling

### Error Response Format

```json
{
  "data": [
    null,
    "Error: <error message>\n<stack trace>"
  ]
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "No valid images found" | ZIP contains no .jpg/.jpeg/.png files | Check ZIP contents and file extensions |
| "CUDA out of memory" | GPU memory exhausted | Reduce `submap_size` or image resolution |
| "Model loading failed" | Network/download issue | Retry; model downloads on first request |

## Performance Tips

1. **First Request**: The first request downloads the VGGT model (~4GB). Subsequent requests are faster.

2. **Submap Size**: Smaller `submap_size` (e.g., 8) uses less memory but may produce more drift.

3. **Loop Closures**: Increase `max_loops` for scenes with revisited areas to improve accuracy.

4. **Disparity Threshold**: Lower `min_disparity` keeps more frames but may slow processing.

5. **GPU Memory**: For GPUs with limited VRAM (<8GB), use smaller images or reduce `submap_size`.

## Rate Limits

The Gradio queue handles concurrent requests automatically. For production deployments, consider:
- Setting `GRADIO_SHARE=false` to disable public links
- Implementing authentication via a reverse proxy
- Monitoring GPU memory usage

## Response Times

Approximate processing times (on NVIDIA A10 GPU):

| Scene Size | Images | Processing Time |
|------------|--------|-----------------|
| Small | 50 images | ~30 seconds |
| Medium | 150 images | ~2 minutes |
| Large | 500 images | ~5-10 minutes |

Times vary based on image resolution, GPU, and parameters.
