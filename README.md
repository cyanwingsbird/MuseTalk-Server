# MuseTalk Server

A production-ready FastAPI wrapper for [MuseTalk](https://github.com/TMElyralab/MuseTalk), providing real-time and batch audio-driven lip-sync video generation.

## Prerequisites

- **OS**: Linux (recommended) or Windows
- **Python**: 3.10 (Conda)
- **CUDA**: 11.8, NVIDIA GPU with 8GB+ VRAM
- **PyTorch**: 2.0.1
- **FFmpeg**: installed and in PATH

## Installation

### 1. Create Conda Environment

```bash
conda create -n MuseTalk python=3.10 -y
conda activate MuseTalk
```

### 2. Install PyTorch and Critical Fixes

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Fix MKL "undefined symbol: iJIT_NotifyEvent"
conda install mkl==2023.1.0 -y

# Fix NumPy binary compatibility
conda install numpy=1.23.5 -y
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install MMLab Packages

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

### 5. Download Model Weights

Ensure `huggingface-cli` is installed (`pip install -U "huggingface_hub[cli]"`).

```bash
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

huggingface-cli download TMElyralab/MuseTalk --local-dir models --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"
huggingface-cli download TMElyralab/MuseTalk --local-dir models --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae --include "config.json" "diffusion_pytorch_model.bin"
huggingface-cli download openai/whisper-tiny --local-dir models/whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
huggingface-cli download yzd-v/DWPose --local-dir models/dwpose --include "dw-ll_ucoco_384.pth"
huggingface-cli download ByteDance/LatentSync --local-dir models/syncnet --include "latentsync_syncnet.pt"

pip install gdown
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth
```

### 6. Setup FFmpeg

```bash
conda install ffmpeg -y
# OR ensure system ffmpeg is in PATH
ffmpeg -version
```

## Running the Server

**CRITICAL**: The server must be started from the `MuseTalk/` subdirectory. The upstream code uses hardcoded relative paths.

### Linux

```bash
./start_server.sh
# Or manually:
conda activate MuseTalk
cd MuseTalk
export PYTHONPATH=$(pwd):$(dirname $(pwd))
python -m musetalk_server.app
```

### Windows

```bat
start_server.bat
```

The server starts at `http://0.0.0.0:8000` by default (model loading takes ~60s).

## Configuration

All settings via environment variables prefixed `MUSETALK_` or a `.env` file at the project root. See `.env.example` for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSETALK_HOST` | `0.0.0.0` | Server bind host |
| `MUSETALK_PORT` | `8000` | Server port |
| `MUSETALK_GPU_ID` | `0` | CUDA device ID |
| `MUSETALK_BATCH_SIZE` | `4` | Inference batch size (reduce to 2 if OOM) |
| `MUSETALK_FPS` | `25` | Output video FPS |
| `MUSETALK_RESULT_DIR` | `./results` | Directory for generated outputs |
| `MUSETALK_PARSING_MODE` | `jaw` | Face parsing mode: `jaw` or `face` |
| `MUSETALK_FFMPEG_PATH` | `ffmpeg` | Path to FFmpeg binary |
| `MUSETALK_VAE_TYPE` | `sd-vae` | VAE model type |
| `MUSETALK_UNET_CONFIG` | `./models/musetalk/musetalk.json` | UNet config path |
| `MUSETALK_UNET_MODEL_PATH` | `./models/musetalk/pytorch_model.bin` | UNet weights path |
| `MUSETALK_WHISPER_DIR` | `./models/whisper` | Whisper model directory |

## API Reference

### System Status

```
GET /health
```

Returns server status, model load state, and cached avatars.

### List Avatars

```
GET /avatars
```

Returns list of preprocessed avatars available on disk.

### Preprocess Avatar (one-time per video)

```
POST /avatars/preprocess
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `avatar_id` | form (string) | Unique ID matching `^[a-zA-Z0-9_\-]{1,64}$` |
| `bbox_shift` | form (int) | Bounding box shift (default 0) |
| `video_file` | file | Source video (MP4) |

### Streaming Inference (MJPEG)

```
POST /inference/stream/{avatar_id}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_file` | file | Input audio (WAV/MP3) |
| `batch_size` | form (int, optional) | Override default batch size (1-32) |

Returns `multipart/x-mixed-replace` MJPEG stream.

### Batch Inference (MP4)

```
POST /inference/batch/{avatar_id}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_file` | file | Input audio (WAV/MP3) |
| `batch_size` | form (int, optional) | Override default batch size (1-32) |

Returns MP4 file download.

### Example Usage

```bash
# Check health
curl http://localhost:8000/health

# Preprocess avatar
curl -X POST "http://localhost:8000/avatars/preprocess" \
  -F "avatar_id=my_avatar" \
  -F "video_file=@/path/to/video.mp4"

# Batch inference
curl -X POST "http://localhost:8000/inference/batch/my_avatar" \
  -F "audio_file=@/path/to/audio.wav" \
  --output result.mp4

# Batch inference with custom batch size
curl -X POST "http://localhost:8000/inference/batch/my_avatar" \
  -F "audio_file=@/path/to/audio.wav" \
  -F "batch_size=2" \
  --output result.mp4

# Streaming inference
curl -X POST "http://localhost:8000/inference/stream/my_avatar" \
  -F "audio_file=@/path/to/audio.wav"
```

### Python Client

```python
import httpx

# Preprocess (once)
files = {"video_file": open("video.mp4", "rb")}
data = {"avatar_id": "avatar1"}
httpx.post("http://localhost:8000/avatars/preprocess", data=data, files=files)

# Streaming inference
files = {"audio_file": open("audio.wav", "rb")}
with httpx.stream("POST", "http://localhost:8000/inference/stream/avatar1", files=files) as r:
    for chunk in r.iter_bytes():
        pass  # process MJPEG chunk
```

## Development and Testing

### Unit Tests (no GPU required)

```bash
PYTHONPATH=".:./MuseTalk" pytest musetalk_server/tests/test_api.py -v
```

### Full Integration Verification (requires GPU + models)

```bash
bash test/run_verify.sh
```

## Common Pitfalls

- **CWD Trap**: Server must run from `MuseTalk/` directory or model paths break
- **Import disambiguation**: `import musetalk` = upstream ML module; `import musetalk_server` = this project
- **Model loading**: Takes ~60s; `/health` returns `models.loaded: false` until complete
- **OOM errors**: Reduce `MUSETALK_BATCH_SIZE` to 2, set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
- **Port conflicts**: Check for zombie `python -m musetalk_server` processes
- **Model weights**: Not tracked in git; must be downloaded separately

## Deployment

- Dockerfile must use `WORKDIR /app/MuseTalk`
- Ensure GPU passthrough (`--gpus all`)
- Models must be present in `MuseTalk/models/`
