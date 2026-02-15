# MuseTalk Server

A production-ready FastAPI wrapper for [MuseTalk](https://github.com/TMElyralab/MuseTalk), providing real-time and batch lip-sync generation. This server allows you to generate talking avatar videos from audio inputs using pre-processed avatar videos.

## 🚀 Quick Start

### Prerequisites
- Conda environment with MuseTalk dependencies installed
- NVIDIA GPU with CUDA support
- `ffmpeg` installed on the system

### Installation
Clone this repository (MuseTalk submodule must be present):
```bash
git clone <repo_url>
cd muse-server
# Ensure MuseTalk submodule is initialized if using git
```

### Running the Server
**CRITICAL:** You must run the server from the `MuseTalk` directory due to upstream path dependencies.

Linux quick start:
```bash
./start_server.sh
```

Manual start (equivalent):

```bash
# 1. Activate environment
conda activate MuseTalk

# 2. Navigate to MuseTalk directory
cd MuseTalk

# 3. Set PYTHONPATH to include both MuseTalk and the server root
export PYTHONPATH=$(pwd):$(dirname $(pwd))

# 4. Start the server
python -m musetalk_server.app
```

The server will start at `http://0.0.0.0:8000`.

## 📚 API Usage

### 1. Check Health
```bash
curl http://localhost:8000/health
```

### 2. Preprocess an Avatar (One-time)
Upload a source video to create an avatar identity.
```bash
curl -X POST "http://localhost:8000/avatars/preprocess" \
  -F "file=@/path/to/video.mp4"
# Returns: {"avatar_id": "video_file_name"}
```

### 3. Generate Lip-Sync (Batch)
Generate a full video file from audio.
```bash
curl -X POST "http://localhost:8000/inference/batch/video_file_name" \
  -F "audio=@/path/to/audio.wav" \
  --output result.mp4
```

### 4. Real-time Streaming
Stream MJPEG frames for real-time applications.
```bash
curl -X POST "http://localhost:8000/inference/stream/video_file_name" \
  -F "audio=@/path/to/audio.wav"
```

## ⚙️ Configuration
Configuration is managed in `musetalk_server/conf.py`. You can override settings via environment variables prefixed with `MUSETALK_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSETALK_HOST` | `0.0.0.0` | Server bind host (set to 0.0.0.0 for LAN access) |
| `MUSETALK_PORT` | `8000` | Server port |
| `MUSETALK_GPU_ID` | `0` | CUDA device ID |
| `MUSETALK_WORKERS` | `1` | Number of Uvicorn workers |

## 🧪 Verification
Run the verification script to test the full pipeline:
```bash
bash test/run_verify.sh
```
