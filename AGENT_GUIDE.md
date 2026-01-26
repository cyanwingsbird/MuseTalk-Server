# AGENT_GUIDE.md

## Project Context
This is a FastAPI wrapper around the **MuseTalk** real-time lip-sync model. It serves as an inference server for generating talking head videos from audio.

## 🏗️ Architecture
- **Root**: `musetalk_server/` (FastAPI app)
- **Upstream**: `MuseTalk/` (Original research repo, UNMODIFIED)
- **Runtime**: Python 3.10+ (Conda: `MuseTalk`)

## ⚠️ Critical Implementation Details

### 1. Execution Context (The "CWD Trap")
The upstream `MuseTalk` code (specifically `preprocessing.py` and `utils/`) uses **hardcoded relative paths** (e.g., `./models/dwpose/...`).
- **RULE**: The server process **MUST** be started with `cwd` set to `MuseTalk/`.
- **RULE**: `PYTHONPATH` must include both `$PWD` (MuseTalk) and `..` (project root).
- **VIOLATION**: Running from root will cause `FileNotFoundError` for models.

### 2. Path Management
- **Models**: Located in `MuseTalk/models/`.
- **Config**: `musetalk_server/conf.py` maps server paths to `MuseTalk/` relative paths.
- **Do NOT** use symlinks at root (we removed them to keep the structure clean).

### 3. Key Components
- **`musetalk_server.app`**: Entrypoint. Loads models on startup (slow, ~60s).
- **`core/model_loader.py`**: Singleton for `AudioProcessor`, `Inference`, `FaceAnalysis`.
- **`services/inference.py`**: Handles the generation pipeline (Audio -> Features -> UNet -> Blending).

## 🛠️ Development & Testing

### Verification
Use `test/run_verify.sh` for all testing. It handles the environment setup correctly:
```bash
bash test/run_verify.sh
```
This script:
1. Sets PYTHONPATH
2. `cd MuseTalk`
3. Starts server in background
4. Runs `test/verify_server.py` client test

### Common Pitfalls
- **Timeout**: Inference is heavy. 60s audio can take >60s on weak GPUs.
- **Port Conflicts**: Server uses port 8000. Ensure it's free or check for zombie `python -m musetalk_server` processes.
- **Imports**: `import musetalk` refers to the upstream module. `import musetalk_server` is this project.

## 📦 Deployment
- Dockerfile should `WORKDIR /app/MuseTalk`.
- Ensure GPU passthrough (`--gpus all`).
- Models must be present in `MuseTalk/models/`.
