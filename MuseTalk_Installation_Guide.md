# MuseTalk Installation Guide (Workable Version)

This guide provides a step-by-step installation procedure for [MuseTalk](https://github.com/TMElyralab/MuseTalk) that resolves common dependency conflicts (MKL, Numpy) encountered during standard setup.

**Tested Environment:**
- **OS**: Linux
- **Python**: 3.10
- **CUDA**: 11.8
- **PyTorch**: 2.0.1

---

## 1. Create Conda Environment

Create and activate a new environment with Python 3.10.

```bash
conda create -n MuseTalk python=3.10 -y
conda activate MuseTalk
```

## 2. Install PyTorch & Core Dependencies

Install PyTorch 2.0.1 with CUDA 11.8 support.

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**Critical Fixes:**
Downgrade MKL and Numpy to prevent "undefined symbol" and binary incompatibility errors.

```bash
# Fix 1: Downgrade MKL to avoid "undefined symbol: iJIT_NotifyEvent"
conda install mkl==2023.1.0 -y

# Fix 2: Pin Numpy to 1.23.5 to match binary expectations
conda install numpy=1.23.5 -y
```

## 3. Install Python Requirements

Install the general dependencies from the project root.

```bash
pip install -r requirements.txt
```

## 4. Install MMLab Packages

Install OpenMIM and the required MMLab libraries.

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

## 5. Download Model Weights

Download all required checkpoints. Ensure you have `huggingface-cli` installed (`pip install -U "huggingface_hub[cli]"`).

*Note: If you are in a region with restricted access to HuggingFace, set the mirror endpoint:*
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Run the following commands to download weights into the `models/` directory:

```bash
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# MuseTalk V1.0
huggingface-cli download TMElyralab/MuseTalk --local-dir models --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# MuseTalk V1.5
huggingface-cli download TMElyralab/MuseTalk --local-dir models --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# SD-VAE
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae --include "config.json" "diffusion_pytorch_model.bin"

# Whisper
huggingface-cli download openai/whisper-tiny --local-dir models/whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# DWPose
huggingface-cli download yzd-v/DWPose --local-dir models/dwpose --include "dw-ll_ucoco_384.pth"

# SyncNet
huggingface-cli download ByteDance/LatentSync --local-dir models/syncnet --include "latentsync_syncnet.pt"

# Face Parse (using gdown and curl)
pip install gdown
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth
```

## 6. Setup FFmpeg

Ensure FFmpeg is installed and accessible.

```bash
conda install ffmpeg -y
# OR ensure system ffmpeg is in PATH
ffmpeg -version
```

## 7. Run Inference

To verify the installation, run the inference script.

```bash
# Run MuseTalk v1.5 inference (Recommended)
sh inference.sh v1.5 normal
```

The results will be generated in `results/test/v15/`.
