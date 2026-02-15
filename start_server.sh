#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Starting MuseTalk Server..."

# Get project root directory (directory containing this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment (default: MuseTalk)
CONDA_ENV_NAME="${MUSETALK_CONDA_ENV:-MuseTalk}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found. Please install conda/miniconda first."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] Could not find conda initialization script at $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
    echo "[INFO] Activating conda environment: $CONDA_ENV_NAME"
    conda activate "$CONDA_ENV_NAME"
fi

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found after activating conda environment."
    exit 1
fi

# Set PYTHONPATH to include project root and MuseTalk submodule
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/MuseTalk:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/MuseTalk"
fi
echo "[INFO] PYTHONPATH set to: $PYTHONPATH"

# Change directory to MuseTalk so relative model paths resolve correctly
cd "$PROJECT_ROOT/MuseTalk"
echo "[INFO] Working Directory: $PWD"

# Run the server using the venv python
echo "[INFO] Server is starting... (This may take nearly a minute to load models)"
echo "[INFO] Please keep this terminal open."
"$PYTHON_BIN" -m musetalk_server.app
