#!/bin/bash
set -e

# Get the root directory
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MUSETALK_DIR="$ROOT_DIR/MuseTalk"
PYTHON_BIN="${PYTHON:-python}"

# Set PYTHONPATH
export PYTHONPATH="$MUSETALK_DIR:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Running unit tests..."
echo "  PYTHONPATH: $PYTHONPATH"

# Run from MuseTalk dir so upstream imports resolve
cd "$MUSETALK_DIR"

"$PYTHON_BIN" -m pytest "$ROOT_DIR/musetalk_server/tests/test_api.py" -v "$@"
