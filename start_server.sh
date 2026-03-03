#!/bin/bash
set -e

# Get the project root directory
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MUSETALK_DIR="$ROOT_DIR/MuseTalk"

if [ ! -d "$MUSETALK_DIR" ]; then
    echo "ERROR: MuseTalk directory not found at $MUSETALK_DIR"
    echo "Ensure the MuseTalk submodule is initialized."
    exit 1
fi

# Set PYTHONPATH to include both MuseTalk (for musetalk module) and root (for musetalk_server)
export PYTHONPATH="$MUSETALK_DIR:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Must run from MuseTalk directory (upstream code uses hardcoded relative paths)
cd "$MUSETALK_DIR"

echo "Starting MuseTalk Server..."
echo "  CWD:        $(pwd)"
echo "  PYTHONPATH:  $PYTHONPATH"

# Use active environment python
PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -m musetalk_server.app "$@"
