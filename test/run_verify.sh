#!/bin/bash
set -e

echo "=== Starting MuseTalk Verification ==="

# Get the root directory (where this script lives is test/, so go up one level)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MUSETALK_DIR="$ROOT_DIR/MuseTalk"

# Set PYTHONPATH to include both MuseTalk (for musetalk module) and root (for musetalk_server)
export PYTHONPATH="$MUSETALK_DIR:$ROOT_DIR:$PYTHONPATH"

# Cleanup previous run
pkill -f musetalk_server.app || true
rm -f "$ROOT_DIR/test/result.mp4"

# Start Server FROM MuseTalk directory (required for hardcoded relative paths in MuseTalk)
echo "Starting server in background from MuseTalk directory..."
cd "$MUSETALK_DIR"
/home/andy/miniconda3/envs/MuseTalk/bin/python -u -m musetalk_server.app > "$ROOT_DIR/test/server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Return to root for verification script
cd "$ROOT_DIR"

# Give server a moment to initialize (though verify script waits)
sleep 2

# Run Verification Script
echo "Running verification client..."
/home/andy/miniconda3/envs/MuseTalk/bin/python -u test/verify_server.py
RET=$?

if [ $RET -eq 0 ]; then
    echo "=== Verification SUCCESS ==="
    RESULT=0
else
    echo "=== Verification FAILED (Exit Code: $RET) ==="
    RESULT=1
    echo "--- Server Logs (Last 50 lines) ---"
    tail -n 50 test/server.log
fi

# Cleanup
echo "Stopping server..."
kill $SERVER_PID || true
wait $SERVER_PID || true

exit $RESULT
