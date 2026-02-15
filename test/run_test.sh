#!/bin/bash
# Use active environment python (avoid absolute paths)
PYTHON_BIN="${PYTHON:-python}"

# Start server in background
echo "Starting server..."
"$PYTHON_BIN" -m musetalk_server.app > server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..30}; do
    if grep -q "Application startup complete" server.log; then
        echo "Server started!"
        break
    fi
    sleep 2
done

# Run client
echo "Running client..."
"$PYTHON_BIN" client_example.py

# Cleanup
echo "Killing server..."
kill $SERVER_PID
