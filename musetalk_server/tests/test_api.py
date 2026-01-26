from fastapi.testclient import TestClient
from musetalk_server.app import app
from musetalk_server.conf import conf as settings
import pytest
import os
import shutil

# Mocking the models/inference for CI/fast testing
# In a real scenario, we might want full integration tests, but without GPUs it will fail or be slow
# For now, let's verify the API endpoints respond correctly

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "models" in data
    assert "loaded_avatars" in data

def test_list_avatars():
    response = client.get("/avatars")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Note: Actual preprocessing and inference tests require large model files and GPU
# We can skip them if environment variables indicate non-GPU env or mock them
# Here we just verify the route existence and validation errors

def test_inference_stream_missing_avatar():
    response = client.post(
        "/inference/stream/nonexistent_avatar",
        files={"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    )
    assert response.status_code == 404

def test_inference_batch_missing_avatar():
    response = client.post(
        "/inference/batch/nonexistent_avatar",
        files={"audio_file": ("test.wav", b"fake audio data", "audio/wav")}
    )
    assert response.status_code == 404
