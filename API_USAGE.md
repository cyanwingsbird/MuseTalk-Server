# MuseTalk Server API Documentation

## Overview
MuseTalk Server provides a high-fidelity audio-driven lip-syncing service. It supports both real-time streaming (MJPEG) and batch processing (MP4).

## Base URL
`http://localhost:8000` (Default)

## Concepts
*   **Avatar**: A preprocessed video profile. You must preprocess a video once to create an "Avatar" (identified by `avatar_id`) before you can run inference on it.
*   **Inference**: Driving an existing Avatar with new Audio.

---

## Endpoints

### 1. System Status
Check if the server is running and models are loaded.

*   **URL**: `/health`
*   **Method**: `GET`
*   **Response**:
    ```json
    {
      "status": "running",
      "models": {
        "loaded": true,
        "device": "cuda:0"
      },
      "loaded_avatars": ["avatar1", "avatar2"]
    }
    ```

### 2. Preprocess Avatar
Upload a video to create a new avatar. This is a heavy operation.

*   **URL**: `/avatars/preprocess`
*   **Method**: `POST`
*   **Form Data**:
    *   `avatar_id` (string): Unique identifier for the new avatar.
    *   `bbox_shift` (int, default=0): Adjusts face bounding box (positive moves down, negative moves up).
*   **Files**:
    *   `video_file`: The source video file (MP4).
*   **Response**:
    ```json
    {
      "message": "Avatar processed successfully",
      "avatar_id": "my_avatar",
      "info": { ... }
    }
    ```

### 3. Real-time Streaming Inference
Drive an avatar with audio and get a video stream back immediately.

*   **URL**: `/inference/stream/{avatar_id}`
*   **Method**: `POST`
*   **Files**:
    *   `audio_file`: The input audio file (WAV/MP3).
*   **Response**: `multipart/x-mixed-replace` stream (MJPEG). Each part is a JPEG frame.
*   **Note**: This endpoint is designed for low-latency playback.

### 4. Batch Inference
Generate a full MP4 video file.

*   **URL**: `/inference/batch/{avatar_id}`
*   **Method**: `POST`
*   **Files**:
    *   `audio_file`: The input audio file.
*   **Response**: Binary MP4 file download.

---

## Python Client Example

See `client_example.py` for a full working implementation using `httpx`.

```python
import httpx

# 1. Preprocess (Once)
files = {"video_file": open("video.mp4", "rb")}
data = {"avatar_id": "avatar1"}
httpx.post("http://localhost:8000/avatars/preprocess", data=data, files=files)

# 2. Inference (Streaming)
files = {"audio_file": open("audio.wav", "rb")}
with httpx.stream("POST", "http://localhost:8000/inference/stream/avatar1", files=files) as r:
    for chunk in r.iter_bytes():
        # Process MJPEG chunk
        pass
```

## Environment Variables
Configure the server via `.env` file or environment variables prefixed with `MUSETALK_`:

*   `MUSETALK_GPU_ID`: GPU index (default 0).
*   `MUSETALK_PORT`: Port (default 8000).
*   `MUSETALK_RESULT_DIR`: Storage path (default `./results`).

## Error Handling
*   **404 Not Found**: Avatar ID does not exist (needs preprocessing).
*   **500 Internal Server Error**: Model failure or GPU error.
