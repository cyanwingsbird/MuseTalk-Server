from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from musetalk_server.core.model_loader import model_loader
from musetalk_server.services.inference import InferenceService
from musetalk_server.routers.avatars import get_avatar
from musetalk_server.conf import conf as settings
import shutil
import os
import uuid

router = APIRouter()

@router.post("/inference/stream/{avatar_id}")
async def stream_inference(
    avatar_id: str,
    audio_file: UploadFile = File(...),
    batch_size: Optional[int] = Form(None, description="Override default batch size (1-32)", ge=1, le=32)
):
    """
    Real-time streaming inference. Returns an MJPEG stream.
    """
    avatar = get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found. Preprocess it first.")

    temp_id = str(uuid.uuid4())
    audio_path = os.path.join(settings.result_dir, "temp", f"{temp_id}.wav")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    models = model_loader.get_models()
    service = InferenceService(models, settings, batch_size_override=batch_size)

    def iterfile():
        try:
            for frame_bytes in service.inference_stream(avatar, audio_path):
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    return StreamingResponse(iterfile(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.post("/inference/batch/{avatar_id}")
async def batch_inference(
    avatar_id: str,
    audio_file: UploadFile = File(...),
    batch_size: Optional[int] = Form(None, description="Override default batch size (1-32)", ge=1, le=32)
):
    """
    Batch inference. Generates a full MP4 video and returns it.
    """
    avatar = get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    temp_id = str(uuid.uuid4())
    temp_dir = os.path.join(settings.result_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, f"{temp_id}.wav")

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    models = model_loader.get_models()
    service = InferenceService(models, settings, batch_size_override=batch_size)

    try:
        output_path = service.inference_batch(avatar, audio_path)
        return FileResponse(output_path, media_type="video/mp4", filename=f"{avatar_id}_{temp_id}.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
