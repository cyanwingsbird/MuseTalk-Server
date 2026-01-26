from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
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
    audio_file: UploadFile = File(...)
):
    """
    Real-time streaming inference. Returns an MJPEG stream.
    """
    # Get avatar
    avatar = get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found. Preprocess it first.")
    
    # Save audio temporarily
    temp_id = str(uuid.uuid4())
    audio_path = f"/tmp/{temp_id}.wav" # Using /tmp for transient files
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    # Get models
    models = model_loader.get_models()

    service = InferenceService(models, settings)
    
    # Clean up audio after response (using background task trick in generator if needed, 
    # but for simple streaming, we might just leave it or use a proper tempfile manager)
    
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
    background_tasks: BackgroundTasks = None
):
    """
    Batch inference. Generates a full MP4 video and returns it.
    """
    # Get avatar
    avatar = get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")

    # Save audio
    temp_id = str(uuid.uuid4())
    temp_dir = os.path.join(settings.result_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, f"{temp_id}.wav")
    
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Get models
    models = model_loader.get_models()
    service = InferenceService(models, settings)

    try:
        # Run inference
        output_path = service.inference_batch(avatar, audio_path)
        
        # Schedule cleanup
        # background_tasks.add_task(os.remove, audio_path) # Cleanup handled by service/temp logic usually
        
        return FileResponse(output_path, media_type="video/mp4", filename=f"{avatar_id}_{temp_id}.mp4")
        
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
