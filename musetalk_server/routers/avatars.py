from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from musetalk_server.conf import conf as settings
from musetalk_server.core.model_loader import model_loader
from musetalk_server.core.avatar import Avatar
from musetalk_server.services.preprocess import AvatarPreprocessor
from musetalk_server.schemas.api import PreprocessResponse, AvatarInfo
import shutil
import os
import json
import traceback

router = APIRouter()
avatars = {} # In-memory cache for loaded avatars

@router.get("/avatars", response_model=list[str])
def list_avatars():
    """List all available (preprocessed) avatars on disk."""
    avatar_root = os.path.join(settings.result_dir, settings.version, "avatars")
    if not os.path.exists(avatar_root):
        return []
    return [d for d in os.listdir(avatar_root) if os.path.isdir(os.path.join(avatar_root, d))]

@router.post("/avatars/preprocess", response_model=PreprocessResponse)
async def preprocess_avatar(
    avatar_id: str = Form(...),
    video_file: UploadFile = File(...),
    bbox_shift: int = Form(0)
):
    """
    Upload a video and preprocess it to create an avatar.
    This is a blocking operation for simplicity, but in production should be backgrounded.
    """
    # Save uploaded video
    video_dir = os.path.join(settings.result_dir, "uploads")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{avatar_id}_{video_file.filename}")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    
    try:
        # Load models if not loaded (lazy loading)
        models = model_loader.get_models()
        
        # Run preprocessing
        preprocessor = AvatarPreprocessor(models['vae'], models['face_parsing'])
        preprocessor.process_avatar(
            video_path,
            avatar_id,
            bbox_shift,
            results_dir=settings.result_dir,
            extra_margin=settings.extra_margin,
            parsing_mode=settings.parsing_mode,
            version=settings.version
        )
        
        # Load the avatar into memory to verify it works and cache it
        avatar = Avatar(avatar_id, results_dir=settings.result_dir, version=settings.version)
        avatar.load_state()
        avatars[avatar_id] = avatar
        
        return PreprocessResponse(
            message="Avatar processed successfully",
            avatar_id=avatar_id,
            info=AvatarInfo(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=bbox_shift,
                version=settings.version
            )
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

def get_avatar(avatar_id: str) -> Avatar:
    """Helper to get avatar from cache or load from disk"""
    if avatar_id in avatars:
        return avatars[avatar_id]
    
    # Try to load from disk
    try:
        avatar = Avatar(avatar_id, results_dir=settings.result_dir, version=settings.version)
        avatar.load_state()
        avatars[avatar_id] = avatar
        return avatar
    except Exception as e:
        print(f"Failed to load avatar {avatar_id}: {e}")
        return None
