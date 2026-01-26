from pydantic import BaseModel
from typing import List, Optional

class ModelStatus(BaseModel):
    loaded: bool
    device: str

class SystemStatus(BaseModel):
    status: str
    models: ModelStatus
    loaded_avatars: List[str]

class AvatarInfo(BaseModel):
    avatar_id: str
    video_path: str
    bbox_shift: int
    version: str

class PreprocessResponse(BaseModel):
    message: str
    avatar_id: str
    info: AvatarInfo
