from fastapi import APIRouter
from musetalk_server.core.model_loader import model_loader
from musetalk_server.routers.avatars import avatars
from musetalk_server.schemas.api import SystemStatus, ModelStatus
import torch

router = APIRouter()

@router.get("/health", response_model=SystemStatus)
def health_check():
    # model_loader is the instance
    models_loaded = model_loader.is_loaded()
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    return SystemStatus(
        status="running",
        models=ModelStatus(
            loaded=models_loaded,
            device=device_name
        ),
        loaded_avatars=list(avatars.keys())
    )
