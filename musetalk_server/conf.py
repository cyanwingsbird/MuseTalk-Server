from pydantic_settings import BaseSettings
from typing import Optional
import os

class MuseTalkSettings(BaseSettings):
    """
    Configuration for MuseTalk Server.
    Defaults match scripts/realtime_inference.py arguments.
    """
    version: str = "v15"
    port: int = 8000
    ffmpeg_path: str = "ffmpeg"
    gpu_id: int = 0
    vae_type: str = "sd-vae"
    unet_config: str = "./models/musetalk/musetalk.json"
    unet_model_path: str = "./models/musetalk/pytorch_model.bin"
    whisper_dir: str = "./models/whisper"
    inference_config: str = "./configs/inference/realtime.yaml"
    bbox_shift: int = 0
    result_dir: str = "./results"
    extra_margin: int = 10
    fps: int = 25
    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2
    batch_size: int = 4  # Reduced from 20 to prevent OOM errors
    output_vid_name: Optional[str] = None
    use_saved_coord: bool = False
    saved_coord: bool = False
    parsing_mode: str = "jaw"
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    skip_save_images: bool = False

    class Config:
        env_prefix = "MUSETALK_"
        # Load .env from project root (one directory up from musetalk_server package)
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

# Create a global instance
conf = MuseTalkSettings()
