from pydantic_settings import BaseSettings
from typing import Optional
import os

class MuseTalkSettings(BaseSettings):
    """
    Configuration for MuseTalk Server.
    Defaults match scripts/realtime_inference.py arguments.
    """
    version: str = "v15"
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
    batch_size: int = 20
    output_vid_name: Optional[str] = None
    use_saved_coord: bool = False
    saved_coord: bool = False
    parsing_mode: str = "jaw"
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    skip_save_images: bool = False

    class Config:
        env_prefix = "MUSETALK_"
        env_file = ".env"

# Create a global instance
conf = MuseTalkSettings()
