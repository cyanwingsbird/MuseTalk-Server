import sys
import torch
import os
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from musetalk.utils.face_parsing import FaceParsing
import scripts.realtime_inference as realtime_module

# Global cache
loaded_models = {}

def load_models(args):
    if loaded_models:
        return loaded_models
    
    print(f"Loading models with config: {args}")
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    # Move to device and half precision (CUDA only)
    if device.type == "cuda":
        try:
            pe = pe.half().to(device)
            vae.vae = vae.vae.half().to(device)
            unet.model = unet.model.half().to(device)
        except Exception as e:
            print(f"Warning: Failed to convert to half precision or move to device: {e}")
            # Fallback if needed, but MuseTalk expects half
    else:
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)

    print("Loading Whisper...")
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    print("Loading FaceParsing...")
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:
        fp = FaceParsing()
        
    loaded_models.update({
        'vae': vae,
        'unet': unet,
        'pe': pe,
        'timesteps': timesteps,
        'audio_processor': audio_processor,
        'whisper': whisper,
        'fp': fp,
        'device': device,
        'weight_dtype': weight_dtype
    })
    
    # Patch the module
    print("Patching realtime_inference module...")
    realtime_module.args = args
    realtime_module.device = device
    realtime_module.vae = vae
    realtime_module.unet = unet
    realtime_module.pe = pe
    realtime_module.timesteps = timesteps
    realtime_module.audio_processor = audio_processor
    realtime_module.whisper = whisper
    realtime_module.fp = fp
    realtime_module.weight_dtype = weight_dtype
    
    return loaded_models
