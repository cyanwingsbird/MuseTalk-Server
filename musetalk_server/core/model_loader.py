import torch
import os
import gc
from transformers import WhisperModel
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk_server.conf import conf

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self.device = torch.device(f"cuda:{conf.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.vae = None
        self.unet = None
        self.pe = None
        self.audio_processor = None
        self.whisper = None
        self.face_parsing = None
        self.timesteps = None
        
        self.initialized = True

    def load(self):
        """
        Loads all required models into memory if they aren't already loaded.
        """
        if self.unet is not None:
            # Already loaded
            return

        print(f"Loading models on device: {self.device}...")
        
        # Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 1. Load VAE, UNet, PE
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=conf.unet_model_path,
            vae_type=conf.vae_type,
            unet_config=conf.unet_config,
            device=self.device
        )

        # Move to device and half precision
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.timesteps = torch.tensor([0], device=self.device)
        
        # Clear cache after loading models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 2. Audio Processor
        self.audio_processor = AudioProcessor(feature_extractor_path=conf.whisper_dir)

        # 3. Whisper
        # Use same dtype as UNet
        weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(conf.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=weight_dtype).eval()
        self.whisper.requires_grad_(False)
        
        # Clear cache after loading Whisper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 4. Face Parsing
        if conf.version == "v15":
            self.face_parsing = FaceParsing(
                left_cheek_width=conf.left_cheek_width,
                right_cheek_width=conf.right_cheek_width
            )
        else:
            self.face_parsing = FaceParsing()
            
        print("All models loaded successfully.")
        
        # Final cache clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def is_loaded(self) -> bool:
        return self.unet is not None

    def get_models(self):
        """
        Ensure models are loaded and return them.
        """
        if self.unet is None:
            self.load()
        
        models = {
            "vae": self.vae,
            "unet": self.unet,
            "pe": self.pe,
            "audio_processor": self.audio_processor,
            "whisper": self.whisper,
            "face_parsing": self.face_parsing,
            "timesteps": self.timesteps,
            "device": self.device
        }
        return models

# Global singleton instance
model_loader = ModelLoader()
