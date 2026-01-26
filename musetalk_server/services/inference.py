import queue
import threading
import time
import copy
import cv2
import torch
import numpy as np
import os
import subprocess
import shutil
from typing import Generator, Optional
from musetalk.utils.utils import datagen
from musetalk.utils.blending import get_image_blending

class InferenceModels:
    """
    Container for the loaded models required for inference.
    """
    def __init__(self, vae, unet, pe, audio_processor, whisper, timesteps):
        self.vae = vae
        self.unet = unet
        self.pe = pe
        self.audio_processor = audio_processor
        self.whisper = whisper
        self.timesteps = timesteps

class InferenceService:
    def __init__(self, models: dict, settings):
        """
        Args:
            models: Dict containing loaded models from ModelLoader
            settings: Configuration object
        """
        self.models = InferenceModels(
            vae=models['vae'],
            unet=models['unet'],
            pe=models['pe'],
            audio_processor=models['audio_processor'],
            whisper=models['whisper'],
            timesteps=models['timesteps']
        )
        self.device = models['device'] if 'device' in models else torch.device('cuda')
        self.settings = settings

    def inference_stream(self, avatar, audio_path: str) -> Generator[bytes, None, None]:
        return inference_stream(
            avatar=avatar,
            audio_path=audio_path,
            models=self.models,
            fps=self.settings.fps,
            batch_size=self.settings.batch_size,
            audio_padding_left=self.settings.audio_padding_length_left,
            audio_padding_right=self.settings.audio_padding_length_right,
            device=self.device
        )

    def inference_batch(self, avatar, audio_path: str) -> str:
        # Generate output path
        output_dir = os.path.join(self.settings.result_dir, "inference")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{avatar.avatar_id}_{os.path.basename(audio_path).split('.')[0]}.mp4"
        output_path = os.path.join(output_dir, filename)

        return inference_batch(
            avatar=avatar,
            audio_path=audio_path,
            output_path=output_path,
            models=self.models,
            fps=self.settings.fps,
            batch_size=self.settings.batch_size,
            ffmpeg_path=self.settings.ffmpeg_path
        )

def inference_stream(
    avatar, # musetalk_server.core.avatar.Avatar
    audio_path: str,
    models: InferenceModels,
    fps: int = 25,
    batch_size: int = 4,
    audio_padding_left: int = 2,
    audio_padding_right: int = 2,
    device: torch.device = torch.device('cuda')
) -> Generator[bytes, None, None]:
    """
    Generates a stream of JPEG bytes for the given avatar and audio.
    """
    
    if not avatar.is_loaded:
        avatar.load_state()

    print(f"Start inference stream for audio: {audio_path}")
    start_time = time.time()

    # 1. Audio Processing
    ap = models.audio_processor
    whisper = models.whisper
    weight_dtype = models.unet.model.dtype 

    whisper_input_features, librosa_length = ap.get_audio_feature(audio_path, weight_dtype=weight_dtype)
    whisper_chunks = ap.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=audio_padding_left,
        audio_padding_length_right=audio_padding_right,
    )

    print(f"Audio processing costs {(time.time() - start_time) * 1000:.2f}ms")

    video_num = len(whisper_chunks)
    
    # Queues for producer-consumer
    # Use simple queues. Thread safety is handled by Queue class.
    # We use separate queues for each request to avoid mixing streams if we were sharing workers,
    # but here we spawn workers per request, so it's isolated.
    
    recon_queue = queue.Queue(maxsize=batch_size * 2)
    result_queue = queue.Queue(maxsize=batch_size * 4)
    SENTINEL = object()
    
    # We need to ensure models are used safely if multiple requests hit at once.
    # PyTorch inference on same model from multiple threads can be tricky or blocking.
    # The models are shared (passed in). 
    # To handle multiple sources correctly without race conditions on model internal state (if any),
    # we might need a lock. However, PyTorch models are generally thread-safe for inference 
    # if no internal state is mutated. 
    # Whisper and Unet/VAE should be stateless during inference.
    # BUT, to be safe and avoid OOM or weirdness, we can use a model lock or semaphore if needed.
    # For now, we assume statelessness is fine, but we'll add a check.
    
    # Actually, datagen uses `models.pe` and `models.unet`.
    # Let's ensure no side effects. `pe` is likely a functional module.
    
    inference_lock = threading.Lock() # Optional: if we want to serialize GPU access per request

    def prediction_worker():
        # Ideally, we lock mainly the GPU heavy parts if we want to prevent OOM from too many concurrent reqs
        # or rely on CUDA scheduling.
        
        gen = datagen(whisper_chunks, avatar.input_latent_list_cycle, batch_size)
        
        try:
            for i, (whisper_batch, latent_batch) in enumerate(gen):
                # We can use a lock here if needed for strict serialization
                # with inference_lock: 
                
                audio_feature_batch = models.pe(whisper_batch.to(device))
                latent_batch = latent_batch.to(device=device, dtype=weight_dtype)

                pred_latents = models.unet.model(
                    latent_batch,
                    models.timesteps,
                    encoder_hidden_states=audio_feature_batch
                ).sample
                
                pred_latents = pred_latents.to(device=device, dtype=models.vae.vae.dtype)
                recon = models.vae.decode_latents(pred_latents)
                
                for res_frame in recon:
                    recon_queue.put(res_frame)
                    
            recon_queue.put(SENTINEL)
        except Exception as e:
            print(f"Prediction worker error: {e}")
            recon_queue.put(SENTINEL) # Ensure consumer doesn't hang

    def blending_worker():
        idx = 0
        while True:
            res_frame = recon_queue.get()
            if res_frame is SENTINEL:
                result_queue.put(SENTINEL)
                break
            
            if idx >= video_num:
                continue

            # Blending logic
            # Use local copies of avatar lists to avoid any weird shared state issues (though lists are read-only here)
            cycle_len = len(avatar.coord_list_cycle)
            bbox = avatar.coord_list_cycle[idx % cycle_len]
            ori_frame = copy.deepcopy(avatar.frame_list_cycle[idx % cycle_len])
            x1, y1, x2, y2 = bbox
            
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                mask = avatar.mask_list_cycle[idx % cycle_len]
                mask_crop_box = avatar.mask_coords_list_cycle[idx % cycle_len]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                
                # Encode to JPEG
                ret, buffer = cv2.imencode('.jpg', combine_frame)
                if ret:
                    result_queue.put(buffer.tobytes())
            except Exception as e:
                print(f"Error blending frame {idx}: {e}")
            
            idx += 1

    # Start threads
    pred_thread = threading.Thread(target=prediction_worker, daemon=True)
    blend_thread = threading.Thread(target=blending_worker, daemon=True)
    
    pred_thread.start()
    blend_thread.start()
    
    # Yield results
    while True:
        data = result_queue.get()
        if data is SENTINEL:
            break
        yield data
        
    pred_thread.join()
    blend_thread.join()


def inference_batch(
    avatar, # musetalk_server.core.avatar.Avatar
    audio_path: str,
    output_path: str,
    models: InferenceModels,
    fps: int = 25,
    batch_size: int = 8,
    ffmpeg_path: str = "ffmpeg"
) -> str:
    """
    Generates a full video file for the given avatar and audio.
    Returns the path to the output video.
    """
    # Use the stream to get frames, save them, then ffmpeg
    
    temp_dir = os.path.join(os.path.dirname(output_path), f"tmp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        frame_gen = inference_stream(
            avatar, audio_path, models, fps, batch_size
        )
        
        idx = 0
        for jpeg_bytes in frame_gen:
            with open(os.path.join(temp_dir, f"{idx:08d}.jpg"), "wb") as f:
                f.write(jpeg_bytes)
            idx += 1
            
        # Compile with ffmpeg
        temp_mp4 = os.path.join(temp_dir, "temp.mp4")
        cmd_img2video = [
            ffmpeg_path, "-y", "-v", "warning", "-r", str(fps), 
            "-f", "image2", "-i", f"{temp_dir}/%08d.jpg", 
            "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18", 
            temp_mp4
        ]
        subprocess.run(cmd_img2video, check=True)
        
        # Combine Audio
        cmd_combine = [
            ffmpeg_path, "-y", "-v", "warning", 
            "-i", audio_path, "-i", temp_mp4, 
            output_path
        ]
        subprocess.run(cmd_combine, check=True)
        
        return output_path
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
