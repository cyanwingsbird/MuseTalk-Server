import queue
import cv2
import numpy as np
import torch
from tqdm import tqdm
import copy
import time
import os
import shutil
import pickle
import glob
import json

from musetalk.utils.utils import datagen
from musetalk.utils.blending import get_image_blending
from musetalk.utils.preprocessing import read_imgs
from scripts.realtime_inference import Avatar, osmakedirs
import scripts.realtime_inference as realtime_module

class ServerAvatar(Avatar):
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, args):
        super().__init__(avatar_id, video_path, bbox_shift, batch_size, preparation)

    def init(self):
        # Access args via module
        args = realtime_module.args

        # Logic adapted from Avatar.init
        if self.preparation:
            if os.path.exists(self.avatar_path):
                print(f"Overwriting existing avatar: {self.avatar_id}")
                shutil.rmtree(self.avatar_path)

            print(f"Creating avatar: {self.avatar_id}")
            osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
            self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                raise ValueError(f"{self.avatar_id} does not exist. Please preprocess first.")

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                 raise ValueError(f"bbox_shift mismatch: saved {avatar_info['bbox_shift']}, requested {self.avatar_info['bbox_shift']}.")
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    @torch.no_grad()
    def inference_stream(self, audio_path, fps):
        # Access patched globals
        audio_processor = realtime_module.audio_processor
        device = realtime_module.device
        weight_dtype = realtime_module.weight_dtype
        whisper = realtime_module.whisper
        unet = realtime_module.unet
        vae = realtime_module.vae
        pe = realtime_module.pe
        timesteps = realtime_module.timesteps
        args = realtime_module.args

        print(f"Start inference stream for audio: {audio_path}")
        start_time = time.time()

        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )

        print(f"Audio processing costs {(time.time() - start_time) * 1000}ms")

        video_num = len(whisper_chunks)
        self.idx = 0

        gen = datagen(whisper_chunks,
                      self.input_latent_list_cycle,
                      self.batch_size)

        for i, (whisper_batch, latent_batch) in enumerate(gen):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch,
                                    timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)

            for res_frame in recon:
                if self.idx >= video_num:
                    break

                bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
                ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
                x1, y1, x2, y2 = bbox

                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
                    mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
                    combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

                    ret, buffer = cv2.imencode('.jpg', combine_frame)
                    yield buffer.tobytes()

                except Exception as e:
                    print(f"Frame processing error: {e}")
                    pass

                self.idx += 1
