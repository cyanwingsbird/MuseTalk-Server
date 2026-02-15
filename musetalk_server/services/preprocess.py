import os
import shutil
import glob
import cv2
import torch
import pickle
import json
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List

# Core imports (assuming these exist in the environment as per existing scripts)
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material

class AvatarPreprocessor:
    def __init__(self, vae, face_parser):
        self.vae = vae
        self.face_parser = face_parser

    def process_avatar(
        self,
        video_path: str,
        avatar_id: str,
        bbox_shift: int = 0,
        results_dir: str = "./results",
        force_recreation: bool = False,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        version: str = "v15"
    ):
        process_avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            vae=self.vae,
            face_parser=self.face_parser,
            bbox_shift=bbox_shift,
            results_dir=results_dir,
            force_recreation=force_recreation,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            version=version
        )

def video2imgs(vid_path: str, save_path: str, ext: str = '.png', cut_frame: int = 10000000):
    """
    Extracts frames from a video file.
    """
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
            count += 1
        else:
            break
    cap.release()

def process_avatar(
    avatar_id: str,
    video_path: str,
    vae: torch.nn.Module,
    face_parser: object,
    bbox_shift: int = 0,
    results_dir: str = "./results",
    force_recreation: bool = False,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    version: str = "v15"
):
    """
    Preprocesses an avatar from a video source.
    Generates frames, landmarks, latents, and masks.
    """
    
    # Define paths
    base_path = os.path.join(results_dir, version, "avatars", avatar_id)
    full_imgs_path = os.path.join(base_path, "full_imgs")
    coords_path = os.path.join(base_path, "coords.pkl")
    latents_out_path = os.path.join(base_path, "latents.pt")
    mask_out_path = os.path.join(base_path, "mask")
    mask_coords_path = os.path.join(base_path, "mask_coords.pkl")
    avatar_info_path = os.path.join(base_path, "avator_info.json")
    video_out_path = os.path.join(base_path, "vid_output") # Needed for structure compatibility

    avatar_info = {
        "avatar_id": avatar_id,
        "video_path": video_path,
        "bbox_shift": bbox_shift,
        "version": version
    }

    # Check if exists
    if os.path.exists(base_path):
        if not force_recreation:
            # Check consistency and required outputs
            required_paths = [coords_path, latents_out_path, mask_coords_path, avatar_info_path]
            has_required = all(os.path.exists(p) for p in required_paths)
            has_masks = os.path.isdir(mask_out_path) and len(os.listdir(mask_out_path)) > 0
            has_frames = os.path.isdir(full_imgs_path) and len(os.listdir(full_imgs_path)) > 0

            if os.path.exists(avatar_info_path):
                with open(avatar_info_path, "r") as f:
                    existing_info = json.load(f)
                if existing_info.get('bbox_shift') == bbox_shift and has_required and has_masks and has_frames:
                    print(f"Avatar {avatar_id} already exists with matching bbox_shift. Skipping.")
                    return
                else:
                    print(f"Avatar {avatar_id} exists but is incomplete or bbox_shift mismatch. Recreating...")
            else:
                print(f"Avatar {avatar_id} exists but info missing. Recreating...")

        # Clean up
        shutil.rmtree(base_path)

    print(f"Creating avatar: {avatar_id}")
    for p in [base_path, full_imgs_path, video_out_path, mask_out_path]:
        os.makedirs(p, exist_ok=True)

    with open(avatar_info_path, "w") as f:
        json.dump(avatar_info, f)

    # 1. Extract Frames
    if os.path.isfile(video_path):
        print(f"Extracting frames from {video_path}...")
        video2imgs(video_path, full_imgs_path, ext='.png')
    elif os.path.isdir(video_path):
        print(f"Copying frames from {video_path}...")
        files = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.png')])
        for filename in files:
            shutil.copyfile(os.path.join(video_path, filename), os.path.join(full_imgs_path, filename))
    else:
        raise FileNotFoundError(f"Video path not found: {video_path}")

    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
    
    # 2. Extract Landmarks & BBox
    print("Extracting landmarks...")
    # coord_list is a list of [x1, y1, x2, y2]
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    
    input_latent_list = []
    valid_coord_list = []
    valid_frame_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    # 3. Process Frames (Crop -> Resize -> VAE Latents)
    print("Processing latents...")
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue

        x1, y1, x2, y2 = bbox
        if version == "v15":
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])
        adjusted_bbox = [x1, y1, x2, y2]

        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Calculate latents
        # Assuming vae.get_latents_for_unet exists and handles device transfer if needed
        latents = vae.get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)
        valid_coord_list.append(adjusted_bbox)
        valid_frame_list.append(frame)

    if not valid_coord_list:
        raise ValueError("No valid face detections found; cannot create avatar.")

    # 4. Cycle Lists (Forward + Backward for smooth looping)
    frame_list_cycle = valid_frame_list + valid_frame_list[::-1]
    coord_list_cycle = valid_coord_list + valid_coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # Replace stored frames with the cycled list so load_state matches lengths
    for filename in os.listdir(full_imgs_path):
        os.remove(os.path.join(full_imgs_path, filename))
    for idx, frame in enumerate(frame_list_cycle):
        cv2.imwrite(os.path.join(full_imgs_path, f"{idx:08d}.png"), frame)
    
    mask_coords_list_cycle = []
    mask_list_cycle = []

    # 5. Generate Masks
    print("Generating masks...")
    for i, frame in enumerate(tqdm(frame_list_cycle)):
        x1, y1, x2, y2 = coord_list_cycle[i]
        
        mask_mode = parsing_mode if version == "v15" else "raw"
        
        # get_image_prepare_material returns (mask_array, crop_box)
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=face_parser, mode=mask_mode)
        
        cv2.imwrite(os.path.join(mask_out_path, f"{i:08d}.png"), mask)
        mask_coords_list_cycle.append(crop_box)
        mask_list_cycle.append(mask)

    # 6. Save State
    print("Saving state...")
    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list_cycle, f)

    torch.save(input_latent_list_cycle, latents_out_path)
    print(f"Avatar {avatar_id} preprocessing complete.")
