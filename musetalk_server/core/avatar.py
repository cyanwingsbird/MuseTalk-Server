import os
import cv2
import pickle
import torch
import glob
from typing import List, Tuple, Optional

class Avatar:
    """
    Represents a preprocessed avatar with all necessary state loaded in memory.
    """
    def __init__(self, avatar_id: str, results_dir: str = "./results", version: str = "v15"):
        self.avatar_id = avatar_id
        # Define paths consistent with the existing structure
        self.avatar_path = os.path.join(results_dir, version, "avatars", avatar_id)
        self.full_imgs_path = os.path.join(self.avatar_path, "full_imgs")
        self.coords_path = os.path.join(self.avatar_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_path, "latents.pt")
        self.mask_out_path = os.path.join(self.avatar_path, "mask")
        self.mask_coords_path = os.path.join(self.avatar_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_path, "avator_info.json")

        # State containers
        self.input_latent_list_cycle: Optional[torch.Tensor] = None
        self.coord_list_cycle: Optional[List[Tuple[int, int, int, int]]] = None
        self.frame_list_cycle: Optional[List[object]] = None # cv2 images
        self.mask_coords_list_cycle: Optional[List[Tuple[int, int, int, int]]] = None
        self.mask_list_cycle: Optional[List[object]] = None # cv2 images
        
        # Info
        self.info: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self.frame_list_cycle is not None

    def exists(self) -> bool:
        return os.path.exists(self.avatar_path) and \
               os.path.exists(self.latents_out_path) and \
               os.path.exists(self.coords_path)

    def load_state(self):
        """
        Loads the avatar state from disk into memory.
        """
        if not self.exists():
            raise FileNotFoundError(f"Avatar {self.avatar_id} data not found at {self.avatar_path}")

        # Load latents
        self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location='cpu')

        # Load coordinates
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)

        # Load frames
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = self._read_imgs(input_img_list)

        # Load mask coordinates
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)

        # Load masks
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = self._read_imgs(input_mask_list)

        print(f"Avatar {self.avatar_id} loaded successfully.")

    def _read_imgs(self, img_list: List[str]) -> List[object]:
        frames = []
        for img_path in img_list:
            frame = cv2.imread(img_path)
            if frame is None:
                 raise ValueError(f"Failed to read image: {img_path}")
            frames.append(frame)
        return frames
