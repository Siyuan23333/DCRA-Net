# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.fft import fftshift, ifftshift, ifft2
from src.transforms import ToKSpace


class PairedDataset(Dataset):
    """Dataset for multi-coil MRI data with precomputed masks and sensitivity maps.

    Expects three directories of matching .npy files (matched by filename from
    a JSON split file):
      - ksp_dir:   k-space data, shape [T, C, H, W] (complex)
      - mask_dir:  sampling masks, shape [T, H, W]
      - sense_dir: coil sensitivity maps, shape [T, C, H, W] (complex)

    The dataset performs coil combination using the sensitivity maps to produce
    single-coil equivalent data, then applies spatial transforms and returns
    the format expected by the DCRA-Net training loop:
      ((kspace, mask), (target,))
    """

    def __init__(
        self,
        ksp_dir,
        mask_dir,
        sense_dir,
        split_json,
        split="train",
        frames=32,
        img_size=None,
    ):
        super().__init__()
        self.frames = frames
        self.img_size = img_size

        with open(split_json, "r") as f:
            split_data = json.load(f)

        filenames = split_data[split]
        self.ksp_files = [os.path.join(ksp_dir, f"{name}.npy") for name in filenames]
        self.mask_files = [os.path.join(mask_dir, f"{name}.npy") for name in filenames]
        self.sense_files = [os.path.join(sense_dir, f"{name}.npy") for name in filenames]

    def __len__(self):
        return len(self.ksp_files)

    @staticmethod
    def _coil_combine(ksp, sense_map):
        """Combine multi-coil k-space using sensitivity maps.

        Args:
            ksp: [T, C, H, W] complex k-space
            sense_map: [T, C, H, W] complex sensitivity maps

        Returns:
            combined: [T, H, W] complex combined image
        """
        coil_images = fftshift(
            ifft2(ifftshift(ksp, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
            axes=(-2, -1),
        )
        combined = (coil_images * np.conj(sense_map)).sum(axis=1)
        return combined

    def __getitem__(self, idx):
        # Load multi-coil data
        ksp = np.load(self.ksp_files[idx])      # [T, C, H, W] complex
        mask = np.load(self.mask_files[idx])     # [T, H, W]
        sense = np.load(self.sense_files[idx])   # [T, C, H, W] complex

        # Coil combine -> single-coil image [T, H, W] complex
        combined_image = self._coil_combine(ksp, sense)

        # Convert to tensors
        combined_image = torch.as_tensor(combined_image.copy())
        mask = torch.as_tensor(mask.copy()).float()

        # CutFrames
        combined_image = combined_image[: self.frames]
        mask = mask[: self.frames]

        # Optional Resize + CenterCrop
        if self.img_size is not None:
            img_real = torch.view_as_real(combined_image)  # [T, H, W, 2]
            img_real = img_real.permute(3, 0, 1, 2)        # [2, T, H, W]
            img_real = transforms.Resize(self.img_size, antialias=True)(img_real)
            img_real = transforms.CenterCrop(self.img_size)(img_real)
            combined_image = torch.view_as_complex(
                img_real.permute(1, 2, 3, 0).contiguous()
            )

            mask = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )(mask)
            mask = transforms.CenterCrop(self.img_size)(mask)

        # Convert combined image to k-space
        kspace = ToKSpace()(combined_image)  # [T, H', W'] complex
        target = combined_image              # [T, H', W'] complex

        # Normalize by max magnitude
        target_delta = target.abs().max()
        target = target / target_delta
        kspace = kspace / target_delta

        return (kspace, mask), (target,)
