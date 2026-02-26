# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import argparse
import json
import os

import numpy as np
from scipy.fft import fftshift, ifft2, ifftshift
from tqdm.auto import tqdm


def coil_combine(ksp, sense_map):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert multi-coil fully-sampled k-space to 2-channel complex images"
    )
    parser.add_argument("--ksp_dir", type=str, required=True,
                        help="directory containing k-space .npy files")
    parser.add_argument("--sense_dir", type=str, required=True,
                        help="directory containing sensitivity map .npy files")
    parser.add_argument("--split_json", type=str, required=True,
                        help="path to JSON file with train/val split")
    parser.add_argument("--split", type=str, default="val",
                        help="which split to process (default: val)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="directory to save output .npy files")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split_json, "r") as f:
        split_data = json.load(f)

    filenames = split_data[args.split]

    for name in tqdm(filenames, desc="Preparing targets"):
        ksp = np.load(os.path.join(args.ksp_dir, f"{name}.npy"))      # [T, C, H, W]
        sense = np.load(os.path.join(args.sense_dir, f"{name}.npy"))   # [T, C, H, W]

        combined = coil_combine(ksp, sense)  # [T, H, W] complex

        # Normalize by max magnitude (matches the dataset normalisation)
        combined = combined / np.abs(combined).max()

        # [T, H, W] complex -> [T, 2, H, W] real-imag
        target_ri = np.stack([combined.real, combined.imag], axis=1)

        np.save(os.path.join(args.output_dir, f"{name}.npy"), target_ri)

    print(f"Saved {len(filenames)} targets to {args.output_dir}")


if __name__ == "__main__":
    main()
