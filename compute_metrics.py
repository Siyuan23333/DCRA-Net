# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import argparse
import os

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim
from tqdm.auto import tqdm


def to_complex(ri):
    """Convert (T, 2, H, W) real-imag array to (T, H, W) complex."""
    return ri[:, 0] + 1j * ri[:, 1]


def nmse(pred, target):
    """Normalized MSE between complex arrays."""
    return np.linalg.norm(pred - target) ** 2 / np.linalg.norm(target) ** 2


def psnr(pred, target):
    """PSNR (dB) computed on magnitudes."""
    pred_mag = np.abs(pred)
    target_mag = np.abs(target)
    max_val = max(pred_mag.max(), target_mag.max())
    mse = np.mean(np.abs(pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim(pred, target):
    """Average SSIM over temporal frames, computed on magnitudes."""
    pred_mag = np.abs(pred)
    target_mag = np.abs(target)
    data_range = max(pred_mag.max(), target_mag.max())
    scores = []
    for t in range(pred_mag.shape[0]):
        scores.append(
            sk_ssim(target_mag[t], pred_mag[t], data_range=data_range)
        )
    return np.mean(scores)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SSIM, PSNR, and NMSE between target and prediction .npy files"
    )
    parser.add_argument("--target_dir", type=str, required=True,
                        help="directory containing target .npy files (T, 2, H, W)")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="directory containing prediction .npy files (T, 2, H, W)")
    parser.add_argument("--output", type=str, default=None,
                        help="optional path to save per-sample results as .pt")
    return parser.parse_args()


def main():
    args = parse_args()

    target_files = sorted(
        f for f in os.listdir(args.target_dir) if f.endswith(".npy")
    )
    pred_files = sorted(
        f for f in os.listdir(args.pred_dir) if f.endswith(".npy")
    )

    common = sorted(set(target_files) & set(pred_files))
    if not common:
        print("No matching .npy files found between the two directories.")
        return

    if len(common) < len(target_files):
        print(
            f"Warning: {len(target_files)} targets but only {len(common)} "
            f"have matching predictions."
        )

    all_ssim, all_psnr, all_nmse = [], [], []

    for fname in tqdm(common, desc="Computing metrics"):
        target_ri = np.load(os.path.join(args.target_dir, fname))  # (T, 2, H, W)
        pred_ri = np.load(os.path.join(args.pred_dir, fname))      # (T, 2, H, W)

        target_c = to_complex(target_ri)  # (T, H, W)
        pred_c = to_complex(pred_ri)      # (T, H, W)

        all_ssim.append(ssim(pred_c, target_c))
        all_psnr.append(psnr(pred_c, target_c))
        all_nmse.append(nmse(pred_c, target_c))

    print(f"\nResults over {len(common)} samples:")
    print(f"  SSIM:  {np.mean(all_ssim):.6f} ± {np.std(all_ssim):.6f}")
    print(f"  PSNR:  {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f} dB")
    print(f"  NMSE:  {np.mean(all_nmse):.6f} ± {np.std(all_nmse):.6f}")

    if args.output:
        torch.save(
            {
                "filenames": common,
                "ssim": all_ssim,
                "psnr": all_psnr,
                "nmse": all_nmse,
            },
            args.output,
        )
        print(f"Per-sample results saved to {args.output}")


if __name__ == "__main__":
    main()
