# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import argparse
import csv
import os
from glob import glob

import matplotlib.pyplot as plt
import torch
from piq import ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-sample metrics and visualize predictions"
    )
    parser.add_argument(
        "--predictions_dir",
        required=True,
        type=str,
        help="Directory containing .pt prediction files from test.py",
    )
    parser.add_argument(
        "--n_vis",
        default=5,
        type=int,
        help="Number of samples to visualize (default: 5)",
    )
    parser.add_argument(
        "--n_frames_vis",
        default=5,
        type=int,
        help="Number of frames to show per sample (default: 5)",
    )
    return parser.parse_args()


def compute_nmse(prediction, target):
    """NMSE between two complex [T, H, W] tensors."""
    error = (prediction - target).abs().pow(2).sum()
    norm = target.abs().pow(2).sum()
    return (error / norm).item()


def compute_psnr(prediction, target):
    """PSNR between two complex [T, H, W] tensors (magnitude-based)."""
    max_val = max(prediction.abs().max(), target.abs().max())
    mse = (prediction - target).abs().pow(2).mean()
    eps = torch.finfo(mse.dtype).eps
    return (20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse + eps)).item()


def compute_ssim(prediction, target):
    """Mean SSIM across frames for two complex [T, H, W] tensors."""
    pred_mag = prediction.abs().unsqueeze(1)  # [T, 1, H, W]
    tgt_mag = target.abs().unsqueeze(1)  # [T, 1, H, W]
    max_val = max(pred_mag.max(), tgt_mag.max())
    return ssim(pred_mag, tgt_mag, data_range=max_val, reduction="mean").item()


def main():
    args = parse_args()

    pt_files = sorted(glob(os.path.join(args.predictions_dir, "*.pt")))
    if len(pt_files) == 0:
        print(f"No .pt files found in {args.predictions_dir}")
        return

    print(f"Found {len(pt_files)} samples\n")

    output_dir = os.path.dirname(args.predictions_dir)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    results = []

    for idx, pt_path in enumerate(pt_files):
        sample = torch.load(pt_path, map_location="cpu")
        prediction = sample["prediction"]  # [T, H, W] complex
        target = sample["target"]  # [T, H, W] complex

        nmse = compute_nmse(prediction, target)
        psnr = compute_psnr(prediction, target)
        ssim_val = compute_ssim(prediction, target)

        name = os.path.basename(pt_path)
        results.append(
            {"file": name, "NMSE": nmse, "PSNR_dB": psnr, "SSIM": ssim_val}
        )
        print(f"[{name}]  NMSE: {nmse:.6f}  PSNR: {psnr:.2f} dB  SSIM: {ssim_val:.4f}")

        # Visualization for first N samples
        if idx < args.n_vis:
            input_img = sample["input"]  # [T, H, W] complex
            n_frames = prediction.shape[0]
            n_show = min(args.n_frames_vis, n_frames)
            frame_indices = torch.linspace(0, n_frames - 1, n_show).long()

            fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
            if n_show == 1:
                axes = axes[:, None]

            error_map = (prediction - target).abs()
            vmax = target.abs().max().item()
            emax = error_map.max().item()

            for col, fi in enumerate(frame_indices):
                axes[0, col].imshow(
                    target[fi].abs().numpy(), cmap="gray", vmin=0, vmax=vmax
                )
                axes[0, col].set_title(f"Target f={fi.item()}", fontsize=9)
                axes[0, col].axis("off")

                axes[1, col].imshow(
                    prediction[fi].abs().numpy(), cmap="gray", vmin=0, vmax=vmax
                )
                axes[1, col].set_title(f"Prediction f={fi.item()}", fontsize=9)
                axes[1, col].axis("off")

                im = axes[2, col].imshow(
                    error_map[fi].numpy(), cmap="hot", vmin=0, vmax=emax
                )
                axes[2, col].set_title(f"|Error| f={fi.item()}", fontsize=9)
                axes[2, col].axis("off")

            fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.02, pad=0.02)
            fig.suptitle(
                f"{name}  —  PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f} | NMSE: {nmse:.6f}",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(vis_dir, f"{os.path.splitext(name)[0]}.png"), dpi=150
            )
            plt.close(fig)

    # Summary statistics
    nmse_vals = [r["NMSE"] for r in results]
    psnr_vals = [r["PSNR_dB"] for r in results]
    ssim_vals = [r["SSIM"] for r in results]

    print("\n--- Summary ---")
    print(
        f"NMSE   mean: {sum(nmse_vals)/len(nmse_vals):.6f}  "
        f"std: {torch.tensor(nmse_vals).std().item():.6f}"
    )
    print(
        f"PSNR   mean: {sum(psnr_vals)/len(psnr_vals):.2f} dB  "
        f"std: {torch.tensor(psnr_vals).std().item():.2f} dB"
    )
    print(
        f"SSIM   mean: {sum(ssim_vals)/len(ssim_vals):.4f}  "
        f"std: {torch.tensor(ssim_vals).std().item():.4f}"
    )

    # Save CSV
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "NMSE", "PSNR_dB", "SSIM"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nMetrics saved to {csv_path}")
    print(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
