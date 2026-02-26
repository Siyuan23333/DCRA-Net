# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import torch
from torch.optim import Adam
import argparse
import os
from tqdm.auto import tqdm
import wandb

import sys

sys.path.append(os.path.abspath("."))
from src.datasets import PairedDataset
from src.metrics import loss_func
from src.dcranet import DCRANet
from src.transforms import (
    ToTime,
    ToFrequency,
    ToImage,
    ToReal,
    AddChannel,
    tensor2complex,
)
from src.utils import dump_yml
from src.utils import plot_pred_orig
from torchvision.utils import save_image


def parse_args():
    """
    Parse the arguments provided with call.

    Returns:
        Namespace() with arguments
    """
    parser = argparse.ArgumentParser(description="Script to run training")

    parser.add_argument(
        "--backbone",
        help="model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dc_mode",
        help="Data consistency mode for old unets",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--image_size", help="resize data to the given size", default=None, type=int
    )
    parser.add_argument(
        "--n_frames",
        help="resize data to the given size in time dim",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--representation_time",
        help="data representation to use: frequency or time",
        default="frequency",
        type=str,
    )
    parser.add_argument(
        "--representation_space",
        help="data representation to use: image or kspace",
        default="image",
        type=str,
    )
    parser.add_argument(
        "--out_channels", help="number of output channels", default=2, type=int
    )
    parser.add_argument(
        "--in_channels", help="number of input channels", default=2, type=int
    )
    parser.add_argument("--batch_size", help="batch size", default=1, type=int)

    parser.add_argument(
        "--start_epoch", help="start with given epoch", default=0, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs to train", default=10, type=int
    )

    parser.add_argument(
        "--ksp_dir",
        help="directory containing k-space .npy files",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--mask_dir",
        help="directory containing precomputed mask .npy files",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--sense_dir",
        help="directory containing sensitivity map .npy files",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--split_json",
        help="path to JSON file with train/val split",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--save_dir",
        help="path to save results",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--wandb_project",
        help="wandb project name",
        type=str,
        default="dcra",
    )

    parser.add_argument("--seed", help="random seed", default=42, type=int)

    parser.add_argument("--dim", help="inital emb dimension", default=64, type=int)
    parser.add_argument(
        "--dim_mults",
        help="multiplicator for unet-based backbones",
        default=(1, 2, 4),
        type=tuple,
    )

    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--loss_type",
        help="loss type",
        default="L1",
        type=str,
    )

    parser.add_argument(
        "--log_every",
        help="log training loss and grad_norm to wandb every N steps",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--verbose",
        help="verbose mode",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return vars(args)


MODELS = {"DCRA-Net": DCRANet}
DC_MODE = {
    "none": 1.0,
    "force": 0.0,
    "learn": "learn",
}


def main(config):
    assert config["backbone"] in MODELS.keys(), "Check the backbone"
    if config["dc_mode"] is not None:
        config["dc_mode"] = DC_MODE[config["dc_mode"]]
    checkpoints_dir = os.path.join(config["save_dir"], "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    dump_yml(
        config,
        os.path.join(
            config["save_dir"],
            f"config_{config['start_epoch']}_{config['n_epochs']}.yml",
        ),
    )

    samples_dir = os.path.join(config["save_dir"], "predictions")
    os.makedirs(samples_dir, exist_ok=True)
    origs_dir = os.path.join(config["save_dir"], "origs")
    os.makedirs(origs_dir, exist_ok=True)
    custom_dir = os.path.join(config["save_dir"], "custom")
    os.makedirs(custom_dir, exist_ok=True)

    inputs_dir = os.path.join(config["save_dir"], "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    assert config["representation_time"] in ["time", "frequency"]
    assert config["representation_space"] == "image"

    train_dataset = PairedDataset(
        ksp_dir=config["ksp_dir"],
        mask_dir=config["mask_dir"],
        sense_dir=config["sense_dir"],
        split_json=config["split_json"],
        split="train",
        frames=config["n_frames"],
        img_size=config["image_size"],
    )
    val_dataset = PairedDataset(
        ksp_dir=config["ksp_dir"],
        mask_dir=config["mask_dir"],
        sense_dir=config["sense_dir"],
        split_json=config["split_json"],
        split="val",
        frames=config["n_frames"],
        img_size=config["image_size"],
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["batch_size"] % (os.cpu_count() - 1),
        pin_memory=True,
        drop_last=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["batch_size"] % (os.cpu_count() - 1),
        pin_memory=True,
        drop_last=False,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = MODELS[config["backbone"]](
        dim=config["dim"],
        channels=config["in_channels"],
        out_dim=config["out_channels"],
        dim_mults=config["dim_mults"],
        representation_time=config["representation_time"],
        norm_fft=None,
        dc_mode=config["dc_mode"],
    )

    print(f"{type(model).__name__} created")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"])

    if config["start_epoch"] > 0:
        data = torch.load(
            os.path.join(
                checkpoints_dir, f"checkpoint-{config['start_epoch']-1:02d}.pt"
            ),
            map_location=device,
        )
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["opt"])

    wandb.init(
        project=config["wandb_project"],
        name=os.path.basename(config["save_dir"]),
        config=config,
        mode="online",
    )

    training_losses = {
        "mse": torch.nn.MSELoss,
        "l1": torch.nn.L1Loss,
        "smothL1": torch.nn.SmoothL1Loss,
        "huber": torch.nn.HuberLoss,
    }

    criterion = training_losses[config["loss_type"].lower()](reduction="mean")

    losses = []
    val_losses = {}
    val_losses_masked = {}

    for k in loss_func.keys():
        val_losses[k] = []
        val_losses_masked[k] = []

    for epoch in range(config["start_epoch"], config["n_epochs"]):
        model.train()
        for step, batch in tqdm(
            enumerate(train_dataloader),
            disable=not config["verbose"],
            total=len(train_dataloader),
            desc="Train Iterations",
        ):
            optimizer.zero_grad()

            # Undersampling
            kspace, mask = [item.cuda() for item in batch[0]][:]
            target = batch[1][0].cuda()

            undersampled = kspace * mask
            undersampled = AddChannel(dim=1)(undersampled)
            k_mask = mask.unsqueeze(1)  # (N, T, H, W) -> (N, 1, T, H, W)

            target = ToFrequency()(target)
            target = ToReal(batched=True)(target)

            if step == 0 and config["verbose"]:
                print(undersampled.abs().mean(), target.abs().mean())
                print(undersampled.abs().max(), target.abs().max())
                print(undersampled.size(), target.size())

            output = model(undersampled, k_mask=k_mask)
            loss = criterion(input=output, target=target)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.step()

            losses.append(loss.item())
            if (step + 1) % config["log_every"] == 0:
                wandb.log({
                    "train/loss": losses[-1],
                    "train/grad_norm": grad_norm.item(),
                })

        offset = -len(train_dataloader)
        current_mean = sum(losses[offset:]) / len(train_dataloader)

        model.eval()
        with torch.inference_mode():
            for v_iter, v_batch in tqdm(
                enumerate(val_dataloader),
                disable=not config["verbose"],
                total=len(val_dataloader),
                desc="Validation",
            ):
                # Undersampling

                v_kspace, v_mask = [item.cuda() for item in v_batch[0]][:]

                v_target = v_batch[1][0].cuda()
                v_target = ToReal(batched=True)(v_target)

                v_undersampled = v_kspace * v_mask
                v_undersampled = AddChannel(dim=1)(v_undersampled)
                v_k_mask = v_mask.unsqueeze(1)  # (N, T, H, W) -> (N, 1, T, H, W)

                v_pred = model(v_undersampled, k_mask=v_k_mask)

                v_undersampled = tensor2complex(v_undersampled)
                v_pred = tensor2complex(v_pred)
                v_target = tensor2complex(v_target)

                # Move to Image-Time for plotting
                v_undersampled = ToImage()(v_undersampled)
                v_pred = ToTime()(v_pred)

                for k in loss_func.keys():
                    val_losses[k].extend(
                        loss_func[k](v_pred, v_target, reduction="none")
                        .mean(dim=(1, 2, 3, 4))
                        .cpu()
                    )
                    mask = v_target.abs() > (0 + 1e-6)
                    val_losses_masked[k].extend(
                        loss_func[k](v_pred, v_target, mask=mask, reduction="none")
                        .mean(dim=(1, 2, 3, 4))
                        .cpu()
                    )

                if v_iter == 0:
                    n_samples = 4
                    v_pred = v_pred.cpu()
                    v_target = v_target.cpu()
                    v_undersampled = v_undersampled.cpu()
                    plot_pred_orig(
                        predictions=v_pred[:, 0].reshape(-1, *v_pred.shape[-2:])[
                            :n_samples
                        ],
                        originals=v_target[:, 0].reshape(-1, *v_pred.shape[-2:])[
                            :n_samples
                        ],
                        save=os.path.join(custom_dir, f"sample-{epoch:02d}.png"),
                        title=f"{os.path.basename(config['save_dir'])}\n{epoch:02d}",
                        loss=True,
                        normalise=False,
                    )

                    save_image(
                        v_pred[:, 0]
                        .reshape(-1, 1, *v_pred.size()[-2:])[:n_samples]
                        .abs(),
                        os.path.join(samples_dir, f"sample-{epoch:02d}.png"),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )

                    save_image(
                        v_undersampled[:, 0]
                        .reshape(-1, 1, *v_undersampled.size()[-2:])[:n_samples]
                        .abs(),
                        os.path.join(inputs_dir, f"input-{epoch:02d}.png"),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )
                    save_image(
                        v_target[:, 0]
                        .reshape(-1, 1, *v_target.size()[-2:])[:n_samples]
                        .abs(),
                        os.path.join(origs_dir, f"orig-{epoch:02d}.png"),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )

        # Log epoch-level metrics
        train_offset = -len(train_dataloader)
        train_mean = sum(losses[train_offset:]) / len(train_dataloader)
        print(f"Epoch {epoch}: train_{config['loss_type']}={train_mean:.6f}", end="")

        epoch_log = {"epoch": epoch, "train/epoch_loss": train_mean}
        for k in val_losses.keys():
            offset = -len(val_dataset)
            current_mean = sum(val_losses[k][offset:]) / len(val_dataset)
            current_mean_masked = sum(val_losses_masked[k][offset:]) / len(val_dataset)
            epoch_log[f"val/{k}"] = current_mean
            epoch_log[f"val/{k}_masked"] = current_mean_masked
            print(f"  val_{k}={current_mean:.6f}", end="")

        print()  # newline
        wandb.log(epoch_log)

        data = {
            "step": None,
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
        }
        torch.save(data, os.path.join(checkpoints_dir, f"checkpoint-{epoch:02d}.pt"))

    wandb.finish()
    print("Done")


if __name__ == "__main__":
    # config
    config = parse_args()
    print("Strating the script")
    print(config)
    main(config)
