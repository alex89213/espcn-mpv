"""Train an ESPCN model on luma patch pairs."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchDataset
from .model import ESPCN


def psnr(mse: float) -> float:
    """PSNR in dB for images in [0, 1]."""
    if mse <= 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="espcn-mpv train", description="Train an ESPCN super-resolution model."
    )
    parser.add_argument("--data-dir", required=True, help="folder of training images")
    parser.add_argument("--val-dir", default=None, help="folder of validation images")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--d1", type=int, default=4, help="conv1 output channels")
    parser.add_argument("--d2", type=int, default=4, help="conv2 output channels")
    parser.add_argument(
        "--degrade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="randomize the LR degradation (kernel, blur, JPEG, noise) so the "
        "model generalizes past bicubic-downscaled input; --no-degrade trains "
        "on plain bicubic pairs",
    )
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--patches-per-image", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-name", default="espcn")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_set = PatchDataset(
        args.data_dir,
        scale=args.scale,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        augment=True,
        degrade=args.degrade,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if args.val_dir:
        # Validation stays on clean bicubic pairs even when training is
        # degraded, so the reported PSNR is comparable across runs.
        val_set = PatchDataset(
            args.val_dir,
            scale=args.scale,
            patch_size=args.patch_size,
            patches_per_image=4,
            augment=False,
            degrade=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model = ESPCN(args.scale, args.d1, args.d2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)],
        gamma=0.1,
    )
    criterion = nn.MSELoss()

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(
        f"training on {len(train_set.paths)} images "
        f"({len(train_set)} patches/epoch), device={device}, "
        f"scale={args.scale}, d1={args.d1}, d2={args.d2}, "
        f"degrade={'on' if args.degrade else 'off'}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for lr_batch, hr_batch in progress:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(model(lr_batch), hr_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        avg_loss = running_loss / max(batches, 1)
        message = f"epoch {epoch + 1}/{args.epochs}  loss {avg_loss:.6f}"

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for lr_batch, hr_batch in val_loader:
                    lr_batch = lr_batch.to(device, non_blocking=True)
                    hr_batch = hr_batch.to(device, non_blocking=True)
                    val_loss += criterion(model(lr_batch), hr_batch).item()
                    val_batches += 1
            val_mse = val_loss / max(val_batches, 1)
            message += f"  val PSNR {psnr(val_mse):.2f} dB"

        print(message)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "scale": args.scale,
                "d1": args.d1,
                "d2": args.d2,
            },
            checkpoint_dir / f"{args.checkpoint_name}_epoch{epoch + 1}.pt",
        )

    final_path = checkpoint_dir / f"{args.checkpoint_name}_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "scale": args.scale,
            "d1": args.d1,
            "d2": args.d2,
        },
        final_path,
    )
    print(f"saved final weights to {final_path}")


if __name__ == "__main__":
    main()
