"""Train DehazeMamba DM-T on MRSHaze."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dehazemamba.data import MRSHazeDataset
from dehazemamba.losses import dehaze_loss
from dehazemamba.model import DehazeMamba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/johnny/code/dataset/MRSHaze")
    parser.add_argument("--output-dir", default="outputs/dehazemamba")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="debug smoke run only")
    return parser.parse_args()


def save_checkpoint(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested reproduction run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_set = MRSHazeDataset(args.data_root, "train")
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = DehazeMamba(args.base_channels, args.state_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        print(f"Resumed from epoch {start_epoch}, step {global_step}", flush=True)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"AMP={amp_dtype} train={len(train_set)} batch={args.batch_size} "
        f"parameters={n_params:,} epochs={args.epochs}",
        flush=True,
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            hazy = batch["hazy"].cuda(non_blocking=True)
            sar = batch["sar"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                prediction = model(hazy, sar)
                loss, spatial, frequency = dehaze_loss(prediction.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            global_step += 1
            progress.set_postfix(
                loss=f"{float(loss.detach()):.5f}",
                spatial=f"{float(spatial):.5f}",
                freq=f"{float(frequency):.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if args.max_steps and global_step >= args.max_steps:
                break

        scheduler.step()
        mean_loss = running_loss / max(1, min(len(loader), batch_index + 1))
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "mean_loss": mean_loss,
            "model_config": {
                "base_channels": args.base_channels,
                "state_dim": args.state_dim,
                "depths": [2, 2, 2, 1, 1],
            },
            "train_config": vars(args),
        }
        save_checkpoint(output_dir / "last.pt", payload)
        print(
            f"epoch={epoch + 1} mean_loss={mean_loss:.6f} "
            f"elapsed_hours={(time.time() - start_time) / 3600:.2f} "
            f"checkpoint={output_dir / 'last.pt'}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"Training stopped after {global_step} optimizer steps", flush=True)


if __name__ == "__main__":
    main()
