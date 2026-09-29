"""Run inference and report PSNR/SSIM on the held-out MRSHaze test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from dehazemamba.data import MRSHazeDataset
from dehazemamba.model import DehazeMamba


def ssim(image: torch.Tensor, target: torch.Tensor) -> float:
    # Gaussian-window SSIM with the standard 11x11, sigma=1.5 settings.
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    kernel_1d = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11)
    mu_x = F.conv2d(image, kernel, padding=5, groups=3)
    mu_y = F.conv2d(target, kernel, padding=5, groups=3)
    sigma_x = F.conv2d(image * image, kernel, padding=5, groups=3) - mu_x * mu_x
    sigma_y = F.conv2d(target * target, kernel, padding=5, groups=3) - mu_y * mu_y
    sigma_xy = F.conv2d(image * target, kernel, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/johnny/code/dataset/MRSHaze")
    parser.add_argument("--checkpoint", default="outputs/dehazemamba/last.pt")
    parser.add_argument("--output-dir", default="outputs/dehazemamba/evaluation")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="limit evaluation for a smoke check")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model_config = checkpoint["model_config"]
    model = DehazeMamba(**model_config).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    dataset = MRSHazeDataset(args.data_root, "test")
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.no_save_images:
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, float]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="test", dynamic_ncols=True):
            hazy = batch["hazy"].cuda(non_blocking=True)
            sar = batch["sar"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                prediction = model(hazy, sar).float().clamp_(0.0, 1.0)
            for index, name in enumerate(batch["name"]):
                pred = prediction[index : index + 1]
                gt = target[index : index + 1]
                mse = float(torch.mean((pred - gt) ** 2))
                psnr = 99.0 if mse == 0 else -10.0 * float(np.log10(mse))
                image_ssim = ssim(pred, gt)
                rows.append((name, psnr, image_ssim))
                if not args.no_save_images:
                    array = (
                        pred[0].permute(1, 2, 0).cpu().numpy() * 255.0
                    ).round().clip(0, 255).astype(np.uint8)
                    Image.fromarray(array, mode="RGB").save(output_dir / "predictions" / name)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("name", "psnr_db", "ssim"))
        writer.writerows(rows)
        writer.writerow(("mean", np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows])))
    print(
        f"test_count={len(rows)} mean_psnr={np.mean([r[1] for r in rows]):.4f} dB "
        f"mean_ssim={np.mean([r[2] for r in rows]):.6f} metrics={csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
