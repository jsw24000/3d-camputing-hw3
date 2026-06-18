"""Evaluate rendered images against hold-out ground truth views."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from project_utils import DEFAULT_CONFIG, ROOT, append_csv, cfg_get, ensure_dir, list_images, load_config, resolve_path

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "logs" / "torch"))


FIELDS = ["method", "image", "psnr", "ssim", "lpips", "status", "note"]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def maybe_lpips(pred: np.ndarray, gt: np.ndarray) -> float | None:
    try:
        import torch  # type: ignore
        import lpips  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = lpips.LPIPS(net="alex").to(device)
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            return float(loss_fn(pred_t, gt_t).item())
    except Exception:
        pass
    try:
        import torch  # type: ignore
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(loss_fn(pred_t, gt_t).item())
    except Exception:
        return None


def match_pairs(pred_dir: Path, gt_dir: Path, extensions: list[str]) -> list[tuple[Path, Path]]:
    preds = list_images(pred_dir, extensions)
    gt_by_name = {p.name: p for p in list_images(gt_dir, extensions)}
    gt_by_stem = {p.stem: p for p in gt_by_name.values()}
    pairs: list[tuple[Path, Path]] = []
    for pred in preds:
        gt = gt_by_name.get(pred.name) or gt_by_stem.get(pred.stem)
        if gt is not None:
            pairs.append((pred, gt))
    return pairs


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def evaluate(method: str, pred_dir: Path, gt_dir: Path, out_csv: Path, extensions: list[str]) -> int:
    pairs = match_pairs(pred_dir, gt_dir, extensions)
    ensure_dir(out_csv.parent)
    psnrs: list[float] = []
    ssims: list[float] = []
    lpips_values: list[float] = []
    for pred_path, gt_path in pairs:
        note = ""
        try:
            pred = load_rgb(pred_path)
            gt = load_rgb(gt_path)
            if pred.shape != gt.shape:
                note = f"shape mismatch pred={pred.shape} gt={gt.shape}"
                row = {
                    "method": method,
                    "image": pred_path.name,
                    "status": "skipped",
                    "note": note,
                    "lpips": "unavailable",
                }
                append_csv(out_csv, row, FIELDS)
                continue
            psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
            ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))
            lp = maybe_lpips(pred, gt)
            psnrs.append(psnr)
            ssims.append(ssim)
            if lp is not None:
                lpips_values.append(lp)
            append_csv(
                out_csv,
                {
                    "method": method,
                    "image": pred_path.name,
                    "psnr": f"{psnr:.6f}",
                    "ssim": f"{ssim:.6f}",
                    "lpips": f"{lp:.6f}" if lp is not None else "unavailable",
                    "status": "ok",
                    "note": note,
                },
                FIELDS,
            )
        except Exception as exc:
            append_csv(
                out_csv,
                {
                    "method": method,
                    "image": pred_path.name,
                    "lpips": "unavailable",
                    "status": "error",
                    "note": repr(exc),
                },
                FIELDS,
            )
    append_csv(
        out_csv,
        {
            "method": method,
            "image": "__mean__",
            "psnr": f"{mean(psnrs):.6f}" if psnrs else "",
            "ssim": f"{mean(ssims):.6f}" if ssims else "",
            "lpips": f"{mean(lpips_values):.6f}" if lpips_values else "unavailable",
            "status": "ok" if pairs else "no_pairs",
            "note": f"pairs={len(pairs)}",
        },
        FIELDS,
    )
    print(f"Evaluated {len(pairs)} matched image pairs. Wrote {out_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    out = args.out or resolve_path("results/metrics.csv")
    assert out is not None
    pairs = match_pairs(args.pred_dir, args.gt_dir, extensions)
    print(f"Matched {len(pairs)} image pairs.")
    if args.dry_run:
        print(f"Would write: {out}")
        return 0
    return evaluate(args.method, args.pred_dir, args.gt_dir, out, extensions)


if __name__ == "__main__":
    raise SystemExit(main())
