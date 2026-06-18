"""Create a reproducible train/test split for extracted frames."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from project_utils import (
    DEFAULT_CONFIG,
    cfg_get,
    ensure_dir,
    list_images,
    load_config,
    resolve_path,
)


def split_images(images: list[Path], test_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not images:
        return [], []
    count = max(1, round(len(images) * test_ratio)) if len(images) > 1 else 1
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    test_set = set(shuffled[:count])
    train = [p for p in images if p not in test_set]
    test = [p for p in images if p in test_set]
    if not train and len(test) > 1:
        train.append(test.pop())
    return train, test


def write_split(path: Path, images: list[Path], base: Path) -> None:
    ensure_dir(path.parent)
    lines = [p.relative_to(base).as_posix() for p in images]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    images_dir = args.images_dir or resolve_path(cfg_get(config, "paths.images"))
    splits_dir = args.splits_dir or resolve_path(cfg_get(config, "paths.splits"))
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    test_ratio = float(cfg_get(config, "data.test_ratio", 0.1))
    seed = int(cfg_get(config, "data.random_seed", 42))
    if images_dir is None or splits_dir is None:
        raise ValueError("Image directory or split directory is not configured.")

    images = list_images(images_dir, extensions)
    train, test = split_images(images, test_ratio, seed)
    print(f"Found {len(images)} images. Train={len(train)} Test={len(test)}")
    print(f"Train split: {splits_dir / 'train.txt'}")
    print(f"Test split: {splits_dir / 'test.txt'}")
    if args.dry_run:
        for label, subset in [("train", train[:5]), ("test", test[:5])]:
            print(f"{label} preview: {[p.name for p in subset]}")
        return 0
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    write_split(splits_dir / "train.txt", train, images_dir)
    write_split(splits_dir / "test.txt", test, images_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
