"""Create physical train/test image directories and filtered Nerfstudio transforms."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, load_config, resolve_path


def read_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_images(names: list[str], src_dir: Path, dst_dir: Path, dry_run: bool) -> None:
    print(f"Copy {len(names)} images: {src_dir} -> {dst_dir}")
    if dry_run:
        return
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    ensure_dir(dst_dir)
    for name in names:
        src = src_dir / Path(name).name
        if not src.exists():
            raise FileNotFoundError(f"Image listed in split is missing: {src}")
        shutil.copy2(src, dst_dir / src.name)


def filtered_transforms(
    all_transforms: Path,
    names: set[str],
    image_dir_name: str,
    out: Path,
    dry_run: bool,
) -> None:
    data = json.loads(all_transforms.read_text(encoding="utf-8"))
    frames = []
    for frame in data.get("frames", []):
        image_name = Path(frame.get("image_name") or frame.get("file_path", "")).name
        if image_name in names:
            copied = dict(frame)
            copied["image_name"] = image_name
            copied["file_path"] = f"../{image_dir_name}/{image_name}"
            frames.append(copied)
    out_data = dict(data)
    out_data["frames"] = frames
    print(f"Write {len(frames)} transforms: {out}")
    if dry_run:
        return
    ensure_dir(out.parent)
    out.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    images = resolve_path(cfg_get(config, "paths.images"))
    images_train = resolve_path(cfg_get(config, "paths.images_train"))
    images_test = resolve_path(cfg_get(config, "paths.images_test"))
    splits = resolve_path(cfg_get(config, "paths.splits"))
    transforms_all = resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms_all"))
    transforms_train = resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms"))
    transforms_test = resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms_test"))
    if not all([images, images_train, images_test, splits, transforms_all, transforms_train, transforms_test]):
        raise ValueError("Training view paths are incomplete in config/project.yaml.")

    train = read_split(splits / "train.txt")
    test = read_split(splits / "test.txt")
    copy_images(train, images, images_train, args.dry_run)
    copy_images(test, images, images_test, args.dry_run)
    if transforms_all.exists():
        filtered_transforms(transforms_all, {Path(x).name for x in train}, "images_train", transforms_train, args.dry_run)
        filtered_transforms(transforms_all, {Path(x).name for x in test}, "images_test", transforms_test, args.dry_run)
    else:
        print(f"Skip transform filtering because missing: {transforms_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
