"""Prepare a GraphDeco 3DGS-style source directory from project images/COLMAP."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, list_images, load_config, resolve_path


def copy_tree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--source-sparse", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    images = resolve_path(args.images_dir) if args.images_dir else resolve_path(cfg_get(config, "paths.images_train"))
    source_sparse = resolve_path(args.source_sparse) if args.source_sparse else resolve_path(cfg_get(config, "paths.colmap_train")) / "sparse"
    out = resolve_path(args.out) if args.out else resolve_path(cfg_get(config, "methods.gaussian_splatting.prepared_data"))
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    if images is None or source_sparse is None or out is None:
        raise ValueError("3DGS prepare paths are incomplete.")

    source_model = source_sparse / "0"
    # 当前 run_colmap.py 会把最佳模型复制回 sparse/0，官方 3DGS 也默认找 sparse/0。
    image_files = list_images(images, extensions)
    print(f"Prepare 3DGS data: {out}")
    print(f"Images: {len(image_files)} from {images}")
    print(f"Sparse model: {source_model}")
    if args.dry_run:
        return 0
    if not image_files:
        raise FileNotFoundError(f"No images found in {images}")
    if not source_model.exists():
        raise FileNotFoundError(f"COLMAP sparse model not found: {source_model}")

    ensure_dir(out)
    out_images = out / "images"
    out_sparse = out / "sparse"
    out_sparse_model = out_sparse / "0"
    if out_images.exists():
        shutil.rmtree(out_images)
    ensure_dir(out_images)
    for image in image_files:
        shutil.copy2(image, out_images / image.name)
    ensure_dir(out_sparse)
    copy_tree_clean(source_model, out_sparse_model)
    print(f"3DGS data ready: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
