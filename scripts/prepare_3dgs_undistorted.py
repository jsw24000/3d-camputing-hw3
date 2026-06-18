"""Prepare undistorted COLMAP datasets for the official 3DGS code.

GraphDeCo's loader only accepts PINHOLE / SIMPLE_PINHOLE cameras. Our SfM
model uses SIMPLE_RADIAL, so this script runs COLMAP image_undistorter and
copies the resulting COLMAP-format dataset back into the project.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, find_tool, load_config, resolve_path, run_command


def copy_tree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def normalize_sparse_zero(dataset_dir: Path) -> None:
    """Match the directory layout expected by GraphDeCo's COLMAP loader."""
    sparse_dir = dataset_dir / "sparse"
    sparse_zero = sparse_dir / "0"
    if sparse_zero.exists():
        return
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Undistorted sparse directory not found: {sparse_dir}")
    model_files = [
        path
        for path in sparse_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".bin", ".txt"}
    ]
    if not model_files:
        raise FileNotFoundError(f"No COLMAP model files found in {sparse_dir}")
    ensure_dir(sparse_zero)
    for path in model_files:
        shutil.move(str(path), sparse_zero / path.name)


def run_undistort(
    config: dict,
    *,
    images: Path,
    sparse_model: Path,
    out: Path,
    workspace_name: str,
    dry_run: bool,
) -> None:
    colmap = find_tool(config, "colmap", "colmap")
    if colmap is None:
        raise FileNotFoundError("COLMAP is not configured and not found on PATH.")
    print(f"Prepare undistorted 3DGS data: {out}")
    print(f"Images: {images}")
    print(f"Sparse model: {sparse_model}")
    if dry_run:
        print(f"Would run COLMAP image_undistorter via ASCII workspace: {workspace_name}")
        return
    if not images.exists():
        raise FileNotFoundError(f"Image directory not found: {images}")
    if not sparse_model.exists():
        raise FileNotFoundError(f"Sparse model not found: {sparse_model}")

    temp_root = Path(tempfile.gettempdir()) / workspace_name
    if temp_root.exists():
        shutil.rmtree(temp_root)
    stage_images = temp_root / "images"
    stage_sparse = temp_root / "sparse" / "0"
    stage_out = temp_root / "undistorted"
    ensure_dir(stage_images)
    ensure_dir(stage_sparse.parent)
    shutil.copytree(images, stage_images, dirs_exist_ok=True)
    shutil.copytree(sparse_model, stage_sparse, dirs_exist_ok=True)

    run_command(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            stage_images,
            "--input_path",
            stage_sparse,
            "--output_path",
            stage_out,
            "--output_type",
            "COLMAP",
        ],
        dry_run=False,
        log_path=resolve_path(f"logs/{workspace_name}_undistort.json"),
    )

    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out.parent)
    copy_tree_clean(stage_out, out)
    normalize_sparse_zero(out)
    print(f"Undistorted 3DGS data ready: {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["full", "train", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = []
    if args.mode in {"full", "all"}:
        jobs.append(
            {
                "images": resolve_path(cfg_get(config, "paths.images")),
                "sparse_model": resolve_path(cfg_get(config, "paths.colmap")) / "sparse" / "0",
                "out": resolve_path(cfg_get(config, "methods.gaussian_splatting.full_source")),
                "workspace_name": "assignment_3d_scene_3dgs_full_undistort",
            }
        )
    if args.mode in {"train", "all"}:
        jobs.append(
            {
                "images": resolve_path(cfg_get(config, "paths.images_train")),
                "sparse_model": resolve_path(cfg_get(config, "paths.colmap_train")) / "sparse" / "0",
                "out": resolve_path(cfg_get(config, "methods.gaussian_splatting.source")),
                "workspace_name": "assignment_3d_scene_3dgs_train_undistort",
            }
        )
    for job in jobs:
        if job["images"] is None or job["sparse_model"] is None or job["out"] is None:
            raise ValueError("Undistort paths are incomplete.")
        run_undistort(config, dry_run=args.dry_run, **job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
