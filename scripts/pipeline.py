"""Small orchestration CLI for the assignment workflow.

The pipeline keeps each step explicit: preprocessing and COLMAP can run now,
while neural methods remain dry-run/status-gated until their external tools are
installed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_utils import DEFAULT_CONFIG, ROOT, cfg_get, load_config, resolve_path


def run_step(label: str, cmd: list[str], dry_run: bool) -> None:
    print(f"\n[{label}]", flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {label} ({completed.returncode})")


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *args]


def preprocessing_steps(config_path: Path) -> list[tuple[str, list[str]]]:
    c = str(config_path)
    return [
        ("inspect video camera metadata", python_cmd("inspect_video_camera.py", "--config", c)),
        ("extract frames", python_cmd("prepare_video.py", "--config", c)),
        ("split train/test", python_cmd("split_dataset.py", "--config", c)),
        ("all-view COLMAP", python_cmd("run_colmap.py", "--config", c)),
        ("all-view COLMAP summary", python_cmd("summarize_colmap.py", "--config", c)),
        ("all-view Nerfstudio transforms", python_cmd("colmap_to_nerfstudio.py", "--config", c)),
        ("physical train/test views", python_cmd("prepare_training_views.py", "--config", c)),
    ]


def train_colmap_steps(config_path: Path) -> list[tuple[str, list[str]]]:
    c = str(config_path)
    return [
        (
            "train-only COLMAP",
            python_cmd(
                "timed_run.py",
                "--config",
                c,
                "--method",
                "colmap_train",
                "--stage",
                "sfm",
                "--",
                sys.executable,
                str(ROOT / "scripts" / "run_colmap.py"),
                "--config",
                c,
                "--images-dir",
                "data/images_train",
                "--database",
                "data/colmap_train/database.db",
                "--sparse",
                "data/colmap_train/sparse",
                "--sparse-txt",
                "data/colmap_train/sparse_txt",
                "--ply",
                "results/colmap/train_sparse_point_cloud.ply",
                "--ascii-workspace",
                "assignment_3d_scene_colmap_train",
            ),
        ),
        (
            "train-only COLMAP summary",
            python_cmd(
                "summarize_colmap.py",
                "--config",
                c,
                "--model-dir",
                "data/colmap_train/sparse_txt",
                "--out",
                "results/colmap/train_summary.json",
            ),
        ),
        (
            "train-only COLMAP geometry analysis",
            python_cmd(
                "analyze_colmap_geometry.py",
                "--config",
                c,
                "--model-dir",
                "data/colmap_train/sparse_txt",
                "--out-json",
                "results/colmap/train_geometry_summary.json",
                "--out-csv",
                "results/colmap/train_camera_centers.csv",
                "--out-plot",
                "results/colmap/train_geometry_preview.png",
            ),
        ),
        ("prepare 3DGS train data", python_cmd("prepare_3dgs_data.py", "--config", c)),
        ("prepare undistorted 3DGS data", python_cmd("prepare_3dgs_undistorted.py", "--config", c)),
    ]


def dry_run_method_steps(config_path: Path) -> list[tuple[str, list[str]]]:
    c = str(config_path)
    return [
        ("write method architecture manifest", python_cmd("write_method_manifest.py", "--config", c)),
        ("nerfstudio train dry-run", python_cmd("run_nerfstudio.py", "--config", c, "--mode", "train", "--dry-run")),
        ("nerfstudio eval dry-run", python_cmd("run_nerfstudio.py", "--config", c, "--mode", "eval", "--dry-run")),
        ("3DGS dry-run", python_cmd("run_3dgs.py", "--config", c, "--dry-run")),
        ("VGGT-GS dry-run", python_cmd("run_vggt_gs.py", "--config", c, "--dry-run")),
        ("configured metrics dry-run", python_cmd("evaluate_all.py", "--config", c, "--dry-run")),
    ]


def print_status(config_path: Path) -> None:
    config = load_config(config_path)
    checks = {
        "raw_video": resolve_path(cfg_get(config, "data.video")),
        "images": resolve_path(cfg_get(config, "paths.images")),
        "images_train": resolve_path(cfg_get(config, "paths.images_train")),
        "images_test": resolve_path(cfg_get(config, "paths.images_test")),
        "colmap_summary": resolve_path("results/colmap/summary.json"),
        "colmap_train_summary": resolve_path("results/colmap/train_summary.json"),
        "nerfstudio_train_transforms": resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms")),
        "3dgs_train_data": resolve_path(cfg_get(config, "methods.gaussian_splatting.prepared_data")),
    }
    print("Pipeline status")
    for name, path in checks.items():
        if path is None:
            print(f"- {name}: not configured")
        elif path.is_dir():
            count = sum(1 for item in path.iterdir())
            print(f"- {name}: exists dir ({count} entries) -> {path}")
        else:
            print(f"- {name}: {'exists' if path.exists() else 'missing'} -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "stage",
        choices=["status", "preprocess", "train-colmap", "prepare-all", "methods-dry-run"],
        help="Which workflow segment to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stage == "status":
        print_status(args.config)
        return 0

    steps: list[tuple[str, list[str]]] = []
    if args.stage in {"preprocess", "prepare-all"}:
        steps.extend(preprocessing_steps(args.config))
    if args.stage in {"train-colmap", "prepare-all"}:
        steps.extend(train_colmap_steps(args.config))
    if args.stage in {"methods-dry-run", "prepare-all"}:
        steps.extend(dry_run_method_steps(args.config))

    for label, cmd in steps:
        run_step(label, cmd, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
