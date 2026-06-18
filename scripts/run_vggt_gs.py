"""Adapter for VGGT COLMAP export followed by gsplat/GS training."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, list_images, load_config, resolve_path, run_command


def sync_scene_images(config: dict, dry_run: bool) -> None:
    source = resolve_path(cfg_get(config, "methods.vggt_gs.images"))
    scene = resolve_path(cfg_get(config, "methods.vggt_gs.scene_dir"))
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    if source is None or scene is None:
        raise ValueError("VGGT image source or scene path is not configured.")
    target = scene / "images"
    images = list_images(source, extensions)
    print(f"VGGT scene image sync: {source} -> {target} ({len(images)} images)")
    if dry_run:
        return
    if not images:
        raise FileNotFoundError(f"No source images found for VGGT: {source}")
    ensure_dir(target)
    for image in images:
        shutil.copy2(image, target / image.name)


def build_commands(config: dict, mode: str, dry_run: bool) -> list[list[str]]:
    repo = resolve_path(cfg_get(config, "methods.vggt_gs.repo"))
    scene = resolve_path(cfg_get(config, "methods.vggt_gs.scene_dir"))
    result = resolve_path(cfg_get(config, "methods.vggt_gs.result_dir"))
    demo = cfg_get(config, "methods.vggt_gs.demo_colmap_script", "demo_colmap.py")
    use_ba = bool(cfg_get(config, "methods.vggt_gs.use_ba", False))
    max_query_pts = str(cfg_get(config, "methods.vggt_gs.max_query_pts", 2048))
    query_frame_num = str(cfg_get(config, "methods.vggt_gs.query_frame_num", 5))
    if not all([repo, scene, result]):
        raise ValueError("VGGT-GS paths are incomplete.")
    ensure_dir(scene)
    ensure_dir(result)

    commands: list[list[str]] = []
    if mode in {"vggt", "all"}:
        cmd = [
            sys.executable,
            str(repo / demo),
            f"--scene_dir={scene}",
            f"--max_query_pts={max_query_pts}",
            f"--query_frame_num={query_frame_num}",
        ]
        if use_ba:
            cmd.append("--use_ba")
        commands.append(cmd)

    if mode in {"gsplat", "all"}:
        gsplat_trainer = shutil.which("simple_trainer.py")
        trainer = gsplat_trainer or "examples/simple_trainer.py"
        commands.append(
            [
                sys.executable,
                trainer,
                "default",
                "--data_factor",
                "1",
                "--data_dir",
                str(scene),
                "--result_dir",
                str(result),
            ]
        )
    if not dry_run and repo is not None and not repo.exists():
        raise FileNotFoundError(
            f"VGGT repo not found: {repo}. Clone facebookresearch/vggt into external/ first."
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["vggt", "gsplat", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode in {"vggt", "all"}:
        sync_scene_images(config, args.dry_run)
    for idx, cmd in enumerate(build_commands(config, args.mode, args.dry_run), start=1):
        run_command(
            cmd,
            dry_run=args.dry_run,
            log_path=resolve_path(f"logs/vggt_gs_{args.mode}_{idx}.json"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
