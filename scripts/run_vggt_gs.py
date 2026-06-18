"""Adapter for VGGT COLMAP export followed by gsplat/GS training."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, list_images, load_config, resolve_path, run_command


def python_launcher(conda_env: str | None) -> list[str]:
    if conda_env is None or conda_env.lower() == "current":
        return [sys.executable]
    return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]


def collect_vggt_images(config: dict, source: Path, extensions: list[str]) -> list[Path]:
    images = list_images(source, extensions)
    if images:
        return images

    fallback_dirs = [
        resolve_path(cfg_get(config, "paths.images_train")),
        resolve_path(cfg_get(config, "paths.images_test")),
    ]
    merged: dict[str, Path] = {}
    for directory in fallback_dirs:
        if directory is None:
            continue
        for image in list_images(directory, extensions):
            merged[image.name] = image
    if merged:
        print(
            "Configured VGGT source is empty; falling back to "
            "paths.images_train + paths.images_test."
        )
    return [merged[name] for name in sorted(merged)]


def sync_scene_images(config: dict, dry_run: bool) -> None:
    source = resolve_path(cfg_get(config, "methods.vggt_gs.images"))
    scene = resolve_path(cfg_get(config, "methods.vggt_gs.scene_dir"))
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    if source is None or scene is None:
        raise ValueError("VGGT image source or scene path is not configured.")
    target = scene / "images"
    images = collect_vggt_images(config, source, extensions)
    print(f"VGGT scene image sync: {source} -> {target} ({len(images)} images)")
    if dry_run:
        return
    if not images:
        raise FileNotFoundError(f"No source images found for VGGT: {source}")
    if target.exists():
        shutil.rmtree(target)
    ensure_dir(target)
    for image in images:
        shutil.copy2(image, target / image.name)


def build_commands(config: dict, mode: str, dry_run: bool, conda_env: str | None) -> list[list[str]]:
    repo = resolve_path(cfg_get(config, "methods.vggt_gs.repo"))
    scene = resolve_path(cfg_get(config, "methods.vggt_gs.scene_dir"))
    result = resolve_path(cfg_get(config, "methods.vggt_gs.result_dir"))
    demo = cfg_get(config, "methods.vggt_gs.demo_colmap_script", "demo_colmap.py")
    use_ba = bool(cfg_get(config, "methods.vggt_gs.use_ba", False))
    max_query_pts = str(cfg_get(config, "methods.vggt_gs.max_query_pts", 2048))
    query_frame_num = str(cfg_get(config, "methods.vggt_gs.query_frame_num", 5))
    if not all([repo, scene, result]):
        raise ValueError("VGGT-GS paths are incomplete.")
    if not dry_run:
        ensure_dir(scene)
        ensure_dir(result)

    commands: list[list[str]] = []
    if mode in {"vggt", "all"}:
        cmd = [
            *python_launcher(conda_env),
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
                *python_launcher(conda_env),
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
    parser.add_argument("--env", default=None, help="Conda env for VGGT/gsplat, or 'current'.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    conda_env = args.env if args.env is not None else cfg_get(config, "fallback_envs.vggt", "scene-vggt")
    if args.mode in {"vggt", "all"}:
        sync_scene_images(config, args.dry_run)
    for idx, cmd in enumerate(build_commands(config, args.mode, args.dry_run, conda_env), start=1):
        run_command(
            cmd,
            dry_run=args.dry_run,
            log_path=resolve_path(f"logs/vggt_gs_{args.mode}_{idx}.json"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
