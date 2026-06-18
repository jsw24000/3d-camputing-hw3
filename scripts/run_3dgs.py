"""Adapter for the official GraphDeco 3D Gaussian Splatting repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, load_config, resolve_path, run_command


def iterations_for_profile(config: dict, profile: str) -> str:
    if profile == "smoke":
        return str(cfg_get(config, "methods.gaussian_splatting.smoke_iterations", 100))
    if profile == "quick":
        return str(cfg_get(config, "methods.gaussian_splatting.quick_iterations", 7000))
    return str(cfg_get(config, "methods.gaussian_splatting.iterations", 30000))


def source_and_model(config: dict, source_mode: str) -> tuple[Path | None, Path | None]:
    if source_mode == "full":
        source = resolve_path(cfg_get(config, "methods.gaussian_splatting.full_source", "data/3dgs"))
        model_dir = resolve_path(cfg_get(config, "methods.gaussian_splatting.full_model_dir", "results/3dgs/full_model"))
    else:
        source = resolve_path(cfg_get(config, "methods.gaussian_splatting.source"))
        model_dir = resolve_path(cfg_get(config, "methods.gaussian_splatting.model_dir"))
    return source, model_dir


def python_launcher(conda_env: str | None) -> list[str]:
    if conda_env is None or conda_env.lower() == "current":
        return [sys.executable]
    return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]


def build_commands(config: dict, mode: str, source_mode: str, profile: str, conda_env: str | None) -> list[list[str]]:
    repo = resolve_path(cfg_get(config, "methods.gaussian_splatting.repo"))
    train_script = cfg_get(config, "methods.gaussian_splatting.train_script", "train.py")
    render_script = cfg_get(config, "methods.gaussian_splatting.render_script", "render.py")
    source, model_dir = source_and_model(config, source_mode)
    images = cfg_get(config, "methods.gaussian_splatting.images", "images")
    iterations = iterations_for_profile(config, profile)
    resolution = str(cfg_get(config, "methods.gaussian_splatting.resolution", 2))
    if not all([repo, source, images, model_dir]):
        raise ValueError("3DGS paths are incomplete.")
    ensure_dir(model_dir)

    commands: list[list[str]] = []
    if mode in {"train", "all"}:
        commands.append(
            python_launcher(conda_env)
            + [
                str(repo / train_script),
                "-s",
                str(source),
                "--images",
                str(images),
                "-m",
                str(model_dir),
                "--iterations",
                iterations,
                "--resolution",
                resolution,
                "--disable_viewer",
            ]
        )
    if mode in {"render", "all"}:
        commands.append(
            python_launcher(conda_env)
            + [
                str(repo / render_script),
                "-s",
                str(source),
                "--images",
                str(images),
                "-m",
                str(model_dir),
                "--iteration",
                iterations,
            ]
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["train", "render", "all"], default="all")
    parser.add_argument("--source-mode", choices=["full", "train"], default="full")
    parser.add_argument("--profile", choices=["smoke", "quick", "full"], default="full")
    parser.add_argument("--env", default=None, help="Conda env for official 3DGS, or 'current'.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    repo = resolve_path(cfg_get(config, "methods.gaussian_splatting.repo"))
    if not args.dry_run and (repo is None or not repo.exists()):
        raise FileNotFoundError(
            f"3DGS repo not found: {repo}. Clone graphdeco-inria/gaussian-splatting into external/ first."
        )
    conda_env = args.env if args.env is not None else cfg_get(config, "fallback_envs.gaussian_splatting", "scene-3dgs")
    for idx, cmd in enumerate(build_commands(config, args.mode, args.source_mode, args.profile, conda_env), start=1):
        run_command(
            cmd,
            dry_run=args.dry_run,
            log_path=resolve_path(f"logs/3dgs_{args.source_mode}_{args.profile}_{args.mode}_{idx}.json"),
            stream=not args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
