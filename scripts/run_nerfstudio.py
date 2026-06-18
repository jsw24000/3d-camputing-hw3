"""Adapter for nerfstudio processing, training, and rendering."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from project_utils import (
    DEFAULT_CONFIG,
    cfg_get,
    command_path,
    ensure_dir,
    load_config,
    resolve_path,
    run_command,
)


def command_launcher(name: str, conda_env: str | None) -> list[str]:
    if conda_env is None or conda_env.lower() == "current":
        return [name]
    return ["conda", "run", "--no-capture-output", "-n", conda_env, name]


def require_command(name: str, dry_run: bool, conda_env: str | None) -> list[str]:
    if conda_env is not None and conda_env.lower() != "current":
        return command_launcher(name, conda_env)
    found = shutil.which(name)
    if found:
        return [found]
    if dry_run:
        return [name]
    raise FileNotFoundError(f"Command not found: {name}. Install nerfstudio first.")


def iterations_for_profile(config: dict, profile: str) -> str:
    if profile == "smoke":
        return str(cfg_get(config, "methods.nerfstudio.smoke_iterations", 500))
    if profile == "quick":
        return str(cfg_get(config, "methods.nerfstudio.quick_iterations", 7000))
    return str(cfg_get(config, "methods.nerfstudio.max_num_iterations", 30000))


def latest_config(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("config.yml")) + list(output_dir.rglob("config.yaml"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_trained_config(config: dict, output_dir: Path, dry_run: bool) -> Path:
    configured = cfg_get(config, "methods.nerfstudio.trained_config")
    if configured:
        path = resolve_path(configured)
        if path is None:
            raise ValueError("Configured nerfstudio trained_config is invalid.")
        return path
    found = latest_config(output_dir)
    if found is not None:
        return found
    placeholder = output_dir / "LATEST_RUN" / "config.yml"
    if dry_run:
        return placeholder
    raise FileNotFoundError(
        "No trained Nerfstudio config found. Train first or set methods.nerfstudio.trained_config."
    )


def build_commands(config: dict, mode: str, profile: str, dry_run: bool, conda_env: str | None) -> list[list[str]]:
    images = resolve_path(cfg_get(config, "paths.images"))
    ns_data = resolve_path(cfg_get(config, "paths.nerfstudio_data"))
    ns_colmap = resolve_path(cfg_get(config, "paths.nerfstudio_colmap"))
    ns_train_transforms = resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms"))
    out = resolve_path(cfg_get(config, "paths.result_nerfstudio"))
    method = cfg_get(config, "methods.nerfstudio.method", "nerfacto")
    implementation = cfg_get(config, "methods.nerfstudio.implementation")
    vis = cfg_get(config, "methods.nerfstudio.vis", "tensorboard")
    max_iters = iterations_for_profile(config, profile)
    eval_json = resolve_path(cfg_get(config, "methods.nerfstudio.eval_json", "results/nerfstudio/eval.json"))
    render_output = resolve_path(cfg_get(config, "methods.nerfstudio.render_output", "results/nerfstudio/eval_renders"))
    if not all([images, ns_data, ns_colmap, out]):
        raise ValueError("Nerfstudio paths are incomplete.")
    ensure_dir(ns_data)
    ensure_dir(out)
    if eval_json is not None:
        ensure_dir(eval_json.parent)
    if render_output is not None:
        ensure_dir(render_output)

    commands: list[list[str]] = []
    if mode in {"process", "all"}:
        ns_process = require_command(
            cfg_get(config, "methods.nerfstudio.command_process", "ns-process-data"),
            dry_run,
            conda_env,
        )
        commands.append(
            [
                *ns_process,
                "images",
                "--data",
                command_path(images),
                "--output-dir",
                command_path(ns_data),
            ]
        )
    if mode in {"train", "all"}:
        ns_train = require_command(
            cfg_get(config, "methods.nerfstudio.command_train", "ns-train"),
            dry_run,
            conda_env,
        )
        train_data = (
            ns_train_transforms.parent
            if ns_train_transforms is not None and (ns_train_transforms.exists() or dry_run)
            else ns_data
        )
        commands.append(
            cmd := [
                *ns_train,
                method,
            ]
        )
        if vis:
            cmd.extend(["--vis", str(vis)])
        cmd.extend(
            [
                "--data",
                command_path(train_data),
                "--output-dir",
                command_path(out),
                "--max-num-iterations",
                max_iters,
            ]
        )
        if implementation:
            cmd.extend(["--pipeline.model.implementation", str(implementation)])
    if mode in {"eval", "render", "all"}:
        ns_eval = require_command(
            cfg_get(config, "methods.nerfstudio.command_eval", "ns-eval"),
            dry_run,
            conda_env,
        )
        trained_config = resolve_trained_config(config, out, dry_run)
        cmd = [
            *ns_eval,
            "--load-config",
            command_path(trained_config),
        ]
        if eval_json is not None:
            cmd.extend(["--output-path", command_path(eval_json)])
        if render_output is not None:
            cmd.extend(["--render-output-path", command_path(render_output)])
        commands.append(cmd)
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["process", "train", "eval", "render", "all"], default="all")
    parser.add_argument("--profile", choices=["smoke", "quick", "full"], default="full")
    parser.add_argument("--env", default=None, help="Conda env for Nerfstudio commands, or 'current'.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    conda_env = args.env if args.env is not None else cfg_get(config, "fallback_envs.nerfstudio", "scene-nerf")
    for idx, cmd in enumerate(build_commands(config, args.mode, args.profile, args.dry_run, conda_env), start=1):
        run_command(
            cmd,
            dry_run=args.dry_run,
            log_path=resolve_path(f"logs/nerfstudio_{args.profile}_{args.mode}_{idx}.json"),
            stream=not args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
