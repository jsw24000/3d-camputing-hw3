"""Unified experiment runner for preparation, training, rendering, and metrics.

The preparation pipeline is intentionally separate from neural-method training,
because installing Nerfstudio / 3DGS / VGGT may require method-specific Conda
environments. This runner keeps the commands reproducible while still allowing
each method to run in its own environment.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from project_utils import DEFAULT_CONFIG, ROOT, cfg_get, load_config, quote_cmd


METHODS = ["colmap", "nerfstudio", "3dgs", "vggt_gs"]


@dataclass
class Step:
    label: str
    command: list[str]


def conda_python(env_name: str) -> list[str]:
    conda = shutil.which("conda") or "conda"
    return [conda, "run", "--no-capture-output", "-n", env_name, "python"]


def python_for(config: dict, method: str, env_mode: str) -> list[str]:
    if env_mode == "current":
        return [sys.executable]
    if env_mode == "primary":
        return conda_python(cfg_get(config, "env.name", "torch_env"))
    if env_mode == "fallback":
        fallback_key = {
            "colmap": "primary",
            "nerfstudio": "nerfstudio",
            "3dgs": "gaussian_splatting",
            "vggt_gs": "vggt",
        }[method]
        if fallback_key == "primary":
            return conda_python(cfg_get(config, "env.name", "torch_env"))
        return conda_python(cfg_get(config, f"fallback_envs.{fallback_key}"))
    if env_mode == "auto":
        if method == "colmap":
            return conda_python(cfg_get(config, "env.name", "torch_env"))
        return python_for(config, method, "fallback")
    raise ValueError(f"Unknown env mode: {env_mode}")


def script_cmd(py: list[str], script: str, *args: str) -> list[str]:
    return [*py, str(ROOT / "scripts" / script), *args]


def timed(config: dict, config_path: Path, method: str, stage: str, inner: list[str], dry_run_inner: bool) -> list[str]:
    py = python_for(config, "colmap", "primary")
    command = [
        *py,
        str(ROOT / "scripts" / "timed_run.py"),
        "--config",
        str(config_path),
        "--method",
        method,
        "--stage",
        stage,
        "--",
        *inner,
    ]
    if dry_run_inner:
        # The timed wrapper itself should not mutate timing.csv during a global
        # dry-run; the caller will skip execution after printing.
        return command
    return command


def prepare_steps(config: dict, config_path: Path) -> list[Step]:
    py = python_for(config, "colmap", "primary")
    return [
        Step(
            "prepare data, COLMAP, transforms, and method inputs",
            script_cmd(py, "pipeline.py", "--config", str(config_path), "prepare-all"),
        )
    ]


def colmap_steps(config: dict, config_path: Path, stage: str, env_mode: str) -> list[Step]:
    py = python_for(config, "colmap", env_mode)
    if stage in {"all", "prepare", "train"}:
        return [
            Step(
                "COLMAP train-only reconstruction and geometry analysis",
                script_cmd(py, "pipeline.py", "--config", str(config_path), "train-colmap"),
            )
        ]
    return []


def nerfstudio_steps(
    config: dict,
    config_path: Path,
    stage: str,
    env_mode: str,
    dry_run: bool,
    nerf_profile: str,
) -> list[Step]:
    py = python_for(config, "colmap", "primary")
    env_args = ["--env", "current"] if env_mode in {"current", "primary"} else []
    steps: list[Step] = []
    if stage in {"all", "train"}:
        inner = script_cmd(
            py,
            "run_nerfstudio.py",
            "--config",
            str(config_path),
            "--mode",
            "train",
            "--profile",
            nerf_profile,
            *env_args,
        )
        steps.append(Step("Nerfstudio train", timed(config, config_path, "nerfstudio", "train", inner, dry_run)))
    if stage in {"all", "render", "eval"}:
        mode = "render" if stage == "render" else "eval"
        inner = script_cmd(
            py,
            "run_nerfstudio.py",
            "--config",
            str(config_path),
            "--mode",
            mode,
            "--profile",
            nerf_profile,
            *env_args,
        )
        steps.append(Step("Nerfstudio native eval/render", timed(config, config_path, "nerfstudio", "eval", inner, dry_run)))
    if stage in {"all", "eval"}:
        local_py = python_for(config, "colmap", "primary")
        steps.append(Step("Nerfstudio shared metrics", script_cmd(local_py, "evaluate_all.py", "--config", str(config_path), "--method", "nerfstudio")))
    return steps


def gs_steps(
    config: dict,
    config_path: Path,
    stage: str,
    env_mode: str,
    dry_run: bool,
    gs_source_mode: str,
    gs_profile: str,
) -> list[Step]:
    py = python_for(config, "3dgs", env_mode)
    steps: list[Step] = []
    if stage in {"all", "train"}:
        inner = script_cmd(
            py,
            "run_3dgs.py",
            "--config",
            str(config_path),
            "--mode",
            "train",
            "--source-mode",
            gs_source_mode,
            "--profile",
            gs_profile,
        )
        steps.append(Step("Official 3DGS train", timed(config, config_path, "3dgs", "train", inner, dry_run)))
    if stage in {"all", "render"}:
        inner = script_cmd(
            py,
            "run_3dgs.py",
            "--config",
            str(config_path),
            "--mode",
            "render",
            "--source-mode",
            gs_source_mode,
            "--profile",
            gs_profile,
        )
        steps.append(Step("Official 3DGS render", timed(config, config_path, "3dgs", "render", inner, dry_run)))
    if stage in {"all", "eval"}:
        local_py = python_for(config, "colmap", "primary")
        steps.append(Step("Official 3DGS shared metrics", script_cmd(local_py, "evaluate_all.py", "--config", str(config_path), "--method", "3dgs")))
    return steps


def vggt_steps(config: dict, config_path: Path, stage: str, env_mode: str, dry_run: bool) -> list[Step]:
    py = python_for(config, "vggt_gs", env_mode)
    steps: list[Step] = []
    if stage in {"all", "train", "render"}:
        inner = script_cmd(py, "run_vggt_gs.py", "--config", str(config_path), "--mode", "all")
        steps.append(Step("VGGT-GS export/train", timed(config, config_path, "vggt_gs", "all", inner, dry_run)))
    if stage in {"all", "eval"}:
        local_py = python_for(config, "colmap", "primary")
        steps.append(Step("VGGT-GS shared metrics", script_cmd(local_py, "evaluate_all.py", "--config", str(config_path), "--method", "vggt_gs")))
    return steps


def collect_steps(config: dict, config_path: Path) -> list[Step]:
    py = python_for(config, "colmap", "primary")
    return [
        Step("readiness report", script_cmd(py, "check_readiness.py", "--config", str(config_path))),
        Step("collect result sizes", script_cmd(py, "collect_results.py", "--config", str(config_path))),
    ]


def expand_methods(method: str) -> list[str]:
    return METHODS if method == "all" else [method]


def steps_for(
    config: dict,
    config_path: Path,
    method: str,
    stage: str,
    env_mode: str,
    dry_run: bool,
    nerf_profile: str,
    gs_source_mode: str,
    gs_profile: str,
) -> list[Step]:
    steps: list[Step] = []
    if stage == "prepare":
        return prepare_steps(config, config_path)
    if stage == "collect":
        return collect_steps(config, config_path)

    for item in expand_methods(method):
        if item == "colmap":
            steps.extend(colmap_steps(config, config_path, stage, env_mode))
        elif item == "nerfstudio":
            steps.extend(nerfstudio_steps(config, config_path, stage, env_mode, dry_run, nerf_profile))
        elif item == "3dgs":
            steps.extend(gs_steps(config, config_path, stage, env_mode, dry_run, gs_source_mode, gs_profile))
        elif item == "vggt_gs":
            steps.extend(vggt_steps(config, config_path, stage, env_mode, dry_run))
        else:
            raise ValueError(f"Unknown method: {item}")
    if method == "all" and stage == "all":
        steps.extend(collect_steps(config, config_path))
    return steps


def run_steps(steps: Iterable[Step], dry_run: bool) -> int:
    for step in steps:
        print(f"\n[{step.label}]", flush=True)
        print(quote_cmd(step.command), flush=True)
        if dry_run:
            continue
        completed = subprocess.run(step.command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", choices=["all", *METHODS], default="all")
    parser.add_argument(
        "--stage",
        choices=["prepare", "train", "render", "eval", "collect", "all"],
        default="all",
    )
    parser.add_argument(
        "--env-mode",
        choices=["current", "primary", "fallback", "auto"],
        default="auto",
        help="current uses this Python; primary uses torch_env; auto uses fallback envs for neural methods.",
    )
    parser.add_argument("--gs-source-mode", choices=["full", "train"], default="full")
    parser.add_argument("--nerf-profile", choices=["smoke", "quick", "full"], default="full")
    parser.add_argument("--gs-profile", choices=["smoke", "quick", "full"], default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    steps = steps_for(
        config,
        config_path,
        args.method,
        args.stage,
        args.env_mode,
        args.dry_run,
        args.nerf_profile,
        args.gs_source_mode,
        args.gs_profile,
    )
    if not steps:
        print("No steps selected.")
        return 0
    return run_steps(steps, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
