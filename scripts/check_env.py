"""Check the project environment for the 3D reconstruction assignment.

This script does not install packages or mutate Conda environments. It records
what is currently available so later training/preprocessing scripts can make
explicit decisions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "project.yaml"
DEFAULT_REPORT = ROOT / "logs" / "env_report.json"


def read_project_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except Exception:
        return {}


def run_command(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "ok": False, "error": repr(exc)}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_path(name: str) -> str | None:
    return shutil.which(name)


def configured_file(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"configured": False, "exists": False, "path": None}
    path = Path(path_value)
    return {
        "configured": True,
        "exists": path.exists(),
        "path": str(path),
    }


def check_torch() -> dict[str, Any]:
    if not module_available("torch"):
        return {"available": False}
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        return {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        }
    except Exception as exc:
        return {"available": True, "error": repr(exc)}


def check_conda_env(env_name: str) -> dict[str, Any]:
    conda = command_path("conda")
    if not conda:
        return {"conda_on_path": False}
    current_prefix = os.environ.get("CONDA_PREFIX")
    current_default_env = os.environ.get("CONDA_DEFAULT_ENV")
    info = {
        "conda_on_path": True,
        "conda_path": conda,
        "requested_env": env_name,
        "current_default_env": current_default_env,
        "current_prefix": current_prefix,
        "current_python": sys.executable,
        "current_env_matches": current_default_env == env_name
        or (current_prefix is not None and current_prefix.endswith(os.sep + env_name)),
    }
    probe = run_command(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            "import sys; print(sys.executable)",
        ],
        timeout=60,
    )
    info["conda_run_probe"] = probe
    return info


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = read_project_config(args.config)
    env_cfg = config.get("env", {}) if isinstance(config, dict) else {}
    configured_colmap = env_cfg.get("colmap")
    configured_ffmpeg = env_cfg.get("ffmpeg")

    modules = [
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "skimage",
        "cv2",
        "lpips",
        "nerfstudio",
        "gsplat",
        "vggt",
        "yaml",
        "tqdm",
        "rich",
    ]

    commands = {
        "python": command_path("python"),
        "conda": command_path("conda"),
        "ffmpeg": command_path("ffmpeg"),
        "colmap": command_path("colmap"),
        "ns-train": command_path("ns-train"),
        "ns-process-data": command_path("ns-process-data"),
        "nvidia-smi": command_path("nvidia-smi"),
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "project_root": str(ROOT),
        "config_path": str(args.config),
        "env": {
            "requested_name": args.env,
            "policy": "reuse existing Conda env; do not create project .venv",
            "configured_python": env_cfg.get("python"),
            "configured_colmap": configured_colmap,
            "configured_ffmpeg": configured_ffmpeg,
            "fallback_envs": config.get("fallback_envs", {}) if isinstance(config, dict) else {},
        },
        "current_python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "conda": check_conda_env(args.env),
        "modules": {name: module_available(name) for name in modules},
        "torch": check_torch(),
        "commands": commands,
        "configured_tools": {
            "colmap": configured_file(configured_colmap),
            "ffmpeg": configured_file(configured_ffmpeg),
        },
        "probes": {},
        "notes": [],
    }

    if commands["nvidia-smi"]:
        report["probes"]["nvidia_smi"] = run_command(["nvidia-smi"], timeout=20)
    if configured_colmap and Path(configured_colmap).exists():
        report["probes"]["configured_colmap_version"] = run_command(
            [configured_colmap, "-h"], timeout=20
        )
    elif commands["colmap"]:
        report["probes"]["colmap_version"] = run_command(["colmap", "-h"], timeout=20)

    if not report["conda"].get("current_env_matches"):
        report["notes"].append(
            f"Current interpreter is not clearly inside Conda env '{args.env}'. "
            f"Use: conda activate {args.env}"
        )
    if not report["modules"].get("lpips"):
        report["notes"].append("LPIPS is optional and currently unavailable.")
    if not commands["ffmpeg"] and not configured_file(configured_ffmpeg)["exists"]:
        report["notes"].append(
            "ffmpeg is not on PATH and no valid config env.ffmpeg path is set."
        )
    if not report["modules"].get("nerfstudio"):
        report["notes"].append("nerfstudio is not installed in the current interpreter.")
    if not report["modules"].get("vggt"):
        report["notes"].append("VGGT Python package/module is not available.")
    if not report["modules"].get("gsplat"):
        report["notes"].append("gsplat/3DGS Python dependency is not available.")

    return report


def print_summary(report: dict[str, Any]) -> None:
    print("Environment check summary")
    print(f"- project: {report['project_root']}")
    print(f"- requested env: {report['env']['requested_name']}")
    print(f"- current python: {report['current_python']['executable']}")
    print(f"- conda env matches: {report['conda'].get('current_env_matches')}")
    torch = report["torch"]
    print(
        "- torch: "
        + (
            f"{torch.get('version')} cuda={torch.get('cuda_available')} "
            f"device={torch.get('device')}"
            if torch.get("available")
            else "unavailable"
        )
    )
    print(f"- colmap configured: {report['configured_tools']['colmap']}")
    print(f"- ffmpeg on PATH: {bool(report['commands'].get('ffmpeg'))}")
    print(f"- nerfstudio module: {report['modules'].get('nerfstudio')}")
    print(f"- VGGT module: {report['modules'].get('vggt')}")
    print(f"- gsplat module: {report['modules'].get('gsplat')}")
    print(f"- LPIPS module: {report['modules'].get('lpips')}")
    if report["notes"]:
        print("Notes:")
        for note in report["notes"]:
            print(f"- {note}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="torch_env", help="Primary Conda env name.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Project YAML config."
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="JSON report output path."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect availability; still writes the environment report.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print summary without writing logs/env_report.json.",
    )
    args = parser.parse_args()

    report = build_report(args)
    print_summary(report)

    if not args.no_write:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
