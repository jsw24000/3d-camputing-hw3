"""Check readiness for the four reconstruction methods."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from evaluate_render import match_pairs
from project_utils import DEFAULT_CONFIG, cfg_get, list_images, load_config, resolve_path, write_json


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_available(name: str, conda_env: str | None = None) -> bool:
    if conda_env is None or conda_env.lower() == "current":
        return shutil.which(name) is not None
    try:
        completed = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                conda_env,
                "python",
                "-c",
                f"import shutil; raise SystemExit(0 if shutil.which({name!r}) else 1)",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        return completed.returncode == 0
    except Exception:
        return False


def module_available_in_env(name: str, conda_env: str | None = None) -> bool:
    if conda_env is None or conda_env.lower() == "current":
        return module_available(name)
    try:
        completed = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                conda_env,
                "python",
                "-c",
                f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({name!r}) else 1)",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        return completed.returncode == 0
    except Exception:
        return False


def file_exists(path: Path | None) -> bool:
    return bool(path and path.exists())


def dir_exists(path: Path | None) -> bool:
    return bool(path and path.exists() and path.is_dir())


def count_images(path: Path | None, extensions: list[str]) -> int:
    if path is None:
        return 0
    return len(list_images(path, extensions))


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def readiness(config: dict[str, Any]) -> dict[str, Any]:
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    raw_video = resolve_path(cfg_get(config, "data.video"))
    images = resolve_path(cfg_get(config, "paths.images"))
    images_train = resolve_path(cfg_get(config, "paths.images_train"))
    images_test = resolve_path(cfg_get(config, "paths.images_test"))
    colmap_summary = resolve_path("results/colmap/summary.json")
    colmap_train_summary = resolve_path("results/colmap/train_summary.json")
    camera_report = resolve_path("logs/video_camera_report.json")
    camera_markdown = resolve_path("results/camera_parameters.md")
    method_manifest = resolve_path("results/method_manifest.json")
    method_architectures = resolve_path("METHOD_ARCHITECTURES.md")
    colmap_train_geometry = resolve_path("results/colmap/train_geometry_summary.json")
    colmap_train_centers = resolve_path("results/colmap/train_camera_centers.csv")
    colmap_train_preview = resolve_path("results/colmap/train_geometry_preview.png")
    ns_train = resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms"))
    ns_test = resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms_test"))
    gs_train = resolve_path(cfg_get(config, "methods.gaussian_splatting.prepared_data"))
    gs_train_undistorted = resolve_path(cfg_get(config, "methods.gaussian_splatting.source"))
    gs_full = resolve_path(cfg_get(config, "methods.gaussian_splatting.full_source", "data/3dgs"))
    gs_full_model = resolve_path(cfg_get(config, "methods.gaussian_splatting.full_model_dir", "results/3dgs/full_model"))
    vggt_scene = resolve_path(cfg_get(config, "methods.vggt_gs.scene_dir"))
    gs_repo = resolve_path(cfg_get(config, "methods.gaussian_splatting.repo"))
    vggt_repo = resolve_path(cfg_get(config, "methods.vggt_gs.repo"))
    eval_methods = cfg_get(config, "evaluation.methods", {}) or {}
    nerfstudio_env = cfg_get(config, "fallback_envs.nerfstudio", "scene-nerf")

    ns_data = read_json(ns_train)
    ns_test_data = read_json(ns_test)
    colmap = read_json(colmap_summary)
    colmap_train = read_json(colmap_train_summary)

    ns_train_command = command_available("ns-train", nerfstudio_env)
    ns_eval_command = command_available("ns-eval", nerfstudio_env)
    ns_module = module_available_in_env("nerfstudio", nerfstudio_env)

    checks = {
        "data": {
            "raw_video": file_exists(raw_video),
            "all_images": count_images(images, extensions),
            "train_images": count_images(images_train, extensions),
            "test_images": count_images(images_test, extensions),
            "camera_report": file_exists(camera_report),
            "camera_markdown": file_exists(camera_markdown),
            "ready": file_exists(raw_video)
            and count_images(images_train, extensions) > 0
            and count_images(images_test, extensions) > 0,
        },
        "colmap_point_cloud": {
            "summary": file_exists(colmap_summary),
            "registered_images": colmap.get("registered_images"),
            "points3D": colmap.get("points3D"),
            "ply": file_exists(resolve_path("results/colmap/sparse_point_cloud.ply")),
            "ready": file_exists(colmap_summary)
            and file_exists(resolve_path("results/colmap/sparse_point_cloud.ply")),
        },
        "train_only_colmap": {
            "summary": file_exists(colmap_train_summary),
            "registered_images": colmap_train.get("registered_images"),
            "points3D": colmap_train.get("points3D"),
            "ply": file_exists(resolve_path("results/colmap/train_sparse_point_cloud.ply")),
            "geometry_summary": file_exists(colmap_train_geometry),
            "camera_centers_csv": file_exists(colmap_train_centers),
            "geometry_preview": file_exists(colmap_train_preview),
            "ready": file_exists(colmap_train_summary)
            and file_exists(resolve_path("results/colmap/train_sparse_point_cloud.ply")),
        },
        "nerfstudio": {
            "env": nerfstudio_env,
            "train_transforms": file_exists(ns_train),
            "test_transforms": file_exists(ns_test),
            "train_data_dir": dir_exists(ns_train.parent if ns_train else None),
            "test_data_dir": dir_exists(ns_test.parent if ns_test else None),
            "train_frames": len(ns_data.get("frames", [])),
            "test_frames": len(ns_test_data.get("frames", [])),
            "ns_train_command": ns_train_command,
            "ns_eval_command": ns_eval_command,
            "module": ns_module,
            "ready_to_train": file_exists(ns_train) and ns_train_command,
        },
        "official_3dgs": {
            "full_data_dir": dir_exists(gs_full),
            "full_images": count_images(gs_full / "images" if gs_full else None, extensions),
            "full_sparse_model": dir_exists(gs_full / "sparse" if gs_full else None)
            or dir_exists(gs_full / "sparse" / "0" if gs_full else None),
            "full_model_dir": dir_exists(gs_full_model),
            "undistorted_train_data_dir": dir_exists(gs_train_undistorted),
            "undistorted_train_images": count_images(gs_train_undistorted / "images" if gs_train_undistorted else None, extensions),
            "undistorted_train_sparse_model": dir_exists(gs_train_undistorted / "sparse" if gs_train_undistorted else None)
            or dir_exists(gs_train_undistorted / "sparse" / "0" if gs_train_undistorted else None),
            "train_data_dir": dir_exists(gs_train),
            "train_images": count_images(gs_train / "images" if gs_train else None, extensions),
            "sparse_model": dir_exists(gs_train / "sparse" / "0" if gs_train else None),
            "repo": dir_exists(gs_repo),
            "train_script": file_exists(gs_repo / cfg_get(config, "methods.gaussian_splatting.train_script", "train.py") if gs_repo else None),
            "ready_to_train": dir_exists(gs_train)
            and dir_exists(gs_train / "sparse" / "0" if gs_train else None)
            and file_exists(gs_repo / cfg_get(config, "methods.gaussian_splatting.train_script", "train.py") if gs_repo else None),
        },
        "vggt_gs": {
            "scene_dir": str(vggt_scene) if vggt_scene else None,
            "repo": dir_exists(vggt_repo),
            "demo_colmap": file_exists(vggt_repo / cfg_get(config, "methods.vggt_gs.demo_colmap_script", "demo_colmap.py") if vggt_repo else None),
            "gsplat_module": module_available("gsplat"),
            "vggt_module": module_available("vggt"),
            "ready_to_train": file_exists(vggt_repo / cfg_get(config, "methods.vggt_gs.demo_colmap_script", "demo_colmap.py") if vggt_repo else None)
            and module_available("gsplat"),
        },
        "metrics": {
            "lpips_module": module_available("lpips"),
            "psnr_ssim_available": module_available("skimage"),
            "ready_basic_metrics": module_available("skimage"),
        },
        "method_manifest": {
            "json": file_exists(method_manifest),
            "markdown": file_exists(method_architectures),
            "ready": file_exists(method_manifest) and file_exists(method_architectures),
        },
    }
    checks["evaluation_outputs"] = {}
    for name, spec in eval_methods.items():
        pred_dir = resolve_path(spec.get("pred_dir"))
        gt_dir = resolve_path(spec.get("gt_dir"))
        pairs = match_pairs(pred_dir, gt_dir, extensions) if pred_dir and gt_dir else []
        checks["evaluation_outputs"][name] = {
            "pred_dir": str(pred_dir) if pred_dir else None,
            "gt_dir": str(gt_dir) if gt_dir else None,
            "pred_exists": dir_exists(pred_dir),
            "gt_exists": dir_exists(gt_dir),
            "matched_pairs": len(pairs),
            "ready_to_evaluate": dir_exists(pred_dir) and dir_exists(gt_dir) and len(pairs) > 0,
        }
    return checks


def print_report(report: dict[str, Any]) -> None:
    for section, values in report.items():
        print(f"\n[{section}]")
        for key, value in values.items():
            print(f"{key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("results/readiness.json"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    report = readiness(config)
    print_report(report)
    if not args.no_write:
        out = resolve_path(args.out)
        assert out is not None
        write_json(out, report)
        print(f"\nReadiness report written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
