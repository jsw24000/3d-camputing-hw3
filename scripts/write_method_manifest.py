"""Write method architecture and experiment manifest files.

The assignment asks for multiple 3D representations and train/test code. Most
heavy training code is delegated to mature external projects, so this manifest
records exactly what each adapter is expected to train, what representation it
produces, and how it is evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project_utils import DEFAULT_CONFIG, cfg_get, load_config, resolve_path, write_json


def rel(path: Any) -> str:
    return "" if path is None else str(path)


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": cfg_get(config, "project.name", "assignment_3d_scene_reconstruction"),
        "shared_data": {
            "raw_video": cfg_get(config, "data.video"),
            "all_images": cfg_get(config, "paths.images"),
            "train_images": cfg_get(config, "paths.images_train"),
            "test_images": cfg_get(config, "paths.images_test"),
            "splits": cfg_get(config, "paths.splits"),
            "pose_source": "COLMAP/SfM estimate; no factory-calibrated Huawei intrinsics are assumed.",
        },
        "methods": {
            "colmap_sparse_geometry": {
                "role": "Traditional geometry baseline and shared pose source.",
                "representation": "Sparse point cloud plus calibrated pinhole/radial camera model.",
                "learned_or_estimated_parameters": [
                    "camera poses",
                    "single shared SIMPLE_RADIAL intrinsics",
                    "3D sparse point coordinates",
                    "track associations and reprojection errors",
                ],
                "architecture_or_optimizer": "Feature extraction + exhaustive matching + incremental SfM + bundle adjustment.",
                "training_objective": "Minimize multi-view reprojection error over camera poses, intrinsics, and 3D points.",
                "inputs": [
                    cfg_get(config, "paths.images"),
                    cfg_get(config, "paths.images_train"),
                ],
                "outputs": [
                    cfg_get(config, "methods.colmap.point_cloud_ply"),
                    "results/colmap/train_sparse_point_cloud.ply",
                    "results/colmap/train_geometry_summary.json",
                    "results/colmap/train_geometry_preview.png",
                ],
                "adapter_script": "scripts/run_colmap.py",
                "status": "completed for current video",
            },
            "nerfstudio_nerfacto": {
                "role": "Neural radiance-field view-synthesis method.",
                "representation": "Continuous neural field queried by 3D position and view direction; renders RGB/density along camera rays.",
                "learned_or_estimated_parameters": [
                    "Nerfacto field/network weights",
                    "proposal sampling components",
                    "appearance/radiance parameters",
                ],
                "architecture_or_optimizer": "Nerfstudio Nerfacto implementation; project adapter supplies COLMAP-derived transforms and timing/eval wrappers.",
                "training_objective": "Photometric reconstruction of training views with Nerfacto losses and ray sampling.",
                "inputs": [
                    cfg_get(config, "methods.nerfstudio.colmap_transforms"),
                    cfg_get(config, "paths.images_train"),
                ],
                "holdout": [
                    cfg_get(config, "methods.nerfstudio.colmap_transforms_test"),
                    cfg_get(config, "paths.images_test"),
                ],
                "outputs": [
                    cfg_get(config, "paths.result_nerfstudio"),
                    cfg_get(config, "methods.nerfstudio.eval_json"),
                    cfg_get(config, "methods.nerfstudio.render_output"),
                ],
                "adapter_script": "scripts/run_nerfstudio.py",
                "train_command": "conda run -n scene-nerf python scripts/run_experiment_suite.py --method nerfstudio --stage train --env-mode fallback",
                "eval_command": "conda run -n scene-nerf python scripts/run_experiment_suite.py --method nerfstudio --stage eval --env-mode fallback",
                "status": "data ready; external nerfstudio dependency not installed yet",
            },
            "official_3dgs": {
                "role": "Explicit radiance primitive representation.",
                "representation": "Set of anisotropic 3D Gaussian primitives with position, covariance/scale/rotation, opacity, and spherical-harmonic color coefficients.",
                "learned_or_estimated_parameters": [
                    "Gaussian means",
                    "Gaussian covariance / scale / rotation",
                    "opacity",
                    "color / spherical harmonic coefficients",
                ],
                "architecture_or_optimizer": "Official GraphDeCo 3D Gaussian Splatting training loop initialized from COLMAP sparse points.",
                "training_objective": "Differentiable Gaussian splatting render loss against training images, with densification/pruning during optimization.",
                "inputs": [
                    cfg_get(config, "methods.gaussian_splatting.prepared_data"),
                    cfg_get(config, "methods.gaussian_splatting.source"),
                ],
                "holdout": [
                    cfg_get(config, "methods.gaussian_splatting.prepared_eval_data"),
                    cfg_get(config, "methods.gaussian_splatting.eval_source"),
                ],
                "outputs": [
                    cfg_get(config, "methods.gaussian_splatting.model_dir"),
                    cfg_get(config, "evaluation.methods.3dgs.pred_dir"),
                    cfg_get(config, "evaluation.methods.3dgs.gt_dir"),
                ],
                "adapter_script": "scripts/run_3dgs.py",
                "train_command": "conda run -n scene-3dgs python scripts/run_experiment_suite.py --method 3dgs --stage train --env-mode fallback",
                "render_command": "conda run -n scene-3dgs python scripts/run_experiment_suite.py --method 3dgs --stage render --env-mode fallback",
                "status": "data ready; official 3DGS repository not installed yet",
            },
            "vggt_gs": {
                "role": "Extra comparison using pretrained feed-forward geometry before GS training.",
                "representation": "VGGT-predicted cameras/point maps converted to COLMAP-style geometry, then optimized with a GS/gsplat path.",
                "learned_or_estimated_parameters": [
                    "VGGT-predicted camera parameters",
                    "VGGT point/depth maps",
                    "downstream Gaussian parameters",
                ],
                "architecture_or_optimizer": "Facebook VGGT export pipeline plus gsplat/simple trainer adapter.",
                "training_objective": "Use pretrained VGGT predictions as geometry initialization, then optimize Gaussian rendering if gsplat trainer is available.",
                "inputs": [
                    cfg_get(config, "methods.vggt_gs.images"),
                    cfg_get(config, "methods.vggt_gs.scene_dir"),
                ],
                "outputs": [
                    cfg_get(config, "methods.vggt_gs.result_dir"),
                    cfg_get(config, "evaluation.methods.vggt_gs.pred_dir"),
                ],
                "adapter_script": "scripts/run_vggt_gs.py",
                "train_command": "conda run -n scene-vggt python scripts/run_experiment_suite.py --method vggt_gs --stage train --env-mode fallback",
                "status": "entry ready; VGGT and gsplat dependencies not installed yet",
            },
        },
        "evaluation": {
            "metrics_csv": cfg_get(config, "evaluation.metrics_csv"),
            "script": "scripts/evaluate_all.py",
            "metrics": ["PSNR", "SSIM", "LPIPS if available"],
            "timing_csv": "results/timing.csv",
        },
    }


def method_section(name: str, spec: dict[str, Any]) -> list[str]:
    lines = [
        f"## {name}",
        "",
        f"- Role: {spec.get('role')}",
        f"- Representation: {spec.get('representation')}",
        f"- Architecture / optimizer: {spec.get('architecture_or_optimizer')}",
        f"- Training objective: {spec.get('training_objective')}",
        f"- Adapter script: `{spec.get('adapter_script')}`",
        f"- Status: {spec.get('status')}",
        "",
        "Inputs:",
    ]
    for item in spec.get("inputs", []):
        lines.append(f"- `{rel(item)}`")
    if spec.get("holdout"):
        lines.extend(["", "Hold-out / evaluation input:"])
        for item in spec.get("holdout", []):
            lines.append(f"- `{rel(item)}`")
    lines.extend(["", "Outputs:"])
    for item in spec.get("outputs", []):
        lines.append(f"- `{rel(item)}`")
    lines.extend(["", "Learned or estimated parameters:"])
    for item in spec.get("learned_or_estimated_parameters", []):
        lines.append(f"- {item}")
    for key in ["train_command", "render_command", "eval_command"]:
        if spec.get(key):
            lines.extend(["", f"{key.replace('_', ' ').title()}:", "", "```powershell", spec[key], "```"])
    lines.append("")
    return lines


def markdown_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        "# Method Architectures",
        "",
        "This file is generated by `scripts/write_method_manifest.py`.",
        "",
        "## Shared Data",
        "",
    ]
    for key, value in manifest["shared_data"].items():
        lines.append(f"- {key}: `{rel(value)}`")
    lines.append("")
    for name, spec in manifest["methods"].items():
        lines.extend(method_section(name, spec))
    lines.extend(
        [
            "## Evaluation",
            "",
            f"- Script: `{manifest['evaluation']['script']}`",
            f"- Metrics CSV: `{manifest['evaluation']['metrics_csv']}`",
            f"- Timing CSV: `{manifest['evaluation']['timing_csv']}`",
            f"- Metrics: {', '.join(manifest['evaluation']['metrics'])}",
            "",
            "COLMAP and VGGT outputs should be described as estimated/proxy geometry, not absolute ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-json", type=Path, default=Path("results/method_manifest.json"))
    parser.add_argument("--out-md", type=Path, default=Path("METHOD_ARCHITECTURES.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = build_manifest(config)
    out_json = resolve_path(args.out_json)
    out_md = resolve_path(args.out_md)
    if out_json is None or out_md is None:
        raise ValueError("Output paths are invalid.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(f"Would write: {out_json}")
        print(f"Would write: {out_md}")
        return 0
    write_json(out_json, manifest)
    out_md.write_text(markdown_manifest(manifest), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
