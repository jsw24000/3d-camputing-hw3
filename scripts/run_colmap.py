"""Run COLMAP SfM and export a sparse point cloud."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from project_utils import (
    DEFAULT_CONFIG,
    cfg_get,
    ensure_dir,
    find_tool,
    load_config,
    list_images,
    resolve_path,
    run_command,
)

def prepare_ascii_workspace(config: dict, dry_run: bool, images_override: Path | None, workspace_override: str | None) -> dict[str, Path]:
    images = images_override or resolve_path(cfg_get(config, "paths.images"))
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    workspace_name = workspace_override or cfg_get(config, "methods.colmap.ascii_workspace", "assignment_3d_scene_colmap")
    stage = Path(tempfile.gettempdir()) / workspace_name
    stage_images = stage / "images"
    stage_sparse = stage / "sparse"
    stage_sparse_txt = stage / "sparse_txt"
    stage_ply = stage / "sparse_point_cloud.ply"
    stage_database = stage / "database.db"
    if images is None:
        raise ValueError("Image path is not configured.")
    source_images = list_images(images, extensions)
    print(f"COLMAP ASCII workspace: {stage}")
    print(f"Stage {len(source_images)} images from {images} to {stage_images}")
    if not dry_run:
        if stage.exists():
            shutil.rmtree(stage)
        ensure_dir(stage_images)
        for image in source_images:
            shutil.copy2(image, stage_images / image.name)
    return {
        "stage": stage,
        "images": stage_images,
        "database": stage_database,
        "sparse": stage_sparse,
        "sparse_txt": stage_sparse_txt,
        "ply": stage_ply,
    }


def copy_stage_back(config: dict, stage_paths: dict[str, Path]) -> None:
    database = resolve_path(cfg_get(config, "methods.colmap.database"))
    sparse = resolve_path(cfg_get(config, "methods.colmap.sparse"))
    sparse_txt = resolve_path(cfg_get(config, "methods.colmap.sparse_txt"))
    ply = resolve_path(cfg_get(config, "methods.colmap.point_cloud_ply"))
    if not all([database, sparse, sparse_txt, ply]):
        raise ValueError("COLMAP copy-back paths are incomplete.")
    ensure_dir(database.parent)
    ensure_dir(sparse)
    ensure_dir(sparse_txt)
    ensure_dir(ply.parent)
    if stage_paths["database"].exists():
        shutil.copy2(stage_paths["database"], database)
    if stage_paths["sparse"].exists():
        shutil.copytree(stage_paths["sparse"], sparse, dirs_exist_ok=True)
    if stage_paths["sparse_txt"].exists():
        shutil.copytree(stage_paths["sparse_txt"], sparse_txt, dirs_exist_ok=True)
    if stage_paths["ply"].exists():
        shutil.copy2(stage_paths["ply"], ply)


def build_sfm_commands(
    config: dict,
    images: Path,
    database: Path,
    sparse: Path,
) -> list[list[str]]:
    colmap = find_tool(config, "colmap", "colmap")
    if colmap is None:
        raise FileNotFoundError("COLMAP is not configured and not found on PATH.")

    ensure_dir(database.parent)
    ensure_dir(sparse)

    single_camera = "1" if cfg_get(config, "methods.colmap.single_camera", True) else "0"
    camera_model = cfg_get(config, "methods.colmap.camera_model", "SIMPLE_RADIAL")
    camera_params = cfg_get(config, "methods.colmap.camera_params")
    matcher = cfg_get(config, "methods.colmap.matcher", "exhaustive")
    matcher_cmd = "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher"

    feature_cmd = [
            str(colmap),
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            single_camera,
            "--ImageReader.camera_model",
            str(camera_model),
        ]
    if camera_params:
        feature_cmd.extend(["--ImageReader.camera_params", str(camera_params)])

    return [
        feature_cmd,
        [str(colmap), matcher_cmd, "--database_path", str(database)],
        [
            str(colmap),
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ],
    ]


def best_sparse_model(sparse: Path) -> Path:
    candidates = [p for p in sparse.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No sparse COLMAP models found in {sparse}")

    def score(path: Path) -> tuple[int, int]:
        points = path / "points3D.bin"
        images = path / "images.bin"
        return (
            points.stat().st_size if points.exists() else 0,
            images.stat().st_size if images.exists() else 0,
        )

    return max(candidates, key=score)


def build_converter_commands(
    config: dict,
    input_model: Path,
    sparse_txt: Path,
    ply: Path,
) -> list[list[str]]:
    colmap = find_tool(config, "colmap", "colmap")
    if colmap is None:
        raise FileNotFoundError("COLMAP is not configured and not found on PATH.")
    ensure_dir(sparse_txt)
    ensure_dir(ply.parent)
    return [
        [
            str(colmap),
            "model_converter",
            "--input_path",
            str(input_model),
            "--output_path",
            str(sparse_txt),
            "--output_type",
            "TXT",
        ],
        [
            str(colmap),
            "model_converter",
            "--input_path",
            str(input_model),
            "--output_path",
            str(ply),
            "--output_type",
            "PLY",
        ],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--sparse", type=Path, default=None)
    parser.add_argument("--sparse-txt", type=Path, default=None)
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--ascii-workspace", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    use_ascii = bool(cfg_get(config, "methods.colmap.use_ascii_workspace", True))
    images_override = resolve_path(args.images_dir) if args.images_dir else None
    if use_ascii:
        stage_paths = prepare_ascii_workspace(config, args.dry_run, images_override, args.ascii_workspace)
        command_paths = stage_paths
    else:
        command_paths = {
            "images": images_override or resolve_path(cfg_get(config, "paths.images")),
            "database": resolve_path(args.database) if args.database else resolve_path(cfg_get(config, "methods.colmap.database")),
            "sparse": resolve_path(args.sparse) if args.sparse else resolve_path(cfg_get(config, "methods.colmap.sparse")),
            "sparse_txt": resolve_path(args.sparse_txt) if args.sparse_txt else resolve_path(cfg_get(config, "methods.colmap.sparse_txt")),
            "ply": resolve_path(args.ply) if args.ply else resolve_path(cfg_get(config, "methods.colmap.point_cloud_ply")),
        }
        stage_paths = {}
    copy_back_paths = {
        "database": resolve_path(args.database) if args.database else resolve_path(cfg_get(config, "methods.colmap.database")),
        "sparse": resolve_path(args.sparse) if args.sparse else resolve_path(cfg_get(config, "methods.colmap.sparse")),
        "sparse_txt": resolve_path(args.sparse_txt) if args.sparse_txt else resolve_path(cfg_get(config, "methods.colmap.sparse_txt")),
        "ply": resolve_path(args.ply) if args.ply else resolve_path(cfg_get(config, "methods.colmap.point_cloud_ply")),
    }
    if not all(command_paths.values()):
        raise ValueError("COLMAP paths are incomplete in config/project.yaml.")
    sfm_commands = build_sfm_commands(
        config,
        command_paths["images"],
        command_paths["database"],
        command_paths["sparse"],
    )
    for idx, cmd in enumerate(sfm_commands, start=1):
        run_command(
            cmd,
            dry_run=args.dry_run,
            log_path=resolve_path(f"logs/colmap_step_{idx}.json"),
        )
    input_model = command_paths["sparse"] / "0" if args.dry_run else best_sparse_model(command_paths["sparse"])
    print(f"COLMAP model selected for export: {input_model}")
    converter_commands = build_converter_commands(
        config,
        input_model,
        command_paths["sparse_txt"],
        command_paths["ply"],
    )
    for offset, cmd in enumerate(converter_commands, start=len(sfm_commands) + 1):
        run_command(
            cmd,
            dry_run=args.dry_run,
            log_path=resolve_path(f"logs/colmap_step_{offset}.json"),
        )
    if use_ascii and not args.dry_run:
        original = {
            "methods": {
                "colmap": {
                    "database": str(copy_back_paths["database"]),
                    "sparse": str(copy_back_paths["sparse"]),
                    "sparse_txt": str(copy_back_paths["sparse_txt"]),
                    "point_cloud_ply": str(copy_back_paths["ply"]),
                }
            }
        }
        copy_stage_back(original, stage_paths)
        print("Copied COLMAP outputs back to project paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
