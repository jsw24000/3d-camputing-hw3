"""Convert the exported COLMAP text model to a Nerfstudio transforms.json file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, load_config, resolve_path


def qvec_to_rotmat(qvec: list[float]) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def parse_camera(path: Path) -> dict[str, float | int | str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]
        if model == "SIMPLE_PINHOLE":
            fl_x = fl_y = params[0]
            cx, cy = params[1], params[2]
            k1 = k2 = p1 = p2 = 0.0
        elif model == "PINHOLE":
            fl_x, fl_y, cx, cy = params[:4]
            k1 = k2 = p1 = p2 = 0.0
        elif model == "SIMPLE_RADIAL":
            fl_x = fl_y = params[0]
            cx, cy = params[1], params[2]
            k1 = params[3]
            k2 = p1 = p2 = 0.0
        elif model == "RADIAL":
            fl_x = fl_y = params[0]
            cx, cy = params[1], params[2]
            k1, k2 = params[3], params[4]
            p1 = p2 = 0.0
        elif model == "OPENCV":
            fl_x, fl_y, cx, cy, k1, k2, p1, p2 = params[:8]
        else:
            raise ValueError(f"Unsupported COLMAP camera model for conversion: {model}")
        return {
            "camera_model": model,
            "w": width,
            "h": height,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "camera_angle_x": 2 * math.atan(width / (2 * fl_x)),
            "camera_angle_y": 2 * math.atan(height / (2 * fl_y)),
        }
    raise ValueError(f"No camera line found in {path}")


def parse_images(path: Path) -> list[dict[str, object]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    frames: list[dict[str, object]] = []
    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        qvec = [float(x) for x in parts[1:5]]
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        image_name = parts[9]

        # COLMAP 保存的是 world-to-camera: x_cam = R * x_world + t。
        # Nerfstudio transforms 使用 camera-to-world，所以这里先取逆：R^T 和 -R^T t。
        r_wc = qvec_to_rotmat(qvec)
        r_cw = r_wc.T
        t_cw = -r_cw @ tvec
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = r_cw
        transform[:3, 3] = t_cw
        # COLMAP/OpenCV 相机坐标为 x右、y下、z前；Nerfstudio/OpenGL 约定为 x右、y上、z后。
        # 因此需要翻转 camera-to-world 矩阵的 y/z 两个相机轴。
        transform[:3, 1:3] *= -1.0

        frames.append(
            {
                "file_path": f"../images/{image_name}",
                "image_name": image_name,
                "transform_matrix": transform.tolist(),
            }
        )
    return frames


def read_split(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {Path(line.strip()).name for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--colmap-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    colmap_dir = args.colmap_dir or resolve_path(cfg_get(config, "methods.colmap.sparse_txt"))
    out = args.out or resolve_path(cfg_get(config, "methods.nerfstudio.colmap_transforms"))
    splits_dir = resolve_path(cfg_get(config, "paths.splits"))
    if colmap_dir is None or out is None or splits_dir is None:
        raise ValueError("COLMAP, transforms output, or splits path is not configured.")

    camera = parse_camera(colmap_dir / "cameras.txt")
    frames = parse_images(colmap_dir / "images.txt")
    test_names = read_split(splits_dir / "test.txt")
    for frame in frames:
        frame["split"] = "test" if frame["image_name"] in test_names else "train"

    data = {
        **camera,
        "orientation_override": "none",
        "applied_transform": np.eye(4, dtype=np.float64)[:3].tolist(),
        "frames": frames,
    }
    print(f"Frames converted: {len(frames)} -> {out}")
    if args.dry_run:
        print(json.dumps({k: data[k] for k in data if k != "frames"}, ensure_ascii=False, indent=2))
        return 0
    ensure_dir(out.parent)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
