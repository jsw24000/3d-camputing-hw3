"""Summarize exported COLMAP text model statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, load_config, resolve_path, write_json


def data_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def parse_camera(line: str) -> dict[str, object]:
    parts = line.split()
    if len(parts) < 5:
        return {}
    return {
        "camera_id": parts[0],
        "model": parts[1],
        "width": int(parts[2]),
        "height": int(parts[3]),
        "params": [float(x) for x in parts[4:]],
    }


def summarize(model_dir: Path) -> dict[str, object]:
    cameras = data_lines(model_dir / "cameras.txt")
    images = data_lines(model_dir / "images.txt")
    points = data_lines(model_dir / "points3D.txt")
    return {
        "model_dir": str(model_dir),
        "camera": parse_camera(cameras[0]) if cameras else {},
        "registered_images": len(images) // 2,
        "points3D": len(points),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir = args.model_dir or resolve_path(cfg_get(config, "methods.colmap.sparse_txt"))
    out = args.out or resolve_path("results/colmap/summary.json")
    if model_dir is None or out is None:
        raise ValueError("COLMAP model dir or output path is not configured.")
    summary = summarize(model_dir)
    write_json(out, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
