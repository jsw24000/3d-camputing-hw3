"""Analyze COLMAP camera trajectory and sparse point geometry.

This script turns COLMAP's text model into report-friendly artifacts:
camera centers, reprojection-error statistics, bounding boxes, and a compact
preview plot.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from project_utils import DEFAULT_CONFIG, ensure_dir, resolve_path, write_json


def data_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qz * qx + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qz * qx - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def parse_images(path: Path) -> list[dict[str, Any]]:
    lines = data_lines(path)
    images: list[dict[str, Any]] = []
    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        if len(parts) < 10:
            continue
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        rot = qvec_to_rotmat(qvec)
        center = -rot.T @ tvec
        images.append(
            {
                "image_id": int(parts[0]),
                "camera_id": int(parts[8]),
                "name": parts[9],
                "qvec": qvec.tolist(),
                "tvec": tvec.tolist(),
                "center": center.tolist(),
            }
        )
    return images


def parse_points(path: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for line in data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            continue
        track_values = parts[8:]
        points.append(
            {
                "point3d_id": int(parts[0]),
                "xyz": [float(parts[1]), float(parts[2]), float(parts[3])],
                "rgb": [int(parts[4]), int(parts[5]), int(parts[6])],
                "error": float(parts[7]),
                "track_length": len(track_values) // 2,
            }
        )
    return points


def bbox(values: np.ndarray) -> dict[str, list[float]] | None:
    if values.size == 0:
        return None
    return {
        "min": values.min(axis=0).round(6).tolist(),
        "max": values.max(axis=0).round(6).tolist(),
        "extent": (values.max(axis=0) - values.min(axis=0)).round(6).tolist(),
    }


def describe(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def write_camera_csv(path: Path, images: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_id",
                "camera_id",
                "name",
                "center_x",
                "center_y",
                "center_z",
                "qvec",
                "tvec",
            ],
        )
        writer.writeheader()
        for image in sorted(images, key=lambda item: item["name"]):
            center = image["center"]
            writer.writerow(
                {
                    "image_id": image["image_id"],
                    "camera_id": image["camera_id"],
                    "name": image["name"],
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "qvec": " ".join(f"{v:.10f}" for v in image["qvec"]),
                    "tvec": " ".join(f"{v:.10f}" for v in image["tvec"]),
                }
            )


def save_preview(path: Path, camera_centers: np.ndarray, points_xyz: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    sample = points_xyz
    if len(sample) > 5000:
        sample = sample[np.linspace(0, len(sample) - 1, 5000).astype(int)]

    axes[0].scatter(sample[:, 0], sample[:, 1], s=1, c="0.35", alpha=0.45)
    axes[0].plot(camera_centers[:, 0], camera_centers[:, 1], color="#d62728", marker="o", markersize=2, linewidth=1)
    axes[0].set_title("XY view")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")

    axes[1].scatter(sample[:, 0], sample[:, 2], s=1, c="0.35", alpha=0.45)
    axes[1].plot(camera_centers[:, 0], camera_centers[:, 2], color="#d62728", marker="o", markersize=2, linewidth=1)
    axes[1].set_title("XZ view")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].axis("equal")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def analyze(model_dir: Path, out_json: Path, out_csv: Path, out_plot: Path | None) -> dict[str, Any]:
    images = parse_images(model_dir / "images.txt")
    points = parse_points(model_dir / "points3D.txt")
    camera_centers = np.array([item["center"] for item in images], dtype=np.float64)
    points_xyz = np.array([item["xyz"] for item in points], dtype=np.float64)
    errors = np.array([item["error"] for item in points], dtype=np.float64)
    track_lengths = np.array([item["track_length"] for item in points], dtype=np.float64)

    write_camera_csv(out_csv, images)
    plot_written = False
    if out_plot is not None and len(camera_centers) > 0 and len(points_xyz) > 0:
        plot_written = save_preview(out_plot, camera_centers, points_xyz)

    summary = {
        "model_dir": str(model_dir),
        "registered_images": len(images),
        "points3D": len(points),
        "camera_centers_csv": str(out_csv),
        "geometry_preview": str(out_plot) if out_plot and plot_written else None,
        "camera_bbox": bbox(camera_centers),
        "point_bbox": bbox(points_xyz),
        "point_reprojection_error_px": describe(errors),
        "point_track_length": describe(track_lengths),
    }

    if len(camera_centers) > 1:
        step_lengths = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
        summary["camera_step_length"] = describe(step_lengths)
        summary["camera_path_length"] = float(np.sum(step_lengths))
    else:
        summary["camera_step_length"] = describe(np.array([]))
        summary["camera_path_length"] = math.nan

    write_json(out_json, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, default=Path("data/colmap_train/sparse_txt"))
    parser.add_argument("--out-json", type=Path, default=Path("results/colmap/train_geometry_summary.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/colmap/train_camera_centers.csv"))
    parser.add_argument("--out-plot", type=Path, default=Path("results/colmap/train_geometry_preview.png"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_dir = resolve_path(args.model_dir)
    out_json = resolve_path(args.out_json)
    out_csv = resolve_path(args.out_csv)
    out_plot = resolve_path(args.out_plot)
    if model_dir is None or out_json is None or out_csv is None:
        raise ValueError("Model and output paths must be configured.")
    if args.dry_run:
        print(f"model_dir: {model_dir}")
        print(f"out_json: {out_json}")
        print(f"out_csv: {out_csv}")
        print(f"out_plot: {out_plot}")
        return 0
    summary = analyze(model_dir, out_json, out_csv, out_plot)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
