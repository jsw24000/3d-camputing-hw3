"""Inspect raw video metadata and summarize available camera parameters.

Phone videos usually do not expose enough calibration data for direct NeRF/GS
training. This script records what is actually available from the video file,
the extracted frames, and the COLMAP-estimated intrinsics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags

from project_utils import DEFAULT_CONFIG, cfg_get, list_images, load_config, resolve_path, write_json


def command_path(name: str) -> str | None:
    return shutil.which(name)


def find_ffprobe(config: dict[str, Any]) -> str | None:
    found = command_path("ffprobe")
    if found:
        return found
    ffmpeg_cfg = cfg_get(config, "env.ffmpeg")
    if ffmpeg_cfg:
        ffmpeg = resolve_path(ffmpeg_cfg)
        if ffmpeg is not None:
            sibling = ffmpeg.with_name("ffprobe.exe")
            if sibling.exists():
                return str(sibling)
    return None


def run_json_command(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    try:
        return {"ok": True, "data": json.loads(completed.stdout)}
    except json.JSONDecodeError:
        return {"ok": False, "returncode": completed.returncode, "stdout": completed.stdout[:1000]}


def inspect_video_cv2(video: Path) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    if not video.exists():
        return {"available": True, "exists": False}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"available": True, "exists": True, "opened": False}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else None
    cap.release()
    return {
        "available": True,
        "exists": True,
        "opened": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
    }


def inspect_video_ffprobe(config: dict[str, Any], video: Path) -> dict[str, Any]:
    ffprobe = find_ffprobe(config)
    if not ffprobe:
        return {"available": False}
    if not video.exists():
        return {"available": True, "exists": False, "path": ffprobe}
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]
    result = run_json_command(cmd)
    result["path"] = ffprobe
    return result


def exif_dict(image: Path) -> dict[str, Any]:
    if not image.exists():
        return {}
    try:
        with Image.open(image) as im:
            raw = im.getexif()
            decoded: dict[str, Any] = {}
            for key, value in raw.items():
                name = ExifTags.TAGS.get(key, str(key))
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                decoded[name] = str(value)
            return decoded
    except Exception as exc:
        return {"error": repr(exc)}


def image_probe(images_dir: Path, extensions: list[str]) -> dict[str, Any]:
    images = list_images(images_dir, extensions)
    if not images:
        return {"count": 0}
    first = images[0]
    with Image.open(first) as im:
        width, height = im.size
    exif = exif_dict(first)
    useful_exif_keys = [
        "Make",
        "Model",
        "LensModel",
        "FocalLength",
        "FocalLengthIn35mmFilm",
        "ExifImageWidth",
        "ExifImageHeight",
    ]
    useful_exif = {key: exif[key] for key in useful_exif_keys if key in exif}
    return {
        "count": len(images),
        "first_image": str(first),
        "first_size": [width, height],
        "first_exif_key_count": len(exif),
        "useful_exif": useful_exif,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def colmap_camera_summary() -> dict[str, Any]:
    all_summary = read_json(resolve_path("results/colmap/summary.json") or Path())
    train_summary = read_json(resolve_path("results/colmap/train_summary.json") or Path())
    return {
        "all_view": all_summary.get("camera", {}),
        "all_registered_images": all_summary.get("registered_images"),
        "all_points3D": all_summary.get("points3D"),
        "train_view": train_summary.get("camera", {}),
        "train_registered_images": train_summary.get("registered_images"),
        "train_points3D": train_summary.get("points3D"),
    }


def markdown_report(report: dict[str, Any]) -> str:
    cv2 = report.get("video_cv2", {})
    images = report.get("extracted_images", {})
    colmap = report.get("colmap_estimate", {})
    all_cam = colmap.get("all_view", {})
    train_cam = colmap.get("train_view", {})
    lines = [
        "# Camera Parameter Audit",
        "",
        "## Raw Video",
        "",
        f"- File: `{report.get('video')}`",
        f"- Exists: `{report.get('video_exists')}`",
        f"- Size MB: `{report.get('video_size_mb')}`",
        f"- OpenCV resolution: `{cv2.get('width')} x {cv2.get('height')}`",
        f"- OpenCV FPS: `{cv2.get('fps')}`",
        f"- OpenCV frame count: `{cv2.get('frame_count')}`",
        f"- OpenCV duration sec: `{cv2.get('duration_sec')}`",
        "",
        "## Extracted Frames",
        "",
        f"- Count: `{images.get('count')}`",
        f"- First frame size: `{images.get('first_size')}`",
        f"- Useful EXIF fields in first extracted frame: `{images.get('useful_exif')}`",
        "",
        "## COLMAP Estimated Intrinsics",
        "",
        f"- All-view camera: `{all_cam}`",
        f"- All-view registered images / points: `{colmap.get('all_registered_images')} / {colmap.get('all_points3D')}`",
        f"- Train-only camera: `{train_cam}`",
        f"- Train-only registered images / points: `{colmap.get('train_registered_images')} / {colmap.get('train_points3D')}`",
        "",
        "## Interpretation",
        "",
        "- The phone video provides resolution, frame rate, and duration, but not a full calibrated camera model.",
        "- Extracted frames may lose phone EXIF fields, especially when generated from video.",
        "- For this project, the reproducible camera parameters are the COLMAP/SfM estimates, not factory Huawei intrinsics.",
        "- VGGT can be used as an additional pose/point initialization comparison, but it is not required before COLMAP.",
    ]
    return "\n".join(lines) + "\n"


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    video = resolve_path(cfg_get(config, "data.video"))
    images_dir = resolve_path(cfg_get(config, "paths.images"))
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    if video is None or images_dir is None:
        raise ValueError("data.video or paths.images is not configured.")
    video_exists = video.exists()
    return {
        "video": str(video),
        "video_exists": video_exists,
        "video_size_bytes": video.stat().st_size if video_exists else None,
        "video_size_mb": round(video.stat().st_size / (1024 * 1024), 3) if video_exists else None,
        "video_cv2": inspect_video_cv2(video),
        "video_ffprobe": inspect_video_ffprobe(config, video),
        "extracted_images": image_probe(images_dir, extensions),
        "colmap_estimate": colmap_camera_summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-json", type=Path, default=Path("logs/video_camera_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/camera_parameters.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_json = resolve_path(args.out_json)
    out_md = resolve_path(args.out_md)
    if out_json is None or out_md is None:
        raise ValueError("Output paths are invalid.")
    report = build_report(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(f"Would write: {out_json}")
        print(f"Would write: {out_md}")
        return 0
    write_json(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown_report(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
