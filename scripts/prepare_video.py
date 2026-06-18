"""Extract frames from the configured raw video with ffmpeg."""

from __future__ import annotations

import argparse
from pathlib import Path

from project_utils import (
    DEFAULT_CONFIG,
    cfg_get,
    ensure_dir,
    find_tool,
    load_config,
    resolve_path,
    run_command,
    write_json,
)


def build_ffmpeg_cmd(config: dict, video: Path, output_dir: Path) -> list[str] | None:
    ffmpeg = find_tool(config, "ffmpeg", "ffmpeg")
    if ffmpeg is None:
        return None
    fps = cfg_get(config, "data.extract_fps", 2)
    width = cfg_get(config, "data.target_width", 1280)
    height = cfg_get(config, "data.target_height", -1)
    quality = cfg_get(config, "data.jpeg_quality", 2)
    pattern = cfg_get(config, "data.frame_pattern", "frame_%06d.jpg")
    scale = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease"
    return [
        str(ffmpeg),
        "-y",
        "-i",
        str(video),
        "-vf",
        scale,
        "-q:v",
        str(quality),
        str(output_dir / pattern),
    ]


def cv2_target_size(src_w: int, src_h: int, target_w: int, target_h: int) -> tuple[int, int]:
    if target_w <= 0 and target_h <= 0:
        return src_w, src_h
    if target_w > 0 and target_h > 0:
        return target_w, target_h
    if target_w > 0:
        scale = target_w / src_w
        height = max(1, round(src_h * scale))
        return target_w, height
    scale = target_h / src_h
    width = max(1, round(src_w * scale))
    return width, target_h


def extract_with_cv2(config: dict, video: Path, output_dir: Path, dry_run: bool) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is unavailable and OpenCV cannot be imported. "
            "Install/configure ffmpeg or use a Conda env with cv2."
        ) from exc

    fps = float(cfg_get(config, "data.extract_fps", 2))
    target_w = int(cfg_get(config, "data.target_width", 1280))
    target_h = int(cfg_get(config, "data.target_height", -1))
    pattern = str(cfg_get(config, "data.frame_pattern", "frame_%06d.jpg"))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_w, out_h = cv2_target_size(src_w, src_h, target_w, target_h)
    duration = frame_count / src_fps if src_fps > 0 else None
    print(
        f"OpenCV fallback: source={src_w}x{src_h} fps={src_fps:.3f} "
        f"frames={frame_count} duration={duration} target={out_w}x{out_h} extract_fps={fps}"
    )
    if dry_run:
        cap.release()
        print(f"Would write frames to: {output_dir / pattern}")
        return

    ensure_dir(output_dir)
    saved = 0
    next_t = 0.0
    period = 1.0 / fps
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if src_fps > 0 and timestamp <= 0:
            timestamp = idx / src_fps
        if timestamp + 1e-9 >= next_t:
            resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            saved += 1
            name = pattern % saved if "%" in pattern else f"frame_{saved:06d}.jpg"
            ok, encoded = cv2.imencode(".jpg", resized)
            if not ok:
                raise RuntimeError(f"Failed to encode frame {saved}")
            encoded.tofile(str(output_dir / name))
            next_t += period
        idx += 1
    cap.release()
    write_json(
        resolve_path("logs/prepare_video_cv2.json"),
        {
            "video": str(video),
            "output_dir": str(output_dir),
            "source_fps": src_fps,
            "source_width": src_w,
            "source_height": src_h,
            "frame_count": frame_count,
            "target_width": out_w,
            "target_height": out_h,
            "extract_fps": fps,
            "saved_frames": saved,
        },
    )
    print(f"Saved {saved} frames to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    video = args.video or resolve_path(cfg_get(config, "data.video"))
    output_dir = args.output_dir or resolve_path(cfg_get(config, "paths.images"))
    if video is None or output_dir is None:
        raise ValueError("Video path or output image directory is not configured.")

    if not args.dry_run and not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    ensure_dir(output_dir)
    cmd = build_ffmpeg_cmd(config, video, output_dir)
    if cmd is not None:
        run_command(
            cmd,
            dry_run=args.dry_run,
            log_path=resolve_path("logs/prepare_video.json"),
        )
    else:
        print("ffmpeg is not configured or on PATH; using OpenCV fallback.")
        extract_with_cv2(config, video, output_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
