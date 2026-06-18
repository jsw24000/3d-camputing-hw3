"""Create a Nerfstudio dataset that matches an old checkpoint's train count.

The original Nerfacto checkpoint was trained from 46 physical train images, but
Nerfstudio's fraction split used only the first 42 as train cameras. This helper
keeps those 42 train filenames so the checkpoint embedding tables still match,
while setting the test split to the project-wide fixed five held-out views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-transforms", type=Path, default=Path("data/nerfstudio_colmap_train/transforms.json"))
    parser.add_argument("--test-transforms", type=Path, default=Path("data/nerfstudio_colmap_test/transforms.json"))
    parser.add_argument("--old-train-count", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data/nerfstudio_colmap_test5_compat"))
    args = parser.parse_args()

    train_data = load_json(args.train_transforms)
    test_data = load_json(args.test_transforms)
    train_frames = list(train_data.get("frames", []))
    test_frames = list(test_data.get("frames", []))

    train_filenames = [frame["file_path"] for frame in train_frames[: args.old_train_count]]
    test_filenames = [frame["file_path"] for frame in test_frames]

    out_data = dict(train_data)
    out_data["frames"] = train_frames + test_frames
    out_data["train_filenames"] = train_filenames
    out_data["val_filenames"] = test_filenames
    out_data["test_filenames"] = test_filenames

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "transforms.json"
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_data['frames'])} frames: {out_path}")
    print(f"train_filenames={len(train_filenames)} test_filenames={len(test_filenames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
