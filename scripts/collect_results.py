"""Collect result sizes and key output paths into summary files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from project_utils import DEFAULT_CONFIG, cfg_get, dir_size_bytes, ensure_dir, load_config, resolve_path, write_json


FIELDS = ["name", "path", "exists", "size_bytes", "size_mb"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    targets = {
        "colmap": cfg_get(config, "paths.result_colmap"),
        "nerfstudio": cfg_get(config, "paths.result_nerfstudio"),
        "3dgs": cfg_get(config, "paths.result_3dgs"),
        "vggt_gs": cfg_get(config, "paths.result_vggt_gs"),
        "metrics": "results/metrics.csv",
        "timing": "results/timing.csv",
    }
    rows = []
    for name, value in targets.items():
        path = resolve_path(value)
        size = dir_size_bytes(path) if path is not None else 0
        row = {
            "name": name,
            "path": str(path),
            "exists": bool(path and path.exists()),
            "size_bytes": size,
            "size_mb": f"{size / (1024 * 1024):.3f}",
        }
        rows.append(row)
        print(row)

    summary_json = resolve_path("results/summary.json")
    summary_csv = resolve_path("results/model_sizes.csv")
    assert summary_json is not None and summary_csv is not None
    if args.dry_run:
        print(f"Would write {summary_json} and {summary_csv}")
        return 0
    write_json(summary_json, rows)
    write_csv(summary_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
