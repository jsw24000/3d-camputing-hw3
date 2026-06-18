"""Evaluate all configured rendered-output directories.

Each method keeps its own renderer, but the assignment metrics should be
collected with one shared script and one shared CSV schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluate_render import evaluate, match_pairs
from project_utils import DEFAULT_CONFIG, cfg_get, load_config, resolve_path


def method_items(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return cfg_get(config, "evaluation.methods", {}) or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", default="all", help="Method name from evaluation.methods, or all.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    extensions = cfg_get(config, "data.image_extensions", [".jpg", ".jpeg", ".png"])
    out = args.out or resolve_path(cfg_get(config, "evaluation.metrics_csv", "results/metrics.csv"))
    if out is None:
        raise ValueError("Metrics CSV path is not configured.")

    methods = method_items(config)
    if args.method != "all":
        if args.method not in methods:
            raise KeyError(f"Unknown method '{args.method}'. Available: {', '.join(methods)}")
        methods = {args.method: methods[args.method]}

    failures = 0
    for name, spec in methods.items():
        pred_dir = resolve_path(spec.get("pred_dir"))
        gt_dir = resolve_path(spec.get("gt_dir"))
        if pred_dir is None or gt_dir is None:
            print(f"[{name}] skipped: pred_dir or gt_dir is not configured")
            failures += 1
            continue
        pairs = match_pairs(pred_dir, gt_dir, extensions)
        print(f"[{name}] pred={pred_dir} gt={gt_dir} pairs={len(pairs)}")
        if args.dry_run:
            continue
        if not pred_dir.exists() or not gt_dir.exists() or not pairs:
            failures += 1
            continue
        evaluate(name, pred_dir, gt_dir, out, extensions)

    if args.dry_run:
        print(f"Would write metrics to: {out}")
        return 0
    if failures:
        print(f"Completed with {failures} skipped/missing method(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
