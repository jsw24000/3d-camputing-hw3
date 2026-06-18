"""Run a command and append timing/resource metadata to results/timing.csv."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from project_utils import DEFAULT_CONFIG, append_csv, cfg_get, load_config, quote_cmd, resolve_path, sample_gpu, write_json


FIELDS = [
    "method",
    "stage",
    "started_at",
    "finished_at",
    "elapsed_sec",
    "returncode",
    "command",
    "gpu_before",
    "gpu_after",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", default="run")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise ValueError("Provide a command after --")

    config = load_config(args.config)
    out = args.out or resolve_path("results/timing.csv")
    assert out is not None
    command_str = quote_cmd(args.cmd)
    print(command_str)
    if args.dry_run:
        print(f"Would append timing to {out}")
        return 0

    gpu_before = sample_gpu()
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    completed = subprocess.run(args.cmd, cwd=resolve_path(cfg_get(config, "project.root", ".")))
    elapsed = time.perf_counter() - t0
    finished = datetime.now().isoformat(timespec="seconds")
    gpu_after = sample_gpu()

    row = {
        "method": args.method,
        "stage": args.stage,
        "started_at": started,
        "finished_at": finished,
        "elapsed_sec": f"{elapsed:.6f}",
        "returncode": completed.returncode,
        "command": command_str,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }
    append_csv(out, row, FIELDS)
    write_json(resolve_path(f"logs/timed_{args.method}_{args.stage}.json"), row)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
