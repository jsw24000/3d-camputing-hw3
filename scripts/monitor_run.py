"""Run a command with wall-clock and GPU resource monitoring.

The script streams the child process output, records start/end timestamps, and
periodically samples nvidia-smi when available. It writes a detailed JSON log
and appends a compact CSV row suitable for experiment reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from project_utils import DEFAULT_CONFIG, cfg_get, ensure_dir, load_config, quote_cmd, resolve_path


SUMMARY_FIELDS = [
    "method",
    "stage",
    "started_at",
    "finished_at",
    "elapsed_sec",
    "returncode",
    "command",
    "sample_count",
    "gpu_peak_memory_mib",
    "gpu_peak_utilization_percent",
    "log_json",
]


def query_gpu() -> list[dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    cmd = [
        nvidia_smi,
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            memory_used = int(float(parts[2]))
            memory_total = int(float(parts[3]))
            utilization = int(float(parts[4]))
        except ValueError:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "utilization_gpu_percent": utilization,
            }
        )
    return rows


def append_summary(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in SUMMARY_FIELDS})


def tee_pipe(pipe: Any, sink: Any, chunks: list[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            chunks.append(line)
            sink.write(line)
            sink.flush()
    finally:
        pipe.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--interval", type=float, default=1.0, help="GPU sampling interval in seconds.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise ValueError("Provide a command after --")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    config = load_config(args.config)
    project_root = resolve_path(cfg_get(config, "project.root", "."))
    assert project_root is not None
    out_json = args.out_json or resolve_path(f"logs/monitor_{args.method}_{args.stage}.json")
    out_csv = args.out_csv or resolve_path("results/resource_timing.csv")
    assert out_json is not None and out_csv is not None

    command_str = quote_cmd(args.cmd)
    print(command_str)
    if args.dry_run:
        print(f"Would write JSON log to {out_json}")
        print(f"Would append summary to {out_csv}")
        return 0

    samples: list[dict[str, Any]] = []
    stop_event = threading.Event()

    def sampler() -> None:
        while not stop_event.is_set():
            samples.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "gpus": query_gpu(),
                }
            )
            stop_event.wait(args.interval)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()

    sample_thread = threading.Thread(target=sampler, daemon=True)
    sample_thread.start()
    process = subprocess.Popen(
        [str(part) for part in args.cmd],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stdout_thread = threading.Thread(target=tee_pipe, args=(process.stdout, sys.stdout, stdout_parts), daemon=True)
    stderr_thread = threading.Thread(target=tee_pipe, args=(process.stderr, sys.stderr, stderr_parts), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    stop_event.set()
    sample_thread.join(timeout=args.interval + 1)

    elapsed = time.perf_counter() - t0
    finished_at = datetime.now().isoformat(timespec="seconds")
    peak_memory = 0
    peak_utilization = 0
    for sample in samples:
        for gpu in sample.get("gpus", []):
            peak_memory = max(peak_memory, int(gpu.get("memory_used_mib", 0)))
            peak_utilization = max(peak_utilization, int(gpu.get("utilization_gpu_percent", 0)))

    detail = {
        "method": args.method,
        "stage": args.stage,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": elapsed,
        "returncode": returncode,
        "command": command_str,
        "sample_interval_sec": args.interval,
        "sample_count": len(samples),
        "gpu_peak_memory_mib": peak_memory,
        "gpu_peak_utilization_percent": peak_utilization,
        "samples": samples,
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
    }
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    append_summary(
        out_csv,
        {
            **detail,
            "elapsed_sec": f"{elapsed:.6f}",
            "log_json": str(out_json),
        },
    )
    print(f"\nMonitor summary: {elapsed:.2f}s, peak GPU memory {peak_memory} MiB, peak GPU util {peak_utilization}%")
    print(f"Detailed log: {out_json}")
    print(f"Summary CSV: {out_csv}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
