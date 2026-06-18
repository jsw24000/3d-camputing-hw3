"""Shared helpers for the assignment pipeline scripts."""

from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "project.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    import yaml  # type: ignore

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data or {}


def cfg_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_path(value: str | Path | None, root: Path = ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs(config: dict[str, Any]) -> None:
    for value in (config.get("paths") or {}).values():
        path = resolve_path(value)
        if path is not None:
            ensure_dir(path)


def find_tool(config: dict[str, Any], key: str, command_name: str) -> Path | str | None:
    configured = cfg_get(config, f"env.{key}")
    if configured:
        path = resolve_path(configured)
        if path and path.exists():
            return path
    found = shutil.which(command_name)
    return found


def quote_cmd(cmd: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def command_path(path: Path | str, cwd: Path = ROOT) -> str:
    """Return a command argument path, relative to cwd when possible.

    Some Windows command launchers mishandle non-ASCII absolute path arguments
    across Conda environments. Keeping project-local paths relative avoids that
    fragile boundary while preserving the process working directory.
    """
    path = Path(path)
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path: Path | str, row: dict[str, Any], fieldnames: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def list_images(directory: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {ext.lower() for ext in extensions}
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def sample_gpu() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False}
    cmd = [
        nvidia_smi,
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        if completed.returncode != 0:
            return {"available": False, "error": completed.stderr.strip()}
        rows = []
        for line in completed.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                rows.append(
                    {
                        "name": parts[0],
                        "memory_used_mib": parts[1],
                        "memory_total_mib": parts[2],
                        "utilization_gpu_percent": parts[3],
                    }
                )
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    started_at: str
    finished_at: str
    elapsed_sec: float
    stdout: str
    stderr: str


def run_command(
    cmd: list[str | Path],
    *,
    dry_run: bool = False,
    cwd: Path = ROOT,
    log_path: Path | None = None,
    check: bool = True,
    stream: bool = False,
) -> CommandResult | None:
    str_cmd = [str(part) for part in cmd]
    print(quote_cmd(str_cmd))
    if dry_run:
        return None
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    env = os.environ.copy()
    mpl_config_dir = ROOT / "logs" / "matplotlib"
    ensure_dir(mpl_config_dir)
    env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    torch_home = ROOT / "logs" / "torch"
    ensure_dir(torch_home)
    env.setdefault("TORCH_HOME", str(torch_home))
    if stream:
        completed_process = subprocess.Popen(
            str_cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        def tee_pipe(pipe: Any, sink: Any, chunks: list[str]) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    chunks.append(line)
                    sink.write(line)
                    sink.flush()
            finally:
                pipe.close()

        stdout_thread = threading.Thread(
            target=tee_pipe,
            args=(completed_process.stdout, sys.stdout, stdout_parts),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=tee_pipe,
            args=(completed_process.stderr, sys.stderr, stderr_parts),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = completed_process.wait()
        stdout_thread.join()
        stderr_thread.join()
        elapsed = time.perf_counter() - t0
        finished = datetime.now().isoformat(timespec="seconds")
        result = CommandResult(
            command=str_cmd,
            returncode=returncode,
            started_at=started,
            finished_at=finished,
            elapsed_sec=elapsed,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
        )
        if log_path is not None:
            ensure_dir(log_path.parent)
            log_path.write_text(
                json.dumps(result.__dict__, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if check and returncode != 0:
            raise RuntimeError(f"Command failed with code {returncode}: {quote_cmd(str_cmd)}")
        return result

    completed = subprocess.run(
        str_cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - t0
    finished = datetime.now().isoformat(timespec="seconds")
    result = CommandResult(
        command=str_cmd,
        returncode=completed.returncode,
        started_at=started,
        finished_at=finished,
        elapsed_sec=elapsed,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if log_path is not None:
        ensure_dir(log_path.parent)
        log_path.write_text(
            json.dumps(result.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {quote_cmd(str_cmd)}")
    return result


def script_path(repo: Path, relative: str) -> Path:
    return repo / relative
