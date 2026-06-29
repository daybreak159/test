"""Common CLI and logging helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable


def require_package(import_name: str, install_hint: str) -> None:
    try:
        __import__(import_name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{import_name}'. {install_hint}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now_ms() -> int:
    return int(time.time() * 1000)


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

