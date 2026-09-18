"""Shared CLI helpers: config loading, logging, paths."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(
    config_path: str | Path | None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load config.yaml and apply key=value overrides.

    Overrides use dot-notation: "ml.policy=advisory_only"
    """
    if config_path is None:
        config_path = Path("config.yaml")
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Invalid override (need KEY=VALUE): {ov}")
            key_path, value = ov.split("=", 1)
            _set_nested(cfg, key_path.split("."), _parse_value(value))

    return cfg


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _parse_value(s: str) -> Any:
    s = s.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"none", "null"}:
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass
    return s


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_format: bool = False,
) -> None:
    """Configure root logger. Forces UTF-8 stdout to avoid emoji crashes."""
    # Force UTF-8 on stdout/stderr (Windows default is cp1252)
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    fmt = (
        '{"ts": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": "%(message)s"}'
        if json_format
        else "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )


def resolve_run_dir(base: str | Path, run_id: str) -> Path:
    """Return the canonical run directory for a given run_id."""
    return Path(base) / run_id


def print_kv(d: dict[str, Any], indent: int = 2) -> None:
    """Pretty-print a flat dict."""
    pad = " " * indent
    width = max((len(str(k)) for k in d), default=0)
    for k, v in d.items():
        print(f"{pad}{str(k):<{width}} : {v}")


def confirm(prompt: str) -> bool:
    """Ask for user confirmation (Y/n)."""
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def fail(msg: str, code: int = 2) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return code