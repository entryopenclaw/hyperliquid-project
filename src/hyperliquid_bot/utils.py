from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def dumps_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True)


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_val = math.exp(-value)
        return 1.0 / (1.0 + exp_val)
    exp_val = math.exp(value)
    return exp_val / (1.0 + exp_val)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def interval_to_milliseconds(interval: str) -> int:
    if not interval:
        raise ValueError("interval must not be empty")
    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(f"invalid interval: {interval}") from exc

    multipliers = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported interval unit: {interval}")
    return value * multipliers[unit]
