from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import StorageConfig
from .utils import ensure_parent, to_jsonable


class StorageManager:
    def __init__(self, config: StorageConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.sqlite_path = Path(config.sqlite_path)
        self.raw_events_path = Path(config.raw_events_path)
        self.features_path = Path(config.features_path)
        self.models_dir = Path(config.models_dir)
        self.reports_dir = Path(config.reports_dir)
        self.status_path = Path(config.status_path)
        self._ensure_layout()
        self._init_sqlite()

    def _ensure_layout(self) -> None:
        for path in [
            self.root_dir,
            self.models_dir,
            self.reports_dir,
            self.sqlite_path.parent,
            self.raw_events_path.parent,
            self.features_path.parent,
            self.status_path.parent,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _init_sqlite(self) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        cursor = connection.cursor()
        cursor.executescript(
            """
            create table if not exists executions (
                id integer primary key autoincrement,
                ts text not null,
                symbol text not null,
                action text not null,
                success integer not null,
                payload text not null
            );
            create table if not exists incidents (
                id integer primary key autoincrement,
                ts text not null,
                severity text not null,
                title text not null,
                details text not null
            );
            create table if not exists health_snapshots (
                id integer primary key autoincrement,
                ts text not null,
                payload text not null
            );
            create table if not exists training_runs (
                id integer primary key autoincrement,
                ts text not null,
                accepted integer not null,
                model_version text not null,
                metrics text not null,
                reasons text not null
            );
            """
        )
        connection.commit()
        connection.close()

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        ensure_parent(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")

    def record_raw_event(self, payload: dict[str, Any]) -> None:
        self.append_jsonl(self.raw_events_path, payload)

    def record_feature_row(self, payload: dict[str, Any]) -> None:
        self.append_jsonl(self.features_path, payload)

    def record_execution(self, payload: dict[str, Any]) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        cursor = connection.cursor()
        cursor.execute(
            "insert into executions(ts, symbol, action, success, payload) values (?, ?, ?, ?, ?)",
            (
                payload["timestamp"],
                payload["symbol"],
                payload["action"],
                int(payload["success"]),
                json.dumps(payload, sort_keys=True),
            ),
        )
        connection.commit()
        connection.close()

    def record_incident(self, payload: dict[str, Any]) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        cursor = connection.cursor()
        cursor.execute(
            "insert into incidents(ts, severity, title, details) values (?, ?, ?, ?)",
            (
                payload["timestamp"],
                payload["severity"],
                payload["title"],
                payload["details"],
            ),
        )
        connection.commit()
        connection.close()

    def record_health(self, payload: dict[str, Any]) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        cursor = connection.cursor()
        cursor.execute(
            "insert into health_snapshots(ts, payload) values (?, ?)",
            (payload["timestamp"], json.dumps(payload, sort_keys=True)),
        )
        connection.commit()
        connection.close()

    def record_training_run(self, payload: dict[str, Any]) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        cursor = connection.cursor()
        cursor.execute(
            "insert into training_runs(ts, accepted, model_version, metrics, reasons) values (?, ?, ?, ?, ?)",
            (
                payload["timestamp"],
                int(payload["accepted"]),
                payload["model_version"],
                json.dumps(payload["metrics"], sort_keys=True),
                json.dumps(payload["reasons"]),
            ),
        )
        connection.commit()
        connection.close()

    def load_feature_rows(self) -> list[dict[str, Any]]:
        if not self.features_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.features_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
