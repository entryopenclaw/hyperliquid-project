from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import MonitoringConfig
from .models import Incident
from .utils import to_jsonable, utc_now

LOGGER = logging.getLogger(__name__)


class MonitoringService:
    def __init__(self, config: MonitoringConfig, status_path: str):
        self.config = config
        self.status_path = status_path
        self.paused = False
        self.last_heartbeat = utc_now()
        self.metrics: dict[str, Any] = {}
        self.incidents: list[Incident] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def heartbeat(self) -> None:
        self.last_heartbeat = utc_now()
        self.write_status_file()

    def report_incident(self, incident: Incident) -> None:
        self.incidents.append(incident)
        LOGGER.warning("%s: %s", incident.severity, incident.title)
        self.write_status_file()

    def status(self) -> dict[str, Any]:
        return {
            "timestamp": utc_now().isoformat(),
            "paused": self.paused,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metrics": self.metrics,
            "incidents": [to_jsonable(item) for item in self.incidents[-20:]],
        }

    def write_status_file(self) -> None:
        Path(self.status_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_path, "w", encoding="utf-8") as handle:
            json.dump(self.status(), handle, indent=2, sort_keys=True)

    def serve(self) -> None:
        if not self.config.enable_http or self._server is not None:
            return
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                payload = json.dumps(service.status(), sort_keys=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer((self.config.host, self.config.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
