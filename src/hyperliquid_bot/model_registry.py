from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import ModelArtifact
from .utils import to_jsonable, utc_now


class ModelRegistry:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_pointer = self.models_dir / "active_model.json"

    def save(self, artifact: ModelArtifact) -> Path:
        target = self.models_dir / f"{artifact.version}.json"
        target.write_text(json.dumps(to_jsonable(artifact), indent=2), encoding="utf-8")
        return target

    def promote(self, artifact: ModelArtifact) -> None:
        self.save(artifact)
        self.active_pointer.write_text(
            json.dumps({"version": artifact.version, "updated_at": utc_now().isoformat()}, indent=2),
            encoding="utf-8",
        )

    def load(self, version: str) -> ModelArtifact | None:
        path = self.models_dir / f"{version}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelArtifact(
            version=data["version"],
            model_type=data["model_type"],
            created_at=utc_now() if "created_at" not in data else datetime.fromisoformat(data["created_at"]),
            weights={key: float(value) for key, value in data["weights"].items()},
            intercept=float(data["intercept"]),
            metrics={key: float(value) for key, value in data["metrics"].items()},
        )

    def load_active(self) -> ModelArtifact | None:
        if not self.active_pointer.exists():
            return None
        pointer = json.loads(self.active_pointer.read_text(encoding="utf-8"))
        return self.load(pointer["version"])
