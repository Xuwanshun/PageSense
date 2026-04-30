from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = ".stage_cache"


class StageCache:
    """Per-stage JSON sidecars that record whether a stage has been run for given inputs."""

    def __init__(self, working_dir: Path) -> None:
        self._dir = working_dir / _CACHE_DIR

    def is_hit(self, stage_name: str, key: str) -> bool:
        sidecar = self._sidecar_path(stage_name)
        if not sidecar.exists():
            return False
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return payload.get("key") == key
        except Exception:
            return False

    def write(self, stage_name: str, key: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        self._sidecar_path(stage_name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def invalidate(self, stage_name: str) -> None:
        sidecar = self._sidecar_path(stage_name)
        if sidecar.exists():
            sidecar.unlink()
            logger.debug("Invalidated cache for stage %s", stage_name)

    def invalidate_all(self) -> None:
        if self._dir.exists():
            for sidecar in self._dir.glob("*.json"):
                sidecar.unlink()

    def _sidecar_path(self, stage_name: str) -> Path:
        return self._dir / f"{stage_name}.json"

    @staticmethod
    def compute_key(*parts: str) -> str:
        return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]
