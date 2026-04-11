"""Run snapshot manager for persisting and loading test results.

Stores each run as a JSON snapshot in its own directory under .ragdiff/runs/,
and maintains a HEAD.json pointer to the latest run for easy comparison.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from rag_diff.models import RunSnapshot


class RunManager:
    """Manages run snapshots in the .ragdiff directory."""

    def __init__(self, base_dir: Path | str = ".ragdiff"):
        self.base_dir = Path(base_dir)

    def save(self, snapshot: RunSnapshot) -> Path:
        """Save a run snapshot to disk and update HEAD."""
        run_dir = self.base_dir / "runs" / snapshot.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = run_dir / "snapshot.json"
        snapshot_path.write_text(
            snapshot.model_dump_json(indent=2),
            encoding="utf-8",
        )

        self._update_head(snapshot.run_id)
        return snapshot_path

    def load(self, run_id: str) -> RunSnapshot:
        """Load a run snapshot by its ID."""
        snapshot_path = self.base_dir / "runs" / run_id / "snapshot.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Run '{run_id}' not found at {snapshot_path}"
            )

        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return RunSnapshot(**data)

    def get_head(self) -> str | None:
        """Get the run ID of the latest run, or None if no runs exist."""
        head_path = self.base_dir / "HEAD.json"
        if not head_path.exists():
            return None
        data = json.loads(head_path.read_text(encoding="utf-8"))
        return data.get("latest_run")

    def list_runs(self) -> list[str]:
        """List all run IDs, sorted alphabetically (chronologically)."""
        runs_dir = self.base_dir / "runs"
        if not runs_dir.exists():
            return []
        return sorted(
            d.name
            for d in runs_dir.iterdir()
            if d.is_dir() and (d / "snapshot.json").exists()
        )

    def generate_run_id(self) -> str:
        """Generate a unique run ID with embedded timestamp.

        Format: run_YYYYMMDD_HHMM_<8-char-uuid>
        """
        now = datetime.now()
        short_uuid = uuid.uuid4().hex[:8]
        return f"run_{now:%Y%m%d}_{now:%H%M}_{short_uuid}"

    def _update_head(self, run_id: str) -> None:
        """Update the HEAD pointer to the given run."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        head_path = self.base_dir / "HEAD.json"
        head_path.write_text(
            json.dumps({"latest_run": run_id}, indent=2),
            encoding="utf-8",
        )
