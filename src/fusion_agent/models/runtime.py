"""Local model configuration kept separate from the Agent's selectable plan."""

import hashlib
import importlib
import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from fusion_agent.schemas import Contract


class ModelOptions(Contract):
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    threads: int = Field(default=2, ge=1, le=32)
    icig_weights: str | None = None
    letters_weights: str | None = None
    visible_y: str | None = None
    valid_mask: str | None = None
    material_mask: str | None = None
    material_mode: Literal["visible", "valid"] = "visible"

    def checkpoint(self, method: str) -> Path:
        filename = {"icig": "model.pth", "letters": "fusion.pt"}[method]
        configured = getattr(self, f"{method}_weights") or os.getenv(
            f"FUSION_{method.upper()}_WEIGHTS"
        )
        path = Path(configured) if configured else Path("weights") / method / filename
        if not path.is_file():
            raise ValueError(
                f"Missing {method} checkpoint. Run scripts/import_research_weights.py "
                f"with your original archive, or set FUSION_{method.upper()}_WEIGHTS. "
                "Checkpoints are not bundled in Git."
            )
        return path


def load_adapter(method: str):
    try:
        return importlib.import_module(f"fusion_agent.models.{method}")
    except ImportError as exc:
        raise RuntimeError("Install research dependencies: pip install -e '.[research]'") from exc


def file_evidence(path: str | Path) -> dict:
    path = Path(path)
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
