"""Dispatch to the canonical repository script without duplicating logic."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[4] / "scripts" / "verify.py"), run_name="__main__"
    )
