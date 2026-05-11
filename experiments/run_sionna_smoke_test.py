"""Compatibility entry point for the Sionna 2.x smoke test."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parents[1] / "scripts" / "sionna_smoke_test.py"
    runpy.run_path(str(script), run_name="__main__")
