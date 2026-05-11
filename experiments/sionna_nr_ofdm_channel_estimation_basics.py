"""Compatibility entry point for the Sionna 2.x NR/OFDM waveform example."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = (
        Path(__file__).resolve().parent
        / "sionna_learning"
        / "01_nr_pusch_ofdm_waveform.py"
    )
    runpy.run_path(str(script), run_name="__main__")
