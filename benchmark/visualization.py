"""Frame capture helpers for benchmark task runs."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np


def save_frame(frame: np.ndarray, output_path: Path) -> str:
    """Persist one RGB frame and return its absolute path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, frame)
    return str(output_path.resolve())
