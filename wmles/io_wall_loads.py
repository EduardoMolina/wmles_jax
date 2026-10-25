"""CSV export for integrated wall forces and moments."""
import csv
from pathlib import Path

import numpy as np


class WallLoadCSVWriter:
    """Write per-face and total wall loads to a CSV file."""

    _header = (
        "step", "time_s", "block", "face",
        "moment_center_x_m", "moment_center_y_m", "moment_center_z_m",
        "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
    )

    def __init__(self, path, moment_center, append=False):
        self.path = Path(path)
        self.moment_center = np.asarray(moment_center, dtype=float)
        if self.moment_center.shape != (3,) or not np.all(np.isfinite(self.moment_center)):
            raise ValueError("moment_center must contain three finite coordinates")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if append and self.path.exists():
            with self.path.open(newline="") as stream:
                header = next(csv.reader(stream), None)
            if header != list(self._header):
                raise ValueError(f"{self.path}: existing wall-load CSV header does not match")
            return
        with self.path.open("w", newline="") as stream:
            csv.writer(stream).writerow(self._header)

    def write(self, step, time, loads):
        rows = []
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        for (block, face), (force, moment) in sorted(loads.items()):
            force, moment = np.asarray(force), np.asarray(moment)
            total_force += force
            total_moment += moment
            rows.append((step, time, block, face, *self.moment_center, *force, *moment))
        rows.append((step, time, "TOTAL", "total", *self.moment_center,
                     *total_force, *total_moment))
        with self.path.open("a", newline="") as stream:
            csv.writer(stream).writerows(rows)
