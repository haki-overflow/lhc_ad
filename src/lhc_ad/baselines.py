"""Single-observable anomaly scores used as reference points for the AE.

Each score is oriented so that larger means "more anomalous" (more top-like),
which is what the evaluation code expects.
"""

from __future__ import annotations

import numpy as np

TOP_MASS = 173.0


def baseline_scores(hl: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "n_constituents": hl["n_constituents"],
        "mass": hl["mass"],
        "mass_window": -np.abs(hl["mass"] - TOP_MASS),
        "inv_tau32": -hl["tau32"],
        "inv_tau21": -hl["tau21"],
    }
