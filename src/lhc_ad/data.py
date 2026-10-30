"""Loading the top-tagging reference dataset, plus a toy generator for tests.

The public dataset (Kasieczka, Plehn et al., Zenodo record 2603256) ships
train.h5 / val.h5 / test.h5 as pandas HDFStores under key "table". Each row is
one jet with up to 200 pT-ordered constituents stored as columns
E_i, PX_i, PY_i, PZ_i (zero padded) and a label column "is_signal_new"
(1 = top, 0 = QCD).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_MAX_CONSTITUENTS = 200
LABEL_COLUMN = "is_signal_new"


@dataclass
class JetSample:
    """Constituent four-vectors (N, C, 4) as [E, px, py, pz] and labels (N,)."""

    constituents: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def select(self, mask: np.ndarray) -> "JetSample":
        return JetSample(self.constituents[mask], self.labels[mask])


def constituent_columns(n_constituents: int = N_MAX_CONSTITUENTS) -> list[str]:
    cols = []
    for i in range(n_constituents):
        cols += [f"E_{i}", f"PX_{i}", f"PY_{i}", f"PZ_{i}"]
    return cols


def load_top_tagging(
    path: str | Path,
    n_jets: int | None = None,
    n_constituents: int = N_MAX_CONSTITUENTS,
    start: int = 0,
) -> JetSample:
    """Read jets from one of the top-tagging HDF files."""
    stop = None if n_jets is None else start + n_jets
    df = pd.read_hdf(path, key="table", start=start, stop=stop)
    flat = df[constituent_columns(n_constituents)].to_numpy(dtype=np.float32)
    constituents = flat.reshape(len(df), n_constituents, 4)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    return JetSample(constituents, labels)


def make_toy_jets(
    n_jets: int,
    signal_fraction: float = 0.5,
    n_constituents: int = 60,
    seed: int = 0,
) -> JetSample:
    """Toy jets with the same layout as the real data.

    QCD-like jets have one hard prong and fewer constituents; top-like jets have
    three prongs and more constituents. This is only meant for exercising the
    pipeline, not for physics.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_jets) < signal_fraction).astype(np.int64)
    out = np.zeros((n_jets, n_constituents, 4), dtype=np.float32)

    for j in range(n_jets):
        is_top = labels[j] == 1
        jet_pt = rng.uniform(550.0, 650.0)
        jet_eta = rng.uniform(-2.0, 2.0)
        jet_phi = rng.uniform(-np.pi, np.pi)
        # Some QCD jets get a hard splitting so the classes overlap.
        n_prongs = 3 if is_top else int(rng.choice([1, 2], p=[0.7, 0.3]))
        mean_mult = 40 if is_top else 32
        mult = int(np.clip(rng.poisson(mean_mult), 5, n_constituents))

        prong_frac = rng.dirichlet(np.ones(n_prongs) * 3.0)
        spread = 0.25 if is_top else 0.12
        prong_deta = rng.normal(0.0, spread, n_prongs)
        prong_dphi = rng.normal(0.0, spread, n_prongs)

        which = rng.choice(n_prongs, size=mult, p=prong_frac)
        z = rng.dirichlet(np.ones(mult) * 0.6)
        pt = jet_pt * z
        eta = jet_eta + prong_deta[which] + rng.normal(0.0, 0.05, mult)
        phi = jet_phi + prong_dphi[which] + rng.normal(0.0, 0.05, mult)

        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        e = pt * np.cosh(eta)  # massless constituents
        order = np.argsort(-pt)
        out[j, :mult] = np.stack([e, px, py, pz], axis=1)[order]

    return JetSample(out, labels)
