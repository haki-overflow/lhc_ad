"""Input normalisation, always fit on the background-only training set."""

from __future__ import annotations

import numpy as np


class Standardizer:
    """Per-feature (x - mean) / std, fit on training data only."""

    def __init__(self, min_std: float = 1e-3):
        # Features (e.g. pixels) that barely vary in training are left unscaled
        # rather than divided by ~0, which would explode on unseen jets.
        self.min_std = min_std
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.std_ = np.where(std < self.min_std, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer used before fit")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}


def normalize_images(images: np.ndarray, scheme: str = "sum") -> np.ndarray:
    """Flatten images and rescale them per jet.

    "sum": divide by the total so each image sums to 1 (removes overall scale).
    "max": divide by the brightest pixel.
    "none": leave as is.
    """
    flat = images.reshape(len(images), -1).astype(np.float32)
    if scheme == "none":
        return flat
    if scheme == "sum":
        denom = flat.sum(axis=1, keepdims=True)
    elif scheme == "max":
        denom = flat.max(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown normalisation {scheme!r}")
    return flat / np.maximum(denom, 1e-8)
