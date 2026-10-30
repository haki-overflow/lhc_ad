"""Jet- and constituent-level features computed from [E, px, py, pz] arrays.

Everything is vectorised over jets; padded constituents have E == 0 and are
masked out.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def constituent_mask(constituents: np.ndarray) -> np.ndarray:
    return constituents[..., 0] > 0


def constituent_kinematics(constituents: np.ndarray):
    """Return pt, eta, phi (each (N, C)) and the validity mask."""
    e, px, py, pz = np.moveaxis(constituents, -1, 0)
    mask = e > 0
    pt = np.hypot(px, py)
    eta = np.where(mask, np.arcsinh(pz / np.maximum(pt, EPS)), 0.0)
    phi = np.where(mask, np.arctan2(py, px), 0.0)
    return pt, eta, phi, mask


def jet_four_vector(constituents: np.ndarray) -> np.ndarray:
    return constituents.sum(axis=1)


def jet_kinematics(constituents: np.ndarray) -> dict[str, np.ndarray]:
    e, px, py, pz = np.moveaxis(jet_four_vector(constituents), -1, 0)
    pt = np.hypot(px, py)
    m2 = e**2 - px**2 - py**2 - pz**2
    return {
        "pt": pt,
        "eta": np.arcsinh(pz / np.maximum(pt, EPS)),
        "phi": np.arctan2(py, px),
        "mass": np.sqrt(np.clip(m2, 0.0, None)),
        "energy": e,
    }


def n_constituents(constituents: np.ndarray) -> np.ndarray:
    return constituent_mask(constituents).sum(axis=1)


def wrap_phi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def relative_coordinates(constituents: np.ndarray):
    """deta, dphi relative to the jet axis and pt fraction, zeroed on padding."""
    pt, eta, phi, mask = constituent_kinematics(constituents)
    jet = jet_kinematics(constituents)
    deta = np.where(mask, eta - jet["eta"][:, None], 0.0)
    dphi = np.where(mask, wrap_phi(phi - jet["phi"][:, None]), 0.0)
    frac = pt / np.maximum(pt.sum(axis=1, keepdims=True), EPS)
    return deta, dphi, frac, mask


def nsubjettiness(
    constituents: np.ndarray,
    n_axes: int,
    radius: float = 0.8,
    n_iter: int = 5,
) -> np.ndarray:
    """N-subjettiness tau_N (beta = 1).

    Axes come from a pt-weighted k-means in (eta, phi) seeded on the N hardest
    constituents. This approximates the usual exclusive-kt / one-pass
    minimisation and is adequate for the baseline comparison here.
    """
    deta, dphi, _, mask = relative_coordinates(constituents)
    pt, _, _, _ = constituent_kinematics(constituents)
    pt = np.where(mask, pt, 0.0)
    pts = np.stack([deta, dphi], axis=-1)  # (N, C, 2)

    # Constituents are pT ordered, so the first n_axes are the hardest.
    axes = pts[:, :n_axes].copy()
    for _ in range(n_iter):
        d2 = ((pts[:, :, None, :] - axes[:, None, :, :]) ** 2).sum(-1)  # (N, C, K)
        assign = d2.argmin(-1)
        onehot = (assign[..., None] == np.arange(n_axes)) * pt[..., None]
        wsum = onehot.sum(axis=1)  # (N, K)
        new_axes = np.einsum("nck,ncd->nkd", onehot, pts) / np.maximum(wsum, EPS)[..., None]
        axes = np.where(wsum[..., None] > 0, new_axes, axes)

    dr = np.sqrt(((pts[:, :, None, :] - axes[:, None, :, :]) ** 2).sum(-1)).min(-1)
    return (pt * dr).sum(axis=1) / np.maximum(pt.sum(axis=1) * radius, EPS)


def jet_images(
    constituents: np.ndarray,
    n_pixels: int = 40,
    extent: float = 1.0,
    mode: str = "pt",
    preprocess: bool = True,
) -> np.ndarray:
    """Pixelised (deta, dphi) jet images, shape (N, n_pixels, n_pixels).

    mode="pt" fills each pixel with the summed pt fraction (sums to 1 per jet);
    mode="count" fills it with the number of constituents, which carries the
    multiplicity information directly and is the "particle counting" variant.

    With preprocess=True the image is centred on the pt centroid, rotated so
    the principal axis is vertical, and flipped so the hardest content sits
    in the positive quadrant.
    """
    deta, dphi, frac, mask = relative_coordinates(constituents)
    x, y = deta, dphi
    if preprocess:
        x, y = _centre_rotate_flip(x, y, frac, mask)

    if mode == "pt":
        weight = frac
    elif mode == "count":
        weight = mask.astype(np.float64)
    else:
        raise ValueError(f"unknown image mode {mode!r}")

    edges = np.linspace(-extent, extent, n_pixels + 1)
    ix = np.clip(np.digitize(x, edges) - 1, 0, n_pixels - 1)
    iy = np.clip(np.digitize(y, edges) - 1, 0, n_pixels - 1)
    inside = mask & (np.abs(x) < extent) & (np.abs(y) < extent)

    n = len(constituents)
    images = np.zeros((n, n_pixels * n_pixels), dtype=np.float32)
    flat_idx = ix * n_pixels + iy
    rows = np.broadcast_to(np.arange(n)[:, None], flat_idx.shape)
    np.add.at(images, (rows[inside], flat_idx[inside]), weight[inside])
    return images.reshape(n, n_pixels, n_pixels)


def _centre_rotate_flip(x, y, w, mask):
    w = np.where(mask, w, 0.0)
    wsum = np.maximum(w.sum(axis=1, keepdims=True), EPS)
    x = np.where(mask, x - (w * x).sum(1, keepdims=True) / wsum, 0.0)
    y = np.where(mask, y - (w * y).sum(1, keepdims=True) / wsum, 0.0)

    # Principal axis of the pt-weighted second-moment tensor.
    sxx = (w * x * x).sum(1)
    syy = (w * y * y).sum(1)
    sxy = (w * x * y).sum(1)
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    # Rotate so the principal axis points along +y.
    angle = (np.pi / 2 - theta)[:, None]
    c, s = np.cos(angle), np.sin(angle)
    xr, yr = c * x - s * y, s * x + c * y

    # Flip so the pt-weighted half-planes are in the positive direction.
    sign_x = np.where((w * xr).sum(1, keepdims=True) < 0, -1.0, 1.0)
    sign_y = np.where((w * yr).sum(1, keepdims=True) < 0, -1.0, 1.0)
    return np.where(mask, xr * sign_x, 0.0), np.where(mask, yr * sign_y, 0.0)


def high_level_features(constituents: np.ndarray) -> dict[str, np.ndarray]:
    """The interpretable observables the audit compares the AE score against."""
    jet = jet_kinematics(constituents)
    tau1 = nsubjettiness(constituents, 1)
    tau2 = nsubjettiness(constituents, 2)
    tau3 = nsubjettiness(constituents, 3)
    return {
        "pt": jet["pt"],
        "mass": jet["mass"],
        "n_constituents": n_constituents(constituents).astype(np.float64),
        "tau1": tau1,
        "tau2": tau2,
        "tau3": tau3,
        "tau21": tau2 / np.maximum(tau1, EPS),
        "tau32": tau3 / np.maximum(tau2, EPS),
    }
