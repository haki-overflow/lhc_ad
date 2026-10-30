"""End-to-end run: data -> features -> normalise -> baselines -> AE -> audit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import baselines, evaluate, features, models, normalize
from .data import JetSample, load_top_tagging, make_toy_jets


@dataclass
class Config:
    name: str = "toy"
    # Data: either a directory with train.h5/val.h5/test.h5, or "toy".
    data: str = "toy"
    n_train: int | None = 100_000
    n_val: int | None = 20_000
    n_test: int | None = 50_000
    n_constituents: int = 200
    # Inputs to the AE.
    image_pixels: int = 40
    image_mode: str = "pt"  # "pt" or "count"
    image_norm: str = "sum"  # "sum", "max", "none"
    # Model / training.
    hidden: list[int] = field(default_factory=lambda: [256, 64])
    latent: int = 16
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0
    # Audit.
    audit_bins: int = 10
    out_dir: str = "runs"


def load_split(cfg: Config, split: str, n: int | None, seed_offset: int) -> JetSample:
    if cfg.data == "toy":
        return make_toy_jets(n or 1000, n_constituents=min(cfg.n_constituents, 60), seed=cfg.seed + seed_offset)
    return load_top_tagging(Path(cfg.data) / f"{split}.h5", n_jets=n, n_constituents=cfg.n_constituents)


def ae_inputs(sample: JetSample, cfg: Config) -> np.ndarray:
    imgs = features.jet_images(sample.constituents, n_pixels=cfg.image_pixels, mode=cfg.image_mode)
    return normalize.normalize_images(imgs, cfg.image_norm)


def run(cfg: Config, verbose: bool = True) -> dict:
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    # Anomaly detection setting: the AE only ever sees background (QCD).
    train = load_split(cfg, "train", cfg.n_train, 0)
    train = train.select(train.labels == 0)
    val = load_split(cfg, "val", cfg.n_val, 1)
    val = val.select(val.labels == 0)
    test = load_split(cfg, "test", cfg.n_test, 2)

    x_train, x_val, x_test = (ae_inputs(s, cfg) for s in (train, val, test))
    scaler = normalize.Standardizer().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(x) for x in (x_train, x_val, x_test))

    model, history = models.train_autoencoder(
        x_train, x_val, hidden=tuple(cfg.hidden), latent=cfg.latent, epochs=cfg.epochs,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed, verbose=verbose,
    )
    ae_score = models.reconstruction_error(model, x_test)

    hl = features.high_level_features(test.constituents)
    scores = {"autoencoder": ae_score, **baselines.baseline_scores(hl)}
    y = test.labels

    results = {
        "config": asdict(cfg),
        "n_jets": {"train_qcd": len(train), "val_qcd": len(val), "test": len(test), "test_top": int(y.sum())},
        "history": history,
        "metrics": {k: evaluate.summarize(y, s) for k, s in scores.items()},
        "auc_in_nconst_bins": {
            k: evaluate.binned_auc(y, s, hl["n_constituents"], cfg.audit_bins) for k, s in scores.items()
        },
        "auc_in_mass_bins": {
            k: evaluate.binned_auc(y, s, hl["mass"], cfg.audit_bins) for k, s in scores.items()
        },
        "ae_correlations": evaluate.correlations(ae_score, hl, y),
    }

    (out / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(out / "scores.npz", labels=y, **scores, **{f"hl_{k}": v for k, v in hl.items()})
    torch.save(model.state_dict(), out / "model.pt")
    if verbose:
        print_summary(results)
    return results


def print_summary(results: dict) -> None:
    print(f"\n{'score':<16}{'AUC':>8}{'rej@0.3':>10}{'AUC|nconst':>12}{'AUC|mass':>10}")
    for k, m in results["metrics"].items():
        nb = results["auc_in_nconst_bins"][k]["weighted_mean"]
        mb = results["auc_in_mass_bins"][k]["weighted_mean"]
        print(f"{k:<16}{m['auc']:>8.3f}{m['rej@0.3']:>10.1f}{nb:>12.3f}{mb:>10.3f}")
