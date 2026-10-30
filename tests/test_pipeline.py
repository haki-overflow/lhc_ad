import numpy as np

from lhc_ad import evaluate, features
from lhc_ad.data import make_toy_jets
from lhc_ad.pipeline import Config, run


def test_toy_layout():
    s = make_toy_jets(50, n_constituents=40, seed=1)
    assert s.constituents.shape == (50, 40, 4)
    assert set(np.unique(s.labels)) <= {0, 1}


def test_features_sane():
    s = make_toy_jets(200, n_constituents=60, seed=2)
    hl = features.high_level_features(s.constituents)
    assert np.all(hl["n_constituents"] >= 5)
    assert np.all(hl["mass"] >= 0)
    assert np.all((hl["tau32"] >= 0) & np.isfinite(hl["tau32"]))
    # Toy tops are 3-prong, so they should be heavier on average.
    assert hl["mass"][s.labels == 1].mean() > hl["mass"][s.labels == 0].mean()


def test_images_pt_sum_to_one():
    s = make_toy_jets(30, seed=3)
    img = features.jet_images(s.constituents, n_pixels=16, mode="pt")
    assert img.shape == (30, 16, 16)
    assert np.all(img.sum(axis=(1, 2)) <= 1.0 + 1e-5)
    counts = features.jet_images(s.constituents, n_pixels=16, mode="count")
    assert np.all(counts.sum(axis=(1, 2)) <= features.n_constituents(s.constituents))


def test_binned_auc_kills_pure_counting():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    nconst = rng.poisson(30 + 15 * y)
    res = evaluate.binned_auc(y, nconst.astype(float) + rng.random(n) * 1e-3, nconst.astype(float), n_bins=8)
    assert abs(res["weighted_mean"] - 0.5) < 0.1


def test_pipeline_end_to_end(tmp_path):
    cfg = Config(n_train=600, n_val=200, n_test=600, image_pixels=10, epochs=2,
                 hidden=[32], latent=4, out_dir=str(tmp_path))
    res = run(cfg, verbose=False)
    assert 0.0 <= res["metrics"]["autoencoder"]["auc"] <= 1.0
    assert (tmp_path / "toy" / "results.json").exists()
