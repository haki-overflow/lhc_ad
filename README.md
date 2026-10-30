# lhc_ad

**Physics or particle counting? Auditing what a jet autoencoder actually uses.**

An autoencoder is trained on QCD jets only and scored by reconstruction error;
top jets play the anomaly. The question is how much of its top-vs-QCD
separation comes from jet substructure (mass, prongs) and how much from simply
counting constituents.

## Pipeline

| Stage | Module |
|---|---|
| Load data (top-tagging HDF, or toy jets) | `lhc_ad.data` |
| Features: kinematics, mass, multiplicity, N-subjettiness, jet images | `lhc_ad.features` |
| Normalisation (fit on QCD training set only) | `lhc_ad.normalize` |
| Single-observable baselines | `lhc_ad.baselines` |
| Dense autoencoder, MSE anomaly score | `lhc_ad.models` |
| Metrics + audit (AUC inside n_constituents / mass bins, correlations) | `lhc_ad.evaluate` |
| End-to-end run, writes `runs/<name>/results.json` | `lhc_ad.pipeline` |

The audit's key number is **AUC|nconst**: the AUC averaged over bins of fixed
constituent multiplicity. A score that only counts particles drops to ~0.5 there.
Running with `image_mode=count` (pixels hold constituent counts rather than pt
fractions) gives the deliberately "counting" variant to compare against.

## Usage

```bash
pip install -e ".[dev]"
pytest
python scripts/run_pipeline.py configs/toy.yaml            # no data needed
python scripts/run_pipeline.py configs/top_tagging.yaml    # real data
python scripts/run_pipeline.py configs/top_tagging.yaml --set name=top_count image_mode=count
```

## Data

Top-tagging reference dataset (Kasieczka, Plehn et al.),
<https://zenodo.org/records/2603256>. Put `train.h5`, `val.h5`, `test.h5` in
`data/top_tagging/` (git-ignored).
