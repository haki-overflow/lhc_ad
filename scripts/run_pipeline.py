"""Run the pipeline from a YAML config: python scripts/run_pipeline.py configs/toy.yaml"""

import argparse

import yaml

from lhc_ad.pipeline import Config, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=5 image_mode=count")
    args = parser.parse_args()
    with open(args.config) as f:
        values = yaml.safe_load(f) or {}
    for item in args.set:
        key, val = item.split("=", 1)
        values[key] = yaml.safe_load(val)
    run(Config(**values))


if __name__ == "__main__":
    main()
