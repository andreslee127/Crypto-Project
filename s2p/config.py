"""Load config.yaml (the single configuration file)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = ROOT / rel
    return cfg
