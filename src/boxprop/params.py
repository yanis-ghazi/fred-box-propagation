from pathlib import Path

import yaml


def load(path: str | Path = "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())
