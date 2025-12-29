import json
from pathlib import Path


def load_label_map(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def map_label(raw_label, label_map):
    if label_map is None:
        return raw_label
    return label_map.get(raw_label, raw_label)
