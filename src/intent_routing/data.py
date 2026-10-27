"""Pinned dataset acquisition and validation; no corpus/model downloads from NLTK."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}/data/data_full.json"
SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"


def download(path):
    path = Path(path)
    if path.exists():
        load(path)
        return
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Dataset checksum mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def load(path):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Dataset checksum mismatch; use the pinned data_full.json")
    return json.loads(content)


def development(data):
    """Select only train/val/oos_val; final-test rows are never used here."""
    train, val, oos = data["train"], data["val"], data["oos_val"]
    labels = sorted({label for _, label in train})
    if len(labels) != 150 or set(Counter(y for _, y in train).values()) != {100}:
        raise ValueError("Expected 150 intents with 100 training examples each")
    if set(Counter(y for _, y in val).values()) != {20} or {y for _, y in val} != set(labels):
        raise ValueError("Expected 20 validation examples for each training intent")
    if len(oos) != 100 or any(y != "oos" for _, y in oos):
        raise ValueError("Expected 100 out-of-scope validation examples")
    return train, val + [[text, "out_of_scope"] for text, _ in oos], labels
