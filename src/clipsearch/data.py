"""COCO val2017 as a search benchmark: 5,000 images, each with ~5 human-written captions (the queries) and object
annotations (the keyword tags a stock-photo library would have). Images are split by id into train/val/test, so no
image, and none of its captions, appears in more than one split."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    image_ids: list[int]
    files: list[str]
    tags: list[list[str]]  # object category names, one entry per annotated instance
    captions: list[str]
    caption_image: list[int]  # index into image_ids for each caption
    split: dict[str, list[int]]  # split name -> image indices


def load(root: str | Path, seed: int = 0, n_train: int = 3000, n_val: int = 1000) -> Dataset:
    root = Path(root)
    caps = json.loads((root / "annotations" / "captions_val2017.json").read_text())
    inst = json.loads((root / "annotations" / "instances_val2017.json").read_text())
    images = sorted(caps["images"], key=lambda im: im["id"])
    index = {im["id"]: i for i, im in enumerate(images)}
    names = {c["id"]: c["name"] for c in inst["categories"]}
    tags: list[list[str]] = [[] for _ in images]
    for a in inst["annotations"]:
        tags[index[a["image_id"]]].append(names[a["category_id"]])
    captions, owner = [], []
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        captions.append(" ".join(a["caption"].split()))
        owner.append(index[a["image_id"]])
    order = list(range(len(images)))
    random.Random(seed).shuffle(order)
    split = {
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train : n_train + n_val]),
        "test": sorted(order[n_train + n_val :]),
    }
    return Dataset([im["id"] for im in images], [im["file_name"] for im in images], tags, captions, owner, split)


def coco_url(file_name: str) -> str:
    return f"http://images.cocodataset.org/val2017/{file_name}"
