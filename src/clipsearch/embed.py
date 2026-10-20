"""CLIP embeddings for every image and caption, computed once and cached."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# the OpenAI weights were trained with QuickGELU; the plain ViT-B-32 config would silently use GELU
MODEL, PRETRAINED = "ViT-B-32-quickgelu", "openai"


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    return model.eval().to(_device()), preprocess, open_clip.get_tokenizer(MODEL)


def encode_texts(model, tokenizer, texts: list[str], batch: int = 512) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            t = model.encode_text(tokenizer(texts[i : i + batch]).to(_device())).float()
            out.append(torch.nn.functional.normalize(t, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)


def encode_images(model, preprocess, paths: list[Path], batch: int = 64) -> np.ndarray:
    import torch
    from PIL import Image

    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch]]).to(_device())
            out.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def cached(data_root: str | Path, files: list[str], captions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    path = root / f"embeddings_{MODEL}_{PRETRAINED}.npz"
    if path.exists():
        z = np.load(path)
        if z["images"].shape[0] == len(files) and z["texts"].shape[0] == len(captions):
            return z["images"], z["texts"]
    model, preprocess, tokenizer = load_model()
    images = encode_images(model, preprocess, [root / "val2017" / f for f in files])
    texts = encode_texts(model, tokenizer, captions)
    np.savez(path, images=images, texts=texts)
    return images, texts
