"""A small trainable layer on top of frozen CLIP embeddings.

Residual linear maps for text and images, initialised to zero so training starts exactly at CLIP, trained with the
symmetric contrastive (InfoNCE) loss on the training images and stopped early on validation recall at 1. With only
3,000 training images the risk is overfitting, so the validation split, not the training loss, decides when to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainResult:
    text_weight: np.ndarray
    image_weight: np.ndarray
    best_epoch: int
    history: list[dict] = field(default_factory=list)

    def text(self, x: np.ndarray) -> np.ndarray:
        y = x + x @ self.text_weight.T
        return y / np.linalg.norm(y, axis=1, keepdims=True)

    def image(self, x: np.ndarray) -> np.ndarray:
        y = x + x @ self.image_weight.T
        return y / np.linalg.norm(y, axis=1, keepdims=True)


def _val_r1(
    res: TrainResult, img: np.ndarray, txt: np.ndarray, val_imgs: np.ndarray, val_caps: np.ndarray, cap_img: np.ndarray
) -> float:
    pos = {g: i for i, g in enumerate(val_imgs)}
    target = np.array([pos[cap_img[c]] for c in val_caps])
    s = res.text(txt[val_caps]) @ res.image(img[val_imgs]).T
    return float((s.argmax(axis=1) == target).mean())


def train(
    img: np.ndarray,
    txt: np.ndarray,
    cap_img: np.ndarray,
    train_imgs: list[int],
    val_imgs: list[int],
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    temperature: float = 0.01,
    batch: int = 256,
    patience: int = 6,
    seed: int = 0,
) -> TrainResult:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = img.shape[1]
    wt = torch.zeros(d, d, requires_grad=True)
    wi = torch.zeros(d, d, requires_grad=True)
    opt = torch.optim.AdamW([wt, wi], lr=lr, weight_decay=weight_decay)
    img_t, txt_t = torch.from_numpy(img), torch.from_numpy(txt)
    caps_of: dict[int, list[int]] = {}
    for c, g in enumerate(cap_img):
        caps_of.setdefault(int(g), []).append(c)
    train_imgs_a, val_imgs_a = np.array(train_imgs), np.array(val_imgs)
    val_caps = np.array([c for g in val_imgs for c in caps_of.get(g, [])])

    def snapshot() -> TrainResult:
        return TrainResult(wt.detach().numpy().copy(), wi.detach().numpy().copy(), 0)

    best = snapshot()
    best_r1 = _val_r1(best, img, txt, val_imgs_a, val_caps, cap_img)
    history = [{"epoch": 0, "val_R@1": best_r1}]
    stale = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train_imgs_a)
        losses = []
        for i in range(0, len(order), batch):
            ims = order[i : i + batch]
            caps = [caps_of[int(g)][rng.integers(len(caps_of[int(g)]))] for g in ims]  # one caption per image
            t = F.normalize(txt_t[caps] + txt_t[caps] @ wt.T, dim=-1)
            v = F.normalize(img_t[ims] + img_t[ims] @ wi.T, dim=-1)
            logits = t @ v.T / temperature
            labels = torch.arange(len(ims))
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        r1 = _val_r1(snapshot(), img, txt, val_imgs_a, val_caps, cap_img)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_R@1": r1})
        if r1 > best_r1:
            best, best_r1, stale = snapshot(), r1, 0
            best.best_epoch = epoch
        else:
            stale += 1
            if stale >= patience:
                break
    best.history = history
    return best
