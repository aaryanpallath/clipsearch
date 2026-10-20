"""Run the whole evaluation on the test split and write results/.

uv run python -m clipsearch.evaluate --data data --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .adapter import train
from .categories import CATEGORIES, categorize
from .data import Dataset, coco_url, load
from .embed import MODEL, PRETRAINED, cached
from .keyword import BM25
from .metrics import bootstrap_ci, paired_difference, ranks, summary

HYBRID_WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]


def split_queries(ds: Dataset, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image indices of the split, the captions whose image is in it, and each caption's target position."""
    imgs = np.array(ds.split[name])
    pos = {g: i for i, g in enumerate(imgs)}
    caps = np.array([c for c, g in enumerate(ds.caption_image) if g in pos])
    target = np.array([pos[ds.caption_image[c]] for c in caps])
    return imgs, caps, target


def keyword_scores(ds: Dataset, imgs: np.ndarray, caps: np.ndarray) -> np.ndarray:
    bm = BM25([ds.tags[g] for g in imgs])
    return bm.score_matrix([ds.captions[c] for c in caps])


def hybrid(clip: np.ndarray, kw: np.ndarray, weight: float) -> np.ndarray:
    peak = kw.max(axis=1, keepdims=True)
    return clip + weight * np.divide(kw, peak, out=np.zeros_like(kw), where=peak > 0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    ds = load(a.data)
    img, txt = cached(a.data, ds.files, ds.captions)
    cap_img = np.array(ds.caption_image)

    # tune the hybrid weight and train the adapter without touching the test split
    v_imgs, v_caps, v_target = split_queries(ds, "val")
    v_clip = txt[v_caps] @ img[v_imgs].T
    v_kw = keyword_scores(ds, v_imgs, v_caps)
    tuning = {w: float((ranks(hybrid(v_clip, v_kw, w), v_target) == 1).mean()) for w in HYBRID_WEIGHTS}
    best_w = max(tuning, key=lambda w: tuning[w])
    adapter = train(img, txt, cap_img, ds.split["train"], ds.split["val"])

    t_imgs, t_caps, target = split_queries(ds, "test")
    groups = target  # cluster = the target image
    clip = txt[t_caps] @ img[t_imgs].T
    kw = keyword_scores(ds, t_imgs, t_caps)
    systems = {
        "keyword (BM25 on object tags)": kw,
        "CLIP zero-shot": clip,
        f"hybrid (CLIP + {best_w:g} x keyword)": hybrid(clip, kw, best_w),
        "CLIP + trained adapter": adapter.text(txt[t_caps]) @ adapter.image(img[t_imgs]).T,
    }
    rk = {name: ranks(s, target) for name, s in systems.items()}

    report: dict = {
        "model": f"{MODEL} ({PRETRAINED})",
        "test": {"images": len(t_imgs), "queries": len(t_caps)},
        "hybrid_tuning_val_R@1": {str(k): v for k, v in tuning.items()},
        "adapter": {"best_epoch": adapter.best_epoch, "history": adapter.history},
        "systems": {},
        "vs_clip_R@1": {},
    }
    for name, r in rk.items():
        s = summary(r)
        cis = {}
        for k in (1, 5, 10):
            mean, lo, hi = bootstrap_ci((r <= k).astype(float), groups)
            cis[f"R@{k}"] = [round(mean, 4), round(lo, 4), round(hi, 4)]
        report["systems"][name] = {**{k: round(v, 4) for k, v in s.items()}, "ci95": cis}
    base = (rk["CLIP zero-shot"] == 1).astype(float)
    for name, r in rk.items():
        if name != "CLIP zero-shot":
            d = paired_difference(base, (r == 1).astype(float), groups)
            report["vs_clip_R@1"][name] = {k: round(v, 4) for k, v in d.items()}

    # where does each system break? recall at 1 by kind of query
    cats = [categorize(ds.captions[c]) for c in t_caps]
    by_cat: dict = {}
    for cat in [*CATEGORIES, "none of these"]:
        mask = np.array([(cat in cs) if cat != "none of these" else not cs for cs in cats])
        if mask.sum() < 20:
            continue
        by_cat[cat] = {"queries": int(mask.sum())}
        for name in ("keyword (BM25 on object tags)", "CLIP zero-shot", "CLIP + trained adapter"):
            mean, lo, hi = bootstrap_ci((rk[name][mask] == 1).astype(float), groups[mask])
            by_cat[cat][name] = [round(mean, 4), round(lo, 4), round(hi, 4)]
        d = paired_difference(
            (rk["CLIP zero-shot"][mask] == 1).astype(float),
            (rk["CLIP + trained adapter"][mask] == 1).astype(float),
            groups[mask],
        )
        by_cat[cat]["adapter_minus_clip"] = {k: round(v, 4) for k, v in d.items()}
    report["by_category_R@1"] = by_cat
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    # failure examples: queries of the hard kinds where CLIP's top result is another image
    lines = [
        "# Where zero-shot CLIP misses",
        "",
        "Test queries whose correct image was not CLIP's first result. Links open the COCO images.",
        "",
    ]
    top = clip.argmax(axis=1)
    rng = np.random.default_rng(1)
    for cat in ("counting", "spatial", "negation", "text_in_image"):
        idx = [i for i, cs in enumerate(cats) if cat in cs and top[i] != target[i]]
        if not idx:
            continue
        lines += [f"## {cat.replace('_', ' ')}", ""]
        for i in rng.choice(idx, size=min(4, len(idx)), replace=False):
            got = t_imgs[top[i]]
            got_caption = next(ds.captions[c] for c, g in enumerate(ds.caption_image) if g == got)
            lines.append(
                f'- **"{ds.captions[t_caps[i]]}"** (correct image ranked {rk["CLIP zero-shot"][i]}): '
                f"[wanted]({coco_url(ds.files[t_imgs[target[i]]])}), "
                f'[CLIP\'s first result]({coco_url(ds.files[got])}), described as "{got_caption}"'
            )
        lines.append("")
    (out / "failures.md").write_text("\n".join(lines))
    _plots(report, out)
    print(json.dumps({k: {m: v[m] for m in ("R@1", "R@5", "R@10")} for k, v in report["systems"].items()}, indent=1))
    print(json.dumps(report["vs_clip_R@1"], indent=1))


def _plots(report: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    names = list(report["systems"])
    for j, k in enumerate(("R@1", "R@5", "R@10")):
        vals = [report["systems"][n]["ci95"][k] for n in names]
        x = np.arange(len(names)) + (j - 1) * 0.27
        ax[0].bar(
            x,
            [v[0] for v in vals],
            0.27,
            label=k,
            yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
            capsize=3,
        )
    ax[0].set_xticks(np.arange(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax[0].set(
        title=f"Text-to-image recall, {report['test']['queries']} queries over "
        f"{report['test']['images']} test images (95% CI)",
        ylim=(0, 1),
    )
    ax[0].legend()
    cats = list(report["by_category_R@1"])
    for j, n in enumerate(("keyword (BM25 on object tags)", "CLIP zero-shot", "CLIP + trained adapter")):
        vals = [report["by_category_R@1"][c][n] for c in cats]
        x = np.arange(len(cats)) + (j - 1) * 0.27
        ax[1].bar(
            x,
            [v[0] for v in vals],
            0.27,
            label=n,
            yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
            capsize=3,
        )
    labels = [f"{c.replace('_', ' ')}\n(n={report['by_category_R@1'][c]['queries']})" for c in cats]
    ax[1].set_xticks(np.arange(len(cats)), labels, fontsize=8)
    ax[1].set(title="Recall@1 by kind of query", ylim=(0, 0.8))
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "recall.png", dpi=130)


if __name__ == "__main__":
    main()
