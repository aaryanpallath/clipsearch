import json

import numpy as np
import pytest

from clipsearch.adapter import train
from clipsearch.categories import categorize
from clipsearch.data import load
from clipsearch.evaluate import hybrid
from clipsearch.keyword import BM25, tokens
from clipsearch.metrics import bootstrap_ci, paired_difference, ranks, recall_at, summary


def test_tokens_fold_plurals_and_drop_stopwords():
    assert tokens("Two dogs and the buses near benches") == ["two", "dog", "bus", "near", "bench"]
    assert tokens("glass") == ["glass"]
    assert tokens("sandwiches dishes boxes") == ["sandwich", "dish", "box"]


def test_bm25_prefers_documents_with_rarer_matching_tags():
    bm = BM25([["dog", "person"], ["person", "person", "car"], ["cat"]])
    s = bm.scores("a dog with a person")
    assert s.argmax() == 0
    assert s[2] == 0
    assert bm.score_matrix([]).shape == (0, 3)


def test_ranks_are_pessimistic_about_ties():
    scores = np.array([[0.9, 0.1, 0.5], [0.3, 0.3, 0.3]])
    r = ranks(scores, np.array([0, 1]))
    assert r.tolist() == [1, 3]
    assert recall_at(r, 1) == 0.5
    assert summary(r)["MRR"] == pytest.approx((1 + 1 / 3) / 2)


def test_cluster_bootstrap_ci_contains_the_mean_and_widens_with_correlated_queries():
    rng = np.random.default_rng(0)
    per_image = rng.random(200) < 0.5
    values = np.repeat(per_image, 5).astype(float)  # five identical captions per image
    groups = np.repeat(np.arange(200), 5)
    mean, lo, hi = bootstrap_ci(values, groups)
    assert lo < mean < hi
    _, lo_naive, hi_naive = bootstrap_ci(values, np.arange(1000))  # pretending queries are independent
    assert (hi - lo) > 1.8 * (hi_naive - lo_naive)


def test_paired_difference_of_identical_systems_is_zero():
    a = (np.arange(100) % 3 == 0).astype(float)
    d = paired_difference(a, a, np.arange(100) // 5)
    assert d == {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_categories():
    assert set(categorize("Two dogs sitting next to a red car")) == {"counting", "spatial", "color", "action"}
    assert categorize("A street sign that reads stop") == ["text_in_image"]
    assert categorize("An empty room with no people") == ["negation"]
    assert categorize("A cat") == []


def test_hybrid_normalizes_keyword_scores_per_query():
    clip = np.array([[0.30, 0.31], [0.2, 0.2]])
    kw = np.array([[4.0, 0.0], [0.0, 0.0]])
    h = hybrid(clip, kw, 0.02)
    assert h[0].tolist() == pytest.approx([0.32, 0.31])
    assert h[1].tolist() == [0.2, 0.2]  # no keyword match: CLIP order unchanged


def test_adapter_starts_at_clip_and_learns_a_systematic_distortion():
    # synthetic "CLIP": text embeddings are a rotated, noisy copy of image embeddings
    rng = np.random.default_rng(0)
    n, d = 600, 32
    img = rng.normal(size=(n, d))
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    mix = 0.6 * np.eye(d) + 0.4 * q
    owner = np.repeat(np.arange(n), 2)
    txt = img[owner] @ mix.T + 0.15 * rng.normal(size=(2 * n, d))
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    res = train(
        img.astype(np.float32),
        txt.astype(np.float32),
        owner,
        list(range(400)),
        list(range(400, 600)),
        epochs=30,
        temperature=0.05,
    )
    assert res.history[0]["epoch"] == 0  # epoch 0 is plain CLIP: the adapter starts as the identity
    assert max(h["val_R@1"] for h in res.history) > res.history[0]["val_R@1"] + 0.1
    assert res.best_epoch > 0


def test_split_is_disjoint_and_keeps_captions_with_their_image(tmp_path):
    (tmp_path / "annotations").mkdir()
    images = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(10)]
    caps = [{"id": 100 + i, "image_id": i // 2, "caption": f"caption {i}"} for i in range(20)]
    inst = {"categories": [{"id": 1, "name": "dog"}], "annotations": [{"image_id": 3, "category_id": 1}]}
    (tmp_path / "annotations" / "captions_val2017.json").write_text(json.dumps({"images": images, "annotations": caps}))
    (tmp_path / "annotations" / "instances_val2017.json").write_text(json.dumps(inst))
    ds = load(tmp_path, n_train=6, n_val=2)
    sets = [set(v) for v in ds.split.values()]
    assert sum(len(s) for s in sets) == 10 and set.union(*sets) == set(range(10))
    assert ds.tags[3] == ["dog"]
    assert all(ds.captions[c] == f"caption {c}" for c in range(20))
    assert ds.caption_image[7] == 3
