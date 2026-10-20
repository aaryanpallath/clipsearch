"""The keyword baseline: BM25 over each image's object tags, as a library searching curated keywords would."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOP = {
    *[
        "a",
        "an",
        "the",
        "of",
        "on",
        "in",
        "at",
        "to",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "for",
        "from",
        "it",
        "its",
        "this",
        "that",
        "there",
        "their",
    ],
    *["his", "her", "some", "very", "while", "has", "have"],
}


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP:
            continue
        # light plural folding so "dogs" matches the tag "dog"
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and (w.endswith(("ches", "shes")) or (w.endswith("es") and w[-3] in "sxz")):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs = [Counter(tokens(" ".join(d))) for d in docs]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lengths.mean() if len(self.docs) else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(self.docs):
            for t, f in d.items():
                self.postings.setdefault(t, []).append((i, f))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for t in set(tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                s[i] += idf * f * (self.k1 + 1) / (f + norm)
        return s

    def score_matrix(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.scores(q) for q in queries]) if queries else np.zeros((0, len(self.docs)))
