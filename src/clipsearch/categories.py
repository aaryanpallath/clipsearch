"""Tag queries by the kind of language they use, to see where each system breaks.

Rules are deliberately simple and transparent (regular expressions); a query can fall in several categories."""

from __future__ import annotations

import re

NUMBERS = r"\b(two|three|four|five|six|seven|eight|nine|ten|several|pair of|couple of|\d+)\b"
CATEGORIES: dict[str, str] = {
    "counting": NUMBERS,
    "spatial": r"\b(left|right|behind|beneath|under|underneath|above|below|on top of|next to|in front of|between|"
    r"beside|near|across from|inside)\b",
    "negation": r"\b(no|not|without|empty|nobody|none)\b",
    "color": r"\b(red|blue|green|yellow|white|black|brown|orange|pink|purple|gray|grey|silver)\b",
    "text_in_image": r"\b(sign|signs|reads|says|written|words|letters|logo|banner)\b",
    "action": r"\b\w+ing\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in CATEGORIES.items()}


def categorize(caption: str) -> list[str]:
    return [k for k, rx in _COMPILED.items() if rx.search(caption)]
