from __future__ import annotations

import math
import re
from collections import Counter


def normalize_text(text: str) -> str:
    return "".join(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()))


def char_ngrams(text: str, size: int = 2) -> Counter[str]:
    normalized = normalize_text(text)
    if not normalized:
        return Counter()
    if len(normalized) <= size:
        return Counter([normalized])
    return Counter(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def cosine_similarity(a: str, b: str) -> float:
    left = char_ngrams(a)
    right = char_ngrams(b)
    if not left or not right:
        return 0.0
    overlap = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return overlap / (left_norm * right_norm)
