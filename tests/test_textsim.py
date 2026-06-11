from __future__ import annotations

from libs.common.textsim import cosine_similarity


def test_cosine_similarity_scores_identical_text_highest() -> None:
    text = "我主要负责整体架构设计并推动项目落地最终取得显著提升"

    assert cosine_similarity(text, text) == 1.0
    assert cosine_similarity(text, "完全不同的问题") < 0.2
