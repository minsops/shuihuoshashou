from __future__ import annotations

from libs.common.config import get_settings
from libs.schemas import BehaviorSignal, QATurn


def extract_behavior_signal(turn: QATurn) -> BehaviorSignal | None:
    if not get_settings().signal_enabled:
        return None
    word_count = max(1, len(turn.answer))
    hesitation = min(1.0, turn.answer.count("嗯") * 0.15 + turn.answer.count("这个") * 0.1)
    fluency = max(0.0, min(1.0, 1.0 - hesitation + (0.1 if word_count > 80 else -0.1)))
    evasive = any(marker in turn.answer for marker in ["不太清楚", "差不多", "别人负责", "记不清"])
    return BehaviorSignal(
        turn_id=turn.turn_id,
        fluency=round(fluency, 3),
        hesitation=round(hesitation, 3),
        evasiveness_hint=evasive,
    )

