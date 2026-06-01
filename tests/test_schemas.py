from __future__ import annotations

import pytest
from pydantic import ValidationError

from libs.schemas import BehaviorSignal


def test_behavior_signal_forbids_compliance_sensitive_extra_fields() -> None:
    signal = BehaviorSignal.model_validate(
        {
            "turn_id": "turn-1",
            "fluency": 0.8,
            "hesitation": 0.2,
            "evasiveness_hint": False,
        }
    )

    assert signal.model_dump() == {
        "turn_id": "turn-1",
        "fluency": 0.8,
        "hesitation": 0.2,
        "evasiveness_hint": False,
    }

    with pytest.raises(ValidationError):
        BehaviorSignal.model_validate(
            {
                "turn_id": "turn-1",
                "fluency": 0.8,
                "hesitation": 0.2,
                "evasiveness_hint": False,
                "personality": "confident",
                "emotion": "nervous",
                "reliability": 0.4,
            }
        )
