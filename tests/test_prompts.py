from __future__ import annotations

import pytest

from libs.common.prompts import load_prompt


def test_load_prompt_reads_prompt_file() -> None:
    prompt = load_prompt("probe_system.md")

    assert prompt.startswith("你是资深面试官教练")
    assert "严格 JSON" in prompt
    assert "resume_claims" in prompt
    assert "简历对质型追问" in prompt
    assert prompt == prompt.strip()


def test_load_prompt_rejects_path_separators() -> None:
    with pytest.raises(ValueError):
        load_prompt("../probe_system.md")


def test_load_prompt_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("missing.md")
