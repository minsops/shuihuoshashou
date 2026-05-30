from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    if "/" in name or "\\" in name:
        raise ValueError("prompt name must not include path separators")
    path = PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {name}")
    return path.read_text(encoding="utf-8").strip()
