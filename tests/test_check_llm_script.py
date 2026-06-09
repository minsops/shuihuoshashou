from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_check_llm():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_llm.py"
    spec = importlib.util.spec_from_file_location("check_llm", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_llm_reports_mock_mode_without_claiming_real_smoke(monkeypatch, capsys) -> None:
    check_llm = _load_check_llm()

    class FakeClient:
        async def complete_json(self, messages, schema, fallback, *, raise_on_error=False):
            assert raise_on_error is False
            return fallback

    monkeypatch.setattr(
        "libs.common.runtime.get_runtime_status",
        lambda: SimpleNamespace(
            llm_provider="mock",
            model_dump_json=lambda indent=None: '{"llm_provider":"mock"}',
        ),
    )
    monkeypatch.setattr("libs.llm_client.get_llm_client", lambda: FakeClient())

    result = __import__("asyncio").run(check_llm.main())

    captured = capsys.readouterr()
    assert result == 0
    assert "LLM mock mode ok. No real model endpoint was called." in captured.out
    assert "LLM smoke test ok." not in captured.out
