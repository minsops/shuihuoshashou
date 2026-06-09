from __future__ import annotations

import importlib.util
from io import BytesIO
import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_run_gateway():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_gateway.py"
    spec = importlib.util.spec_from_file_location("run_gateway", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gateway_script_defaults_to_local_demo_port() -> None:
    run_gateway = _load_run_gateway()
    args = run_gateway.parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8001
    assert args.no_reload is False
    assert args.no_default_port_redirect is False
    assert run_gateway.display_url("127.0.0.1", 8001) == "http://127.0.0.1:8001/"
    assert run_gateway.display_url("0.0.0.0", 8001) == "http://127.0.0.1:8001/"


def test_gateway_script_accepts_redirect_disable_flag() -> None:
    run_gateway = _load_run_gateway()
    args = run_gateway.parse_args(["--no-default-port-redirect"])

    assert args.no_default_port_redirect is True


def test_gateway_script_reports_occupied_port(capsys) -> None:
    run_gateway = _load_run_gateway()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]

        result = run_gateway.main(["--port", str(port)])

    captured = capsys.readouterr()
    assert result == 2
    assert f"Port {port} is already in use" in captured.err
    assert "python scripts/run_gateway.py --port 8002" in captured.err


def test_gateway_script_warns_when_docker_port_is_occupied(monkeypatch) -> None:
    run_gateway = _load_run_gateway()

    def fake_port_is_available(host: str, port: int) -> bool:
        assert host == "127.0.0.1"
        return port != 8000

    monkeypatch.setattr(run_gateway, "port_is_available", fake_port_is_available)

    lines = run_gateway.local_port_warning_lines("127.0.0.1", 8001)

    assert lines == [
        "Note: port 8000 is already in use on 127.0.0.1.",
        "Open http://127.0.0.1:8001/ for this local gateway, not http://127.0.0.1:8000/.",
    ]


def test_gateway_script_redirects_default_port_to_selected_port() -> None:
    run_gateway = _load_run_gateway()
    redirect = run_gateway.start_port_redirect("127.0.0.1", 0, 8123)
    assert redirect is not None
    server, redirect_url = redirect
    source_port = server.server_address[1]

    assert redirect_url == f"http://127.0.0.1:{source_port}/"
    server.shutdown()
    server.server_close()


def test_gateway_script_redirect_handler_preserves_path_and_query() -> None:
    run_gateway = _load_run_gateway()
    handler_cls = run_gateway._redirect_handler("http://127.0.0.1:8123/")
    handler = handler_cls.__new__(handler_cls)
    headers: list[tuple[str, str]] = []
    handler.path = "/interview?demo=1"
    handler.wfile = BytesIO()
    handler.send_response = lambda status: headers.append(("status", str(status)))
    handler.send_header = lambda name, value: headers.append((name, value))
    handler.end_headers = lambda: None

    handler._redirect()

    assert ("status", "307") in headers
    assert ("Location", "http://127.0.0.1:8123/interview?demo=1") in headers
    assert b"Redirecting to http://127.0.0.1:8123/interview?demo=1" in handler.wfile.getvalue()


def test_gateway_script_runtime_summary_hides_secrets() -> None:
    run_gateway = _load_run_gateway()
    status = SimpleNamespace(
        llm_provider="openai_compatible",
        llm_model="deepseek-v4-pro",
        llm_base_url_configured=True,
        llm_api_key_configured=True,
        asr_provider="aliyun_nls_ws",
        aliyun_nls_app_key_configured=True,
        aliyun_nls_token_configured=True,
        aliyun_nls_token_auto_configured=False,
        aliyun_nls_endpoint_configured=True,
        aliyun_asr_api_key_configured=False,
        aliyun_asr_endpoint_configured=False,
        asr_base_url_configured=False,
        database_url="sqlite:///data/shuihuo_killer.db",
    )

    lines = run_gateway.runtime_summary_lines(status)
    summary = "\n".join(lines)

    assert "Model: openai_compatible / deepseek-v4-pro (configured)" in summary
    assert "ASR: aliyun_nls_ws (configured)" in summary
    assert "Database: sqlite:///data/shuihuo_killer.db" in summary
    assert "secret" not in summary.lower()
    assert "token" not in summary.lower()
    assert "key" not in summary.lower()


def test_gateway_script_runtime_summary_marks_mock_modes() -> None:
    run_gateway = _load_run_gateway()
    status = SimpleNamespace(
        llm_provider="mock",
        llm_model="mimo-v2.5-pro",
        llm_base_url_configured=True,
        llm_api_key_configured=False,
        asr_provider="stub",
        aliyun_nls_app_key_configured=False,
        aliyun_nls_token_configured=False,
        aliyun_nls_token_auto_configured=False,
        aliyun_nls_endpoint_configured=True,
        aliyun_asr_api_key_configured=False,
        aliyun_asr_endpoint_configured=True,
        asr_base_url_configured=False,
        database_url="sqlite:///data/shuihuo_killer.db",
    )

    lines = run_gateway.runtime_summary_lines(status)
    summary = "\n".join(lines)

    assert "Model: mock / mimo-v2.5-pro (mock)" in summary
    assert "ASR: stub (mock)" in summary
    assert "Model: mock / mimo-v2.5-pro (configured)" not in summary
    assert "ASR: stub (configured)" not in summary


def test_gateway_script_runtime_summary_accepts_nls_auto_token() -> None:
    run_gateway = _load_run_gateway()
    status = SimpleNamespace(
        llm_provider="mock",
        llm_model="mimo-v2.5-pro",
        llm_base_url_configured=True,
        llm_api_key_configured=False,
        asr_provider="aliyun_nls_ws",
        aliyun_nls_app_key_configured=True,
        aliyun_nls_token_configured=False,
        aliyun_nls_token_auto_configured=True,
        aliyun_nls_endpoint_configured=True,
        aliyun_asr_api_key_configured=False,
        aliyun_asr_endpoint_configured=False,
        asr_base_url_configured=False,
        database_url="sqlite:///data/shuihuo_killer.db",
    )

    assert "ASR: aliyun_nls_ws (configured)" in "\n".join(
        run_gateway.runtime_summary_lines(status)
    )


def test_gateway_script_reports_runtime_config_error(monkeypatch, capsys) -> None:
    run_gateway = _load_run_gateway()
    runtime_module = ModuleType("libs.common.runtime")

    def fail_runtime_status():
        raise ValueError("ASR_PROVIDER=aliyun_nls_ws requires ALIYUN_NLS_TOKEN")

    runtime_module.get_runtime_status = fail_runtime_status
    monkeypatch.setitem(sys.modules, "libs.common.runtime", runtime_module)

    result = run_gateway.main(["--port", "0"])

    captured = capsys.readouterr()
    assert result == 3
    assert "Runtime configuration error" in captured.err
    assert "ALIYUN_NLS_TOKEN" in captured.err
