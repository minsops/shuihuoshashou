from __future__ import annotations

import importlib.util
import socket
from pathlib import Path


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
    assert run_gateway.display_url("127.0.0.1", 8001) == "http://127.0.0.1:8001/"
    assert run_gateway.display_url("0.0.0.0", 8001) == "http://127.0.0.1:8001/"


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
