from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scripts_with_side_effects_support_help_without_running() -> None:
    for script_name in [
        "check_llm.py",
        "diagnose_llm_auth.py",
        "diagnose_llm_network.py",
        "run_offline_demo.py",
    ]:
        script = ROOT / "scripts" / script_name
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "usage:" in result.stdout
        assert "Traceback" not in result.stderr
