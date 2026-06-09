from __future__ import annotations

import argparse
import socket
import ssl
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose DNS, TCP, and TLS for LLM_BASE_URL.")
    return parser.parse_args(argv)


def main() -> int:
    parse_args()
    from libs.common.config import get_settings

    settings = get_settings()
    parsed = urlparse(settings.llm_base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        print("LLM_BASE_URL is empty or invalid.")
        return 1

    print(f"base_url: {settings.llm_base_url}")
    print(f"host: {host}")
    print(f"port: {port}")

    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        print(f"dns: failed ({exc})")
        _print_public_dns(host)
        return 2

    unique_addresses = sorted({item[4][0] for item in addresses})
    print(f"dns: ok ({', '.join(unique_addresses)})")
    _print_public_dns(host)

    try:
        with socket.create_connection((host, port), timeout=10) as sock:
            print("tcp: ok")
            if parsed.scheme == "https":
                try:
                    import certifi

                    context = ssl.create_default_context(cafile=certifi.where())
                except Exception:
                    context = ssl.create_default_context()
                with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                    cert = tls_sock.getpeercert()
                    subject = cert.get("subject", ())
                    common_names = [
                        value
                        for group in subject
                        for key, value in group
                        if key == "commonName"
                    ]
                    print(f"tls: ok ({', '.join(common_names) or 'certificate received'})")
    except OSError as exc:
        print(f"connect: failed ({type(exc).__name__}: {exc})")
        return 3
    except ssl.SSLError as exc:
        print(f"tls: failed ({exc})")
        return 4

    return 0


def _print_public_dns(host: str) -> None:
    for server in ["8.8.8.8", "1.1.1.1", "223.5.5.5"]:
        try:
            result = subprocess.run(
                ["dig", f"@{server}", "+short", host],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            print(f"dns @{server}: check failed ({type(exc).__name__}: {exc})")
            continue
        output = result.stdout.strip()
        print(f"dns @{server}: {output or 'no records'}")


if __name__ == "__main__":
    raise SystemExit(main())
