from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.asr_service.nls_token import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_REGION,
    AliyunNLSToken,
    create_token,
    create_token_from_env,
)

__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_REGION",
    "AliyunNLSToken",
    "create_token",
    "create_token_from_env",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an Aliyun NLS Access Token.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ALIYUN_NLS_TOKEN_ENDPOINT", DEFAULT_ENDPOINT),
    )
    parser.add_argument("--region", default=os.environ.get("ALIYUN_NLS_TOKEN_REGION", DEFAULT_REGION))
    args = parser.parse_args()

    access_key_id = os.environ.get("ALIYUN_AK_ID", "").strip()
    access_key_secret = os.environ.get("ALIYUN_AK_SECRET", "").strip()
    if not access_key_id or not access_key_secret:
        print("ALIYUN_AK_ID and ALIYUN_AK_SECRET are required.", file=sys.stderr)
        return 2
    try:
        token = create_token(
            access_key_id,
            access_key_secret,
            endpoint=args.endpoint,
            region_id=args.region,
        )
    except Exception as exc:
        print(f"Aliyun NLS token creation failed: {exc}", file=sys.stderr)
        return 1
    print(f"ALIYUN_NLS_TOKEN={token.id}")
    if token.expire_time is not None:
        print(f"ALIYUN_NLS_TOKEN_EXPIRE_TIME={token.expire_time}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
