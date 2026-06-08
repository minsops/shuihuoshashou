from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from libs.common.config import Settings
from scripts.create_aliyun_nls_token import create_token_from_env
from services.asr_service.nls_engine import AliyunNLSSession

DEFAULT_PCM_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_16k_mono.pcm"


async def _run(pcm_path: Path, *, allow_empty_result: bool = False) -> int:
    app_key = os.environ.get("ALIYUN_NLS_APP_KEY", "").strip()
    token = os.environ.get("ALIYUN_NLS_TOKEN", "").strip()
    if not app_key:
        print("ALIYUN_NLS_APP_KEY is required.", file=sys.stderr)
        return 2
    if not token:
        created_token = create_token_from_env()
        if created_token is None:
            print(
                "ALIYUN_NLS_TOKEN is required, or set ALIYUN_AK_ID and ALIYUN_AK_SECRET "
                "to create one automatically.",
                file=sys.stderr,
            )
            return 2
        token = created_token.id
    if not pcm_path.exists():
        print(f"PCM file not found: {pcm_path}", file=sys.stderr)
        return 2

    settings = Settings(
        asr_provider="aliyun_nls_ws",
        aliyun_nls_app_key=app_key,
        aliyun_nls_token=token,
    )
    session = AliyunNLSSession("aliyun-nls-smoke", settings=settings)
    try:
        await session.connect()
        pcm = pcm_path.read_bytes()
        chunk_size = max(3200, settings.aliyun_nls_sample_rate // 10 * 2)
        for offset in range(0, len(pcm), chunk_size):
            await session.send_audio(pcm[offset : offset + chunk_size])
            await asyncio.sleep(0.1)
        await session.close()
    except Exception as exc:
        if session.started and not session.finished:
            try:
                await session.close()
            except Exception as close_exc:
                print(f"Aliyun NLS ASR cleanup failed: {close_exc}", file=sys.stderr)
        print(f"Aliyun NLS ASR smoke test failed: {exc}", file=sys.stderr)
        return 1

    texts: list[str] = []
    while not session.result_queue.empty():
        segment = await session.result_queue.get()
        if segment is not None and segment.text.strip():
            texts.append(segment.text.strip())
    if not texts and not allow_empty_result:
        print("Aliyun NLS ASR returned no transcript text.", file=sys.stderr)
        return 1
    if texts:
        print("\n".join(texts))
    print("Aliyun NLS ASR smoke test ok.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Aliyun NLS realtime ASR.")
    parser.add_argument(
        "--pcm-path",
        type=Path,
        default=DEFAULT_PCM_PATH,
        help=f"16kHz mono PCM16 audio file to send. Defaults to {DEFAULT_PCM_PATH}.",
    )
    parser.add_argument(
        "--allow-empty-result",
        action="store_true",
        help="Treat a completed ASR session with no transcript text as success.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.pcm_path, allow_empty_result=args.allow_empty_result))


if __name__ == "__main__":
    raise SystemExit(main())
