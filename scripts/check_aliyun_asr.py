from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from libs.common.config import Settings
from services.asr_service.aliyun_engine import AliyunASRSession


async def _run(pcm_path: Path) -> int:
    api_key = os.environ.get("ALIYUN_ASR_API_KEY", "").strip()
    if not api_key:
        print("ALIYUN_ASR_API_KEY is required.", file=sys.stderr)
        return 2
    if not pcm_path.exists():
        print(f"PCM file not found: {pcm_path}", file=sys.stderr)
        return 2

    settings = Settings(asr_provider="aliyun_ws", aliyun_asr_api_key=api_key)
    session = AliyunASRSession("aliyun-smoke", settings=settings)
    await session.connect()
    pcm = pcm_path.read_bytes()
    chunk_size = max(3200, settings.aliyun_asr_sample_rate // 10 * 2)
    for offset in range(0, len(pcm), chunk_size):
        await session.send_audio(pcm[offset : offset + chunk_size])
        await asyncio.sleep(0.1)
    await session.close()

    texts: list[str] = []
    while not session.result_queue.empty():
        segment = await session.result_queue.get()
        if segment is not None:
            texts.append(segment.text)
    if texts:
        print("\n".join(texts))
    print("Aliyun ASR smoke test ok.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Aliyun Paraformer realtime ASR.")
    parser.add_argument(
        "--pcm-path",
        type=Path,
        required=True,
        help="16kHz mono PCM16 audio file to send.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.pcm_path))


if __name__ == "__main__":
    raise SystemExit(main())
