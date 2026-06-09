from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from libs.common.config import Settings
from services.asr_service.aliyun_engine import AliyunASRSession

DEFAULT_PCM_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_16k_mono.pcm"


async def _run(pcm_path: Path, *, allow_empty_result: bool = False) -> int:
    api_key = os.environ.get("ALIYUN_ASR_API_KEY", "").strip()
    if not api_key:
        print("ALIYUN_ASR_API_KEY is required.", file=sys.stderr)
        return 2
    if not pcm_path.exists():
        print(f"PCM file not found: {pcm_path}", file=sys.stderr)
        return 2

    settings = Settings(asr_provider="aliyun_ws", aliyun_asr_api_key=api_key)
    session = AliyunASRSession("aliyun-smoke", settings=settings)
    try:
        await session.connect()
        pcm = pcm_path.read_bytes()
        chunk_size = max(3200, settings.aliyun_asr_sample_rate // 10 * 2)
        for offset in range(0, len(pcm), chunk_size):
            await session.send_audio(pcm[offset : offset + chunk_size])
            await asyncio.sleep(0.1)
        await session.close()
    except Exception as exc:
        if session.started and not session.finished:
            try:
                await session.close()
            except Exception as close_exc:
                print(f"Aliyun ASR cleanup failed: {close_exc}", file=sys.stderr)
        print(f"Aliyun ASR smoke test failed: {exc}", file=sys.stderr)
        return 1

    texts: list[str] = []
    while not session.result_queue.empty():
        segment = await session.result_queue.get()
        if segment is not None and segment.text.strip():
            texts.append(segment.text.strip())
    if not texts and not allow_empty_result:
        print("Aliyun ASR returned no transcript text.", file=sys.stderr)
        return 1
    if texts:
        print("\n".join(texts))
        print("Aliyun ASR smoke test ok.")
    else:
        print("Aliyun ASR session completed, but no transcript text was verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Aliyun Paraformer realtime ASR.")
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
