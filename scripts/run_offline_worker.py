from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume offline scoring tasks from Redis Streams.")
    parser.add_argument("--once", action="store_true", help="Process one poll cycle and exit.")
    parser.add_argument("--block-ms", type=int, default=1000)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--group", default="offline-workers")
    parser.add_argument("--consumer", default="worker-1")
    args = parser.parse_args()

    from libs.common.config import get_settings
    from libs.common.tasks import RedisStreamWorker
    from services.interview_orchestrator.service import run_offline_scoring_task

    settings = get_settings()
    worker = RedisStreamWorker(
        "interview.offline_scoring",
        lambda payload: run_offline_scoring_task(str(payload["interview_id"])),
        redis_url=settings.redis_url,
        stream_prefix=settings.redis_stream_prefix,
        group=args.group,
        consumer=args.consumer,
    )
    while True:
        processed = worker.consume_once(block_ms=args.block_ms)
        if args.once:
            print(f"processed={processed}")
            return
        if processed == 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
