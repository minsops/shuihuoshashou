from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local offline interview demo.")
    return parser.parse_args(argv)


def main() -> None:
    parse_args()
    from libs.common.database import init_db
    from libs.schemas import CandidateCreate, InterviewCreate, JobCreate, QATurn
    from services.interview_orchestrator.service import (
        add_turn,
        create_candidate,
        create_interview,
        end_interview,
    )
    from services.jd_kb_service.service import create_job

    init_db()
    job = create_job(
        JobCreate(
            title="AI 后端工程师",
            jd_text="负责 Python/FastAPI 微服务、LLM 应用、评估体系和可靠性工程。",
        )
    )
    candidate = create_candidate(CandidateCreate(name="Demo Candidate"))
    interview = create_interview(InterviewCreate(job_id=job.id, candidate_id=candidate.id))
    add_turn(
        interview.id,
        QATurn(
            question="介绍一个你最核心的 AI 项目。",
            answer="我主要负责整体架构设计并推动项目落地，优化后效果有显著提升。",
            answer_start_ms=0,
            answer_end_ms=32000,
        ),
    )
    add_turn(
        interview.id,
        QATurn(
            question="你具体写了哪部分？",
            answer="我写了 FastAPI 的调用编排、重试和 JSON 解析，因为线上模型偶发返回格式错误。",
            answer_start_ms=33000,
            answer_end_ms=61000,
        ),
    )
    report = end_interview(interview.id)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
