from __future__ import annotations

import re

from libs.schemas import ConsistencyFlag, FactClaim, QATurn

TECH_PATTERNS = [
    "Python",
    "FastAPI",
    "Redis",
    "PostgreSQL",
    "SQLite",
    "LLM",
    "RAG",
    "ASR",
    "WebSocket",
]
RESPONSIBILITY_PATTERNS = ["架构", "编排", "重试", "校验", "优化", "重构", "上线", "排查", "评估"]


def extract_fact_claim(turn: QATurn) -> FactClaim:
    answer = turn.answer
    if any(word in answer for word in ["独立完成", "独立负责", "我一个人"]):
        scope = "solo"
    elif any(word in answer for word in ["主导", "牵头", "owner"]):
        scope = "lead"
    elif any(word in answer for word in ["参与", "配合", "协助"]):
        scope = "participant"
    elif any(word in answer for word in ["团队主导", "团队负责", "别人负责", "同事负责"]):
        scope = "team"
    else:
        scope = "unknown"

    technologies = [tech for tech in TECH_PATTERNS if tech.lower() in answer.lower()]
    responsibilities = [item for item in RESPONSIBILITY_PATTERNS if item in answer]
    metrics = re.findall(r"(?:提升|降低|减少|增加)?\s*\d+(?:\.\d+)?\s*(?:%|ms|秒|分钟|倍|qps|QPS)", answer)
    return FactClaim(
        turn_id=turn.turn_id,
        contribution_scope=scope,
        responsibilities=responsibilities,
        technologies=technologies,
        metrics=[metric.strip() for metric in metrics],
    )


def detect_consistency(turns: list[QATurn]) -> list[ConsistencyFlag]:
    claims = [extract_fact_claim(turn) for turn in turns]
    flags: list[ConsistencyFlag] = []
    for left_index, left in enumerate(claims):
        for right in claims[left_index + 1 :]:
            if {left.contribution_scope, right.contribution_scope} == {"solo", "team"}:
                flags.append(
                    ConsistencyFlag(
                        turn_id_a=left.turn_id,
                        turn_id_b=right.turn_id,
                        description="候选人对个人贡献边界的描述冲突：一处称独立完成，另一处称团队或他人负责。",
                        severity="high",
                    )
                )
            if (
                left.contribution_scope == "lead"
                and right.contribution_scope == "participant"
                or left.contribution_scope == "participant"
                and right.contribution_scope == "lead"
            ):
                flags.append(
                    ConsistencyFlag(
                        turn_id_a=left.turn_id,
                        turn_id_b=right.turn_id,
                        description="候选人对项目角色的描述存在漂移：主导者与参与者边界需要追问确认。",
                        severity="low",
                    )
                )
    return flags
