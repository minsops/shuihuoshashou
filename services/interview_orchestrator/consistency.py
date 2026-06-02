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
METRIC_VERB_DIRECTIONS = {
    "提升": "up",
    "增加": "up",
    "降低": "down",
    "减少": "down",
}
METRIC_PATTERN = re.compile(
    r"(?P<verb>提升|降低|减少|增加)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|ms|秒|分钟|倍|qps|QPS)"
)


def extract_fact_claim(turn: QATurn) -> FactClaim:
    answer = turn.answer
    if any(word in answer for word in ["团队主导", "团队负责", "别人负责", "同事负责"]):
        scope = "team"
    elif any(word in answer for word in ["独立完成", "独立负责", "我一个人"]):
        scope = "solo"
    elif any(word in answer for word in ["主导", "牵头", "owner"]):
        scope = "lead"
    elif any(word in answer for word in ["参与", "配合", "协助"]):
        scope = "participant"
    else:
        scope = "unknown"

    technologies = [tech for tech in TECH_PATTERNS if tech.lower() in answer.lower()]
    responsibilities = [item for item in RESPONSIBILITY_PATTERNS if item in answer]
    metrics = [match.group(0) for match in METRIC_PATTERN.finditer(answer)]
    return FactClaim(
        turn_id=turn.turn_id,
        contribution_scope=scope,
        responsibilities=responsibilities,
        technologies=technologies,
        metrics=[metric.strip() for metric in metrics],
    )


def detect_consistency(turns: list[QATurn]) -> list[ConsistencyFlag]:
    claims = [extract_fact_claim(turn) for turn in turns]
    return detect_claim_conflicts(claims)


def detect_claim_conflicts(claims: list[FactClaim]) -> list[ConsistencyFlag]:
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
            metric_conflict = _metric_conflict(left, right)
            if metric_conflict is not None:
                left_metric, right_metric = metric_conflict
                flags.append(
                    ConsistencyFlag(
                        turn_id_a=left.turn_id,
                        turn_id_b=right.turn_id,
                        description=(
                            "候选人对同一职责或技术的成果数字描述不一致："
                            f"一处称“{left_metric}”，另一处称“{right_metric}”。"
                        ),
                        severity="high",
                    )
                )
    return flags


def _metric_conflict(left: FactClaim, right: FactClaim) -> tuple[str, str] | None:
    if not left.metrics or not right.metrics or not _shares_fact_context(left, right):
        return None
    for left_metric in left.metrics:
        left_value = _parse_metric(left_metric)
        if left_value is None:
            continue
        for right_metric in right.metrics:
            right_value = _parse_metric(right_metric)
            if right_value is None:
                continue
            if _metric_values_conflict(left_value, right_value):
                return left_metric, right_metric
    return None


def _shares_fact_context(left: FactClaim, right: FactClaim) -> bool:
    return bool(
        set(left.responsibilities) & set(right.responsibilities)
        or set(left.technologies) & set(right.technologies)
    )


def _parse_metric(metric: str) -> tuple[str, float, str] | None:
    match = METRIC_PATTERN.search(metric)
    if match is None:
        return None
    return (
        match.group("verb") or "",
        float(match.group("value")),
        match.group("unit").lower(),
    )


def _metric_values_conflict(
    left: tuple[str, float, str],
    right: tuple[str, float, str],
) -> bool:
    left_verb, left_value, left_unit = left
    right_verb, right_value, right_unit = right
    if left_unit != right_unit:
        return False
    left_direction = METRIC_VERB_DIRECTIONS.get(left_verb)
    right_direction = METRIC_VERB_DIRECTIONS.get(right_verb)
    if left_direction and right_direction and left_direction != right_direction:
        return True
    return left_value != right_value
