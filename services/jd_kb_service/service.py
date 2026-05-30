from __future__ import annotations

import re
from datetime import datetime

from libs.common.database import connect, dumps, init_db, loads
from libs.schemas import CompetencyItem, CompetencyModel, JobCreate, JobRecord, ProbePatternHit


DEFAULT_DIMENSIONS = [
    ("专业能力深度", "核心技能掌握、技术取舍、问题定位与解决能力", 0.30),
    ("项目真实性", "能否讲清楚个人职责、细节、困难和真实贡献", 0.25),
    ("岗位匹配度", "经历、动机与岗位职责和业务场景的契合程度", 0.20),
    ("沟通与逻辑", "表达结构、因果链和信息密度", 0.15),
    ("注水风险", "AIGC、模板化、前后矛盾与追问露馅风险", -0.10),
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_+\-.#]+|[\u4e00-\u9fff]{2,}", text.lower()))


def _score_pattern(query: str, competency: str, pattern: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    haystack = _tokens(f"{competency} {pattern}")
    overlap = len(query_tokens & haystack)
    phrase_bonus = 0.0
    lowered = f"{competency} {pattern}".lower()
    for token in query_tokens:
        if token in lowered:
            phrase_bonus += 0.25
    return round(overlap + phrase_bonus, 3)


def generate_competency_model(job_id: str, title: str, jd_text: str) -> CompetencyModel:
    text = jd_text.lower()
    items: list[CompetencyItem] = []
    for name, description, weight in DEFAULT_DIMENSIONS:
        patterns = [
            f"请围绕{name}追问一个可验证的具体案例。",
            "请要求候选人说明自己负责的具体部分、为什么这样做、遇到的坑。",
        ]
        if "python" in text or "fastapi" in text:
            patterns.append("请追问 Python/FastAPI 的工程设计、性能和异常处理细节。")
        if "ai" in text or "llm" in text or "大模型" in text:
            patterns.append("请追问 LLM 调用、评估、成本、失败降级和安全边界。")
        items.append(CompetencyItem(name=name, description=description, probe_patterns=patterns, weight=weight))
    return CompetencyModel(job_id=job_id, job_title=title, items=items)


def retrieve_probe_patterns(
    competency_model: CompetencyModel,
    query: str,
    *,
    limit: int = 5,
) -> list[ProbePatternHit]:
    hits: list[ProbePatternHit] = []
    for item in competency_model.items:
        for pattern in item.probe_patterns:
            score = _score_pattern(query, item.name, pattern)
            if score > 0:
                hits.append(
                    ProbePatternHit(
                        job_id=competency_model.job_id,
                        competency=item.name,
                        pattern=pattern,
                        score=score,
                    )
                )
    hits.sort(key=lambda hit: (-hit.score, hit.competency, hit.pattern))
    return hits[:limit]


def retrieve_job_probe_patterns(job_id: str, query: str, *, limit: int = 5) -> list[ProbePatternHit]:
    return retrieve_probe_patterns(get_job(job_id).competency_model, query, limit=limit)


def create_job(payload: JobCreate) -> JobRecord:
    init_db()
    record = JobRecord(
        title=payload.title,
        jd_text=payload.jd_text,
        competency_model=generate_competency_model("", payload.title, payload.jd_text),
    )
    record.competency_model.job_id = record.id
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, title, jd_text, competency_model, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                record.id,
                record.title,
                record.jd_text,
                dumps(record.competency_model.model_dump()),
                record.created_at.isoformat(),
            ),
        )
    return record


def get_job(job_id: str) -> JobRecord:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(f"job not found: {job_id}")
    return JobRecord(
        id=row["id"],
        title=row["title"],
        jd_text=row["jd_text"],
        competency_model=CompetencyModel.model_validate(loads(row["competency_model"])),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
