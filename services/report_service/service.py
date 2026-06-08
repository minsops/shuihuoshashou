from __future__ import annotations

import base64
import json
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path

from jinja2 import Template

from libs.common.config import get_settings
from libs.common.storage import get_artifact_store
from libs.schemas import AIGCResult, ConsistencyFlag, InterviewContext, InterviewScore, Report


REPORT_TEMPLATE = Template(
    """
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <title>水货杀手面试报告</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }
        h1, h2 { margin: 0 0 12px; }
        section { margin: 24px 0; }
        table { border-collapse: collapse; width: 100%; }
        td, th { border: 1px solid #d9dee8; padding: 8px; text-align: left; }
        .risk { color: #b42318; font-weight: 700; }
        .flagged { background: #fff1f0; color: #912018; font-weight: 700; }
        .highlight { background: #ecfdf3; color: #05603a; font-weight: 700; }
      </style>
    </head>
    <body>
      <h1>水货杀手面试报告</h1>
      <p>总分：<strong>{{ score.total_score }}</strong> ｜ 建议：<strong>{{ score.recommendation }}</strong></p>
      <section>
        <h2>总评</h2>
        <p>{{ summary }}</p>
      </section>
      {% if candidate_resume_text %}
      <section>
        <h2>简历摘要</h2>
        <p>{{ candidate_resume_text }}</p>
      </section>
      {% endif %}
      <section>
        <h2>亮点</h2>
        {% if strengths %}
        <ul>
          {% for item in strengths %}
          <li class="highlight">{{ item }}</li>
          {% endfor %}
        </ul>
        {% else %}
        <p>暂无明确高可信亮点，建议结合追问继续观察。</p>
        {% endif %}
      </section>
      <section>
        <h2>维度得分</h2>
        {% if radar_chart_uri %}
        <img src="{{ radar_chart_uri }}" alt="维度雷达图" style="max-width: 520px; width: 100%; margin: 8px 0 18px;">
        {% endif %}
        <table>
          <tr><th>编号</th><th>维度</th><th>分数</th><th>权重</th><th>证据</th></tr>
          {% for d in score.dimensions %}
          <tr>
            <td>D{{ loop.index }}</td>
            <td>{{ d.dimension }}</td>
            <td>{{ d.score }}</td>
            <td>{{ d.weight }}</td>
            <td>{% for e in d.evidence %}{{ e.excerpt }}<br>{% endfor %}</td>
          </tr>
          {% endfor %}
        </table>
      </section>
      <section>
        <h2>风险点</h2>
        {% for note in risk_highlights %}<p class="risk">{{ note }}</p>{% endfor %}
      </section>
      <section>
        <h2>AIGC 察重</h2>
        <table>
          <tr><th>Turn ID</th><th>AI 概率</th><th>模板相似度</th><th>命中模板</th><th>状态</th></tr>
          {% for item in aigc_results %}
          <tr class="{{ 'flagged' if item.flagged else '' }}">
            <td>{{ item.turn_id }}</td>
            <td>{{ item.ai_generated_prob }}</td>
            <td>{{ item.template_similarity }}</td>
            <td>{{ item.matched_template or "" }}</td>
            <td>{{ "疑似注水/模板化" if item.flagged else "未命中" }}</td>
          </tr>
          {% endfor %}
        </table>
      </section>
      <section>
        <h2>转写全文</h2>
        <table>
          <tr><th>序号</th><th>来源</th><th>问题</th><th>回答</th><th>时间</th><th>追问目标</th></tr>
          {% for turn in transcript %}
          <tr>
            <td>{{ loop.index }}</td>
            <td>{{ turn.question_source }}</td>
            <td>{{ turn.question }}</td>
            <td>{{ turn.answer }}</td>
            <td>{{ turn.answer_start_ms }}ms - {{ turn.answer_end_ms }}ms</td>
            <td>{{ turn.probe_target or "" }}</td>
          </tr>
          {% endfor %}
        </table>
      </section>
    </body>
    </html>
    """
)


def _strengths(score: InterviewScore) -> list[str]:
    strengths: list[str] = []
    for dimension in score.dimensions:
        if dimension.score < 75 or not dimension.evidence:
            continue
        excerpt = dimension.evidence[0].excerpt
        strengths.append(f"{dimension.dimension} 得分 {dimension.score}：{excerpt}")
    return strengths[:3]


def _radar_chart_uri(score: InterviewScore) -> str:
    if not score.dimensions:
        return ""

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [f"D{index}" for index, _ in enumerate(score.dimensions, start=1)]
        values = [dimension.score for dimension in score.dimensions]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig = plt.figure(figsize=(4.8, 4.2), dpi=150)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, color="#2563eb", linewidth=2)
        ax.fill(angles, values, color="#60a5fa", alpha=0.25)
        ax.set_ylim(0, 100)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.grid(color="#cbd5e1", linewidth=0.8)
        fig.tight_layout()

        image = BytesIO()
        fig.savefig(image, format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _risk_highlights(score: InterviewScore, flags: list[ConsistencyFlag]) -> list[str]:
    highlights: list[str] = []
    for note in [*score.risk_notes, *(flag.description for flag in flags)]:
        clean = note.strip()
        if clean and clean not in highlights:
            highlights.append(clean)
    return highlights


def _write_pdf(html: str, pdf_path: Path, fallback_lines: list[str]) -> None:
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from weasyprint import HTML

            HTML(string=html).write_pdf(pdf_path)
        return
    except Exception:
        # Local fallback for machines without WeasyPrint's native Pango/GObject stack.
        _write_text_fallback_pdf(fallback_lines, pdf_path)


def _write_text_fallback_pdf(lines: list[str], pdf_path: Path) -> None:
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(_wrap_pdf_text(line, width=92))
    wrapped_lines = wrapped_lines[:42]
    commands = ["BT", "/F1 10 Tf", "72 740 Td", "14 TL"]
    for index, line in enumerate(wrapped_lines):
        if index:
            commands.append("T*")
        commands.append(f"{_pdf_font_for_line(line)} 10 Tf")
        commands.append(f"{_pdf_text_operand(line)} Tj")
    commands.append("ET")
    stream = "\n".join(commands)
    stream_bytes = stream.encode("latin-1")
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        (
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R /F2 6 0 R >> >> /Contents 5 0 R >> endobj"
        ),
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream_bytes)} >> stream\n{stream}\nendstream endobj",
        (
            "6 0 obj << /Type /Font /Subtype /Type0 /BaseFont /STSong-Light "
            "/Encoding /UniGB-UCS2-H /DescendantFonts [7 0 R] >> endobj"
        ),
        (
            "7 0 obj << /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 5 >> "
            "/FontDescriptor 8 0 R >> endobj"
        ),
        (
            "8 0 obj << /Type /FontDescriptor /FontName /STSong-Light /Flags 4 "
            "/FontBBox [0 -120 1000 880] /ItalicAngle 0 /Ascent 880 /Descent -120 "
            "/CapHeight 880 /StemV 80 >> endobj"
        ),
    ]
    content = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(content.encode("latin-1")))
        content += obj + "\n"
    xref_pos = len(content.encode("latin-1"))
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        content += f"{offset:010d} 00000 n \n"
    content += (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )
    pdf_path.write_bytes(content.encode("latin-1"))


def _pdf_font_for_line(value: str) -> str:
    return "/F2" if any(ord(char) > 126 for char in value) else "/F1"


def _pdf_text_operand(value: str) -> str:
    if any(ord(char) > 126 for char in value):
        return f"<{value.encode('utf-16-be').hex().upper()}>"
    return f"({_pdf_escape(value)})"


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_text(value: str, *, width: int) -> list[str]:
    if len(value) <= width:
        return [value]
    lines: list[str] = []
    remaining = value
    while len(remaining) > width:
        split_at = remaining.rfind(" ", 0, width)
        if split_at <= 0:
            split_at = width
        lines.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    if remaining:
        lines.append(remaining)
    return lines


def _write_report_json(report: Report, json_path: Path) -> None:
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_report(ctx: InterviewContext, score: InterviewScore, aigc: list[AIGCResult]) -> tuple[Report, str]:
    _validate_report_inputs(ctx, score, aigc)
    summary = (
        f"候选人在本场面试中获得 {score.total_score} 分，系统建议为 {score.recommendation}。"
        "报告中的每项评分均绑定原始回答证据，注水风险需由面试官结合上下文复核。"
    )
    html = REPORT_TEMPLATE.render(
        score=score,
        summary=summary,
        strengths=_strengths(score),
        aigc_results=aigc,
        risk_highlights=_risk_highlights(score, ctx.flags),
        transcript=ctx.turns,
        candidate_resume_text=ctx.candidate_resume_text,
        radar_chart_uri=_radar_chart_uri(score),
    )
    settings = get_settings()
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(settings.report_dir / f"{ctx.session_id}.report.json")
    html_path = Path(settings.report_dir / f"{ctx.session_id}.html")
    pdf_path = Path(settings.report_dir / f"{ctx.session_id}.pdf")
    transcript_path = Path(settings.report_dir / f"{ctx.session_id}.transcript.json")
    html_path.write_text(html, encoding="utf-8")
    transcript_path.write_text(
        json.dumps(
            [turn.model_dump() for turn in ctx.turns],
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_pdf(html, pdf_path, _fallback_pdf_lines(ctx, score, aigc, summary))
    artifact_store = get_artifact_store()
    html_artifact = artifact_store.put_file(
        f"reports/{ctx.session_id}.html",
        html_path,
        "text/html; charset=utf-8",
    )
    pdf_artifact = artifact_store.put_file(
        f"reports/{ctx.session_id}.pdf",
        pdf_path,
        "application/pdf",
    )
    transcript_artifact = artifact_store.put_file(
        f"reports/{ctx.session_id}.transcript.json",
        transcript_path,
        "application/json; charset=utf-8",
    )
    json_artifact_name = f"reports/{ctx.session_id}.report.json"
    report = Report(
        interview_id=ctx.session_id,
        score=score,
        aigc_results=aigc,
        consistency_flags=ctx.flags,
        transcript=ctx.turns,
        candidate_resume_text=ctx.candidate_resume_text,
        summary=summary,
        json_path=str(json_path),
        html_path=str(html_path),
        pdf_path=str(pdf_path),
        transcript_path=str(transcript_path),
        artifact_uris={
            "html": html_artifact.uri,
            "pdf": pdf_artifact.uri,
            "transcript": transcript_artifact.uri,
            "json": artifact_store.artifact_uri(json_artifact_name, json_path),
        },
    )
    _write_report_json(report, json_path)
    artifact_store.put_file(
        json_artifact_name,
        json_path,
        "application/json; charset=utf-8",
    )
    return report, html


def _fallback_pdf_lines(
    ctx: InterviewContext,
    score: InterviewScore,
    aigc: list[AIGCResult],
    summary: str,
) -> list[str]:
    lines = [
        "Shuihuo Killer Interview Report",
        f"Interview: {ctx.session_id}",
        f"Total score: {score.total_score}",
        f"Recommendation: {score.recommendation}",
        f"Summary: {summary}",
        "Dimension scores:",
    ]
    for dimension in score.dimensions:
        evidence = dimension.evidence[0].excerpt if dimension.evidence else ""
        lines.append(
            f"- {dimension.dimension}: score={dimension.score}, weight={dimension.weight}, "
            f"evidence={evidence}"
        )
    lines.append("Risk notes:")
    risk_highlights = _risk_highlights(score, ctx.flags)
    if risk_highlights:
        lines.extend(f"- {note}" for note in risk_highlights)
    else:
        lines.append("- None")
    lines.append("AIGC checks:")
    for result in aigc:
        lines.append(
            f"- turn={result.turn_id}, ai_prob={result.ai_generated_prob}, "
            f"template_similarity={result.template_similarity}, flagged={result.flagged}"
        )
    lines.append("Transcript:")
    for index, turn in enumerate(ctx.turns, start=1):
        lines.append(f"- T{index} [{turn.question_source}] Q={turn.question} A={turn.answer}")
    return lines


def _validate_report_inputs(
    ctx: InterviewContext,
    score: InterviewScore,
    aigc: list[AIGCResult],
) -> None:
    if score.session_id != ctx.session_id:
        raise ValueError("score session_id must match interview context session_id")
    expected_dimension_names = [item.name for item in ctx.competency_model.items]
    actual_dimension_names = [dimension.dimension for dimension in score.dimensions]
    if actual_dimension_names != expected_dimension_names:
        raise ValueError("score dimensions must match competency model items")
    for dimension, item in zip(score.dimensions, ctx.competency_model.items, strict=True):
        if abs(dimension.weight - item.weight) > 1e-9:
            raise ValueError(f"score dimension weight must match competency model: {dimension.dimension}")
    expected_total = _compute_total_score(score)
    if abs(score.total_score - expected_total) > 0.01:
        raise ValueError("score total_score must match dimension scores and weights")
    expected_recommendation = _recommendation(expected_total)
    if score.recommendation != expected_recommendation:
        raise ValueError("score recommendation must match total_score")
    turns_by_id = {turn.turn_id: turn for turn in ctx.turns}
    if not turns_by_id:
        raise ValueError("report requires at least one transcript turn")
    for flag in ctx.flags:
        if flag.turn_id_a not in turns_by_id:
            raise ValueError(f"consistency flag references unknown turn_id: {flag.turn_id_a}")
        if flag.turn_id_b not in turns_by_id:
            raise ValueError(f"consistency flag references unknown turn_id: {flag.turn_id_b}")
    for dimension in score.dimensions:
        evidence_refs: set[tuple[str, int, int, str]] = set()
        for evidence in dimension.evidence:
            turn = turns_by_id.get(evidence.turn_id)
            if turn is None:
                raise ValueError(f"score evidence references unknown turn_id: {evidence.turn_id}")
            evidence_ref = (
                evidence.turn_id,
                evidence.quote_start_ms,
                evidence.quote_end_ms,
                evidence.excerpt,
            )
            if evidence_ref in evidence_refs:
                raise ValueError(
                    f"score evidence contains duplicate reference: {dimension.dimension}"
                )
            evidence_refs.add(evidence_ref)
            if (
                evidence.quote_start_ms < turn.answer_start_ms
                or evidence.quote_end_ms > turn.answer_end_ms
            ):
                raise ValueError(
                    f"score evidence timestamp is outside turn range: {evidence.turn_id}"
                )
            if evidence.excerpt not in turn.answer:
                raise ValueError(f"score evidence excerpt is not in turn answer: {evidence.turn_id}")
    aigc_turn_ids = [result.turn_id for result in aigc]
    duplicate_aigc_turn_ids = {
        turn_id for turn_id in aigc_turn_ids if aigc_turn_ids.count(turn_id) > 1
    }
    if duplicate_aigc_turn_ids:
        raise ValueError("AIGC results must not contain duplicate turn_id values")
    unknown_aigc_turn_ids = set(aigc_turn_ids) - set(turns_by_id)
    if unknown_aigc_turn_ids:
        raise ValueError(f"AIGC result references unknown turn_id: {sorted(unknown_aigc_turn_ids)[0]}")
    missing_aigc_turn_ids = set(turns_by_id) - set(aigc_turn_ids)
    if missing_aigc_turn_ids:
        raise ValueError("AIGC results must cover every transcript turn")
    settings = get_settings()
    for result in aigc:
        if not result.flagged and (
            result.ai_generated_prob >= settings.aigc_ai_prob_threshold
            or result.template_similarity >= settings.aigc_template_similarity_threshold
        ):
            raise ValueError("AIGC result flagged must be true when thresholds are exceeded")


def _compute_total_score(score: InterviewScore) -> float:
    positive = [dimension for dimension in score.dimensions if dimension.weight > 0]
    weight_sum = sum(dimension.weight for dimension in positive) or 1.0
    total = sum(dimension.score * dimension.weight for dimension in positive) / weight_sum
    for dimension in score.dimensions:
        if dimension.weight < 0:
            total -= (100.0 - dimension.score) * abs(dimension.weight)
    return round(max(0.0, min(100.0, total)), 2)


def _recommendation(total_score: float) -> str:
    if total_score >= 88:
        return "strong_yes"
    if total_score >= 75:
        return "yes"
    if total_score >= 60:
        return "hold"
    return "no"
