from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path

from jinja2 import Template

from libs.common.config import get_settings
from libs.common.storage import get_artifact_store
from libs.schemas import AIGCResult, InterviewContext, InterviewScore, Report


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
        {% for note in score.risk_notes %}<p class="risk">{{ note }}</p>{% endfor %}
        {% for flag in flags %}<p class="risk">{{ flag.description }}</p>{% endfor %}
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


def _write_pdf(html: str, pdf_path: Path) -> None:
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from weasyprint import HTML

            HTML(string=html).write_pdf(pdf_path)
        return
    except Exception:
        # Local fallback for machines without WeasyPrint's native Pango/GObject stack.
        text = "Shuihuo Killer interview report. Open the paired HTML report for full details."
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        objects = [
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            (
                "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj"
            ),
            "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
            f"5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj",
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


def build_report(ctx: InterviewContext, score: InterviewScore, aigc: list[AIGCResult]) -> tuple[Report, str]:
    summary = (
        f"候选人在本场面试中获得 {score.total_score} 分，系统建议为 {score.recommendation}。"
        "报告中的每项评分均绑定原始回答证据，注水风险需由面试官结合上下文复核。"
    )
    html = REPORT_TEMPLATE.render(
        score=score,
        summary=summary,
        strengths=_strengths(score),
        aigc_results=aigc,
        flags=ctx.flags,
        transcript=ctx.turns,
        radar_chart_uri=_radar_chart_uri(score),
    )
    settings = get_settings()
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    html_path = Path(settings.report_dir / f"{ctx.session_id}.html")
    pdf_path = Path(settings.report_dir / f"{ctx.session_id}.pdf")
    html_path.write_text(html, encoding="utf-8")
    _write_pdf(html, pdf_path)
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
    report = Report(
        interview_id=ctx.session_id,
        score=score,
        aigc_results=aigc,
        consistency_flags=ctx.flags,
        transcript=ctx.turns,
        summary=summary,
        html_path=str(html_path),
        pdf_path=str(pdf_path),
        artifact_uris={"html": html_artifact.uri, "pdf": pdf_artifact.uri},
    )
    return report, html
