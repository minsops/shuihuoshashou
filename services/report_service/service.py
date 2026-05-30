from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from jinja2 import Template

from libs.common.config import get_settings
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
        <h2>维度得分</h2>
        <table>
          <tr><th>维度</th><th>分数</th><th>权重</th><th>证据</th></tr>
          {% for d in score.dimensions %}
          <tr>
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
    </body>
    </html>
    """
)


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
    html = REPORT_TEMPLATE.render(score=score, summary=summary, flags=ctx.flags)
    settings = get_settings()
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    html_path = Path(settings.report_dir / f"{ctx.session_id}.html")
    pdf_path = Path(settings.report_dir / f"{ctx.session_id}.pdf")
    html_path.write_text(html, encoding="utf-8")
    _write_pdf(html, pdf_path)
    report = Report(
        interview_id=ctx.session_id,
        score=score,
        aigc_results=aigc,
        consistency_flags=ctx.flags,
        summary=summary,
        html_path=str(html_path),
        pdf_path=str(pdf_path),
    )
    return report, html
