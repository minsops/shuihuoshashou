from __future__ import annotations

import builtins
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from libs.common.config import get_settings
from libs.llm_client import LLMClient
from services.document_service import service as document_service
from services.report_service.service import _write_text_fallback_pdf
from services.document_service.service import parse_document


def _docx_bytes(text: str) -> bytes:
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_parse_text_document_without_llm(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    result = parse_document(
        "resume.md",
        "候选人负责 FastAPI 网关和模型编排".encode("utf-8"),
        kind="resume",
        content_type="text/markdown",
    )

    assert result.source == "text"
    assert result.llm_attempted is False
    assert result.used_llm is False
    assert "FastAPI" in result.text


def test_parse_docx_document_without_python_docx(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    result = parse_document(
        "resume.docx",
        _docx_bytes("候选人做过评估报告链路和异常重试"),
        kind="resume",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert result.source == "docx"
    assert "异常重试" in result.text


def test_parse_pdf_document(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    pdf_path = tmp_path / "sample.pdf"
    _write_text_fallback_pdf(["岗位要求 Python FastAPI 稳定性治理"], pdf_path)
    pdf = pdf_path.read_bytes()

    result = parse_document("jd.pdf", pdf, kind="jd", content_type="application/pdf")

    assert result.source == "pdf"
    assert "FastAPI" in result.text


def test_parse_document_uses_mime_type_without_extension(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    pdf_path = tmp_path / "sample.pdf"
    _write_text_fallback_pdf(["岗位要求 Python FastAPI 稳定性治理"], pdf_path)

    pdf_result = parse_document(
        "upload",
        pdf_path.read_bytes(),
        kind="jd",
        content_type="application/pdf; charset=binary",
    )
    docx_result = parse_document(
        "upload",
        _docx_bytes("候选人做过评估报告链路和异常重试"),
        kind="resume",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert pdf_result.source == "pdf"
    assert "FastAPI" in pdf_result.text
    assert docx_result.source == "docx"
    assert "异常重试" in docx_result.text


def test_parse_document_unsupported_type_lists_log(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    with pytest.raises(ValueError) as exc:
        parse_document("resume.xyz", b"opaque", kind="resume", content_type="application/x-unknown")

    message = str(exc.value)
    assert "unsupported document type" in message
    assert "log" in message


def test_parse_image_document_missing_ocr_dependency_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ocrmac":
            raise ImportError("missing ocrmac")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError) as exc:
        parse_document("resume.png", b"image-bytes", kind="resume", content_type="image/png")

    message = str(exc.value)
    assert "pip install -e '.[ocr]'" in message
    assert "upload a PDF/DOCX/text resume instead" in message


def test_parse_legacy_doc_failure_has_conversion_hint(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    def fail_textutil(*args, **kwargs):
        raise FileNotFoundError("textutil")

    monkeypatch.setattr(document_service.subprocess, "run", fail_textutil)

    with pytest.raises(ValueError) as exc:
        parse_document("resume.doc", b"doc-bytes", kind="resume", content_type="application/msword")

    message = str(exc.value)
    assert "legacy .doc parsing requires macOS textutil" in message
    assert "convert the file to .docx or PDF" in message


def test_parse_document_uses_deepseek_cleanup(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "候选人简历" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"text": "候选人负责 FastAPI 网关、LLM JSON 解析和报告生成"},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "services.document_service.service.get_llm_client",
        lambda: LLMClient(transport=httpx.MockTransport(handler)),
    )

    result = parse_document("resume.txt", b"noise\nFastAPI", kind="resume")

    assert result.llm_attempted is True
    assert result.used_llm is True
    assert result.text == "候选人负责 FastAPI 网关、LLM JSON 解析和报告生成"
    assert result.warning == ""


def test_parse_document_warns_when_deepseek_cleanup_fails(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, text="bad key")

    monkeypatch.setattr(
        "services.document_service.service.get_llm_client",
        lambda: LLMClient(transport=httpx.MockTransport(handler)),
    )

    result = parse_document("resume.txt", b"noise\nFastAPI", kind="resume")

    assert result.llm_attempted is True
    assert result.used_llm is False
    assert result.text == "noise\nFastAPI"
    assert "DeepSeek 文档清洗失败" in result.warning
    assert "已使用原始解析文本" in result.warning
    assert "HTTP 401" in result.warning
    assert "bad key" not in result.warning
