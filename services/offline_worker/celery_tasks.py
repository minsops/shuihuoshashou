from __future__ import annotations

from typing import Any

from libs.common.config import get_settings

OFFLINE_SCORING_TASK = "interview.offline_scoring"


def create_celery_app() -> Any:
    try:
        from celery import Celery
    except ImportError as exc:
        raise RuntimeError(
            "Celery worker requires optional celery dependencies. "
            "Install with `pip install -e .[celery]`."
        ) from exc
    settings = get_settings()
    app = Celery(
        "shuihuo",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    app.conf.task_default_queue = "shuihuo-offline"
    return app


def register_tasks(app: Any) -> Any:
    @app.task(name=OFFLINE_SCORING_TASK)
    def offline_scoring(interview_id: str) -> dict[str, Any]:
        from services.interview_orchestrator.service import run_offline_scoring_task

        report = run_offline_scoring_task(interview_id)
        return report.model_dump(mode="json")

    return offline_scoring


def create_registered_celery_app() -> Any:
    app = create_celery_app()
    register_tasks(app)
    return app


try:
    celery_app = create_registered_celery_app()
except RuntimeError:
    celery_app = None
