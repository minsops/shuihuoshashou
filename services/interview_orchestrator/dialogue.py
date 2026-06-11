from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from libs.schemas import QATurn, TranscriptSegment, Utterance


FALLBACK_QUESTION = "(未捕捉到面试官提问)"


@dataclass
class DialogueFeedResult:
    utterances: list[Utterance] = field(default_factory=list)
    turns: list[QATurn] = field(default_factory=list)

    def extend(self, other: "DialogueFeedResult") -> None:
        self.utterances.extend(other.utterances)
        self.turns.extend(other.turns)


class DialogueAssembler:
    """Aggregate sentence-level ASR segments into utterances and QA turns."""

    def __init__(self, silence_close_ms: int = 2500) -> None:
        self.silence_close_ms = silence_close_ms
        self._buffer: list[TranscriptSegment] = []
        self._pending_question: Utterance | None = None
        self._fallback_question = FALLBACK_QUESTION
        self._question_source: Literal["interviewer", "ai_probe"] = "interviewer"
        self._probe_target: str | None = None
        self._probe_chain_id: str | None = None
        self._pending_question_text: str | None = None
        self._pending_asked_option_id: str | None = None
        self._pending_question_origin: Literal["system_suggested", "interviewer_custom"] | None = None

    def feed(
        self,
        segment: TranscriptSegment,
        *,
        fallback_question: str | None = None,
        question_source: str = "interviewer",
        probe_target: str | None = None,
        probe_chain_id: str | None = None,
        force_close: bool = False,
    ) -> DialogueFeedResult:
        if not segment.is_final:
            return DialogueFeedResult()

        normalized_source: Literal["interviewer", "ai_probe"] = (
            "ai_probe" if question_source == "ai_probe" else "interviewer"
        )
        if not self._buffer:
            self._start_buffer(
                segment,
                fallback_question,
                normalized_source,
                probe_target,
                probe_chain_id,
            )
            if force_close:
                return self._seal_current()
            return DialogueFeedResult()

        result = DialogueFeedResult()
        if self._continues_current_utterance(segment):
            self._buffer.append(segment)
            self._merge_buffer_metadata(
                fallback_question,
                normalized_source,
                probe_target,
                probe_chain_id,
            )
            if force_close:
                result.extend(self._seal_current())
            return result

        result.extend(self._seal_current())
        self._start_buffer(
            segment,
            fallback_question,
            normalized_source,
            probe_target,
            probe_chain_id,
        )
        if force_close:
            result.extend(self._seal_current())
        return result

    def flush(self) -> DialogueFeedResult:
        if not self._buffer:
            return DialogueFeedResult()
        return self._seal_current()

    def annotate_pending_question(
        self,
        question: str,
        *,
        asked_option_id: str | None,
        question_origin: Literal["system_suggested", "interviewer_custom"],
    ) -> None:
        if self._pending_question is None:
            return
        clean_question = question.strip()
        if not clean_question:
            return
        self._pending_question_text = clean_question
        self._pending_asked_option_id = _clean_optional(asked_option_id)
        self._pending_question_origin = question_origin

    def _start_buffer(
        self,
        segment: TranscriptSegment,
        fallback_question: str | None,
        question_source: Literal["interviewer", "ai_probe"],
        probe_target: str | None,
        probe_chain_id: str | None,
    ) -> None:
        self._buffer = [segment]
        self._fallback_question = _clean_question(fallback_question)
        self._question_source = question_source
        self._probe_target = _clean_optional(probe_target)
        self._probe_chain_id = _clean_optional(probe_chain_id)

    def _merge_buffer_metadata(
        self,
        fallback_question: str | None,
        question_source: Literal["interviewer", "ai_probe"],
        probe_target: str | None,
        probe_chain_id: str | None,
    ) -> None:
        clean_question = _clean_question(fallback_question)
        if self._fallback_question == FALLBACK_QUESTION and clean_question != FALLBACK_QUESTION:
            self._fallback_question = clean_question
        if question_source == "ai_probe":
            self._question_source = "ai_probe"
        clean_probe_target = _clean_optional(probe_target)
        if self._probe_target is None and clean_probe_target is not None:
            self._probe_target = clean_probe_target
        clean_chain_id = _clean_optional(probe_chain_id)
        if self._probe_chain_id is None and clean_chain_id is not None:
            self._probe_chain_id = clean_chain_id

    def _continues_current_utterance(self, segment: TranscriptSegment) -> bool:
        last = self._buffer[-1]
        if segment.speaker != last.speaker:
            return False
        gap_ms = segment.start_ms - last.end_ms
        return gap_ms <= self.silence_close_ms

    def _seal_current(self) -> DialogueFeedResult:
        utterance = self._build_utterance()
        fallback_question = self._fallback_question
        question_source = self._question_source
        probe_target = self._probe_target
        probe_chain_id = self._probe_chain_id
        self._buffer = []
        self._fallback_question = FALLBACK_QUESTION
        self._question_source = "interviewer"
        self._probe_target = None
        self._probe_chain_id = None

        result = DialogueFeedResult(utterances=[utterance])
        if utterance.speaker == "interviewer":
            self._pending_question = utterance
            self._pending_question_text = None
            self._pending_asked_option_id = None
            self._pending_question_origin = None
            return result
        if utterance.speaker != "candidate":
            return result

        (
            turn_question,
            source,
            question_utterance_id,
            resolved_probe_target,
            asked_option_id,
            question_origin,
        ) = self._resolve_question(fallback_question, question_source, probe_target)
        result.turns.append(
            QATurn(
                question=turn_question,
                question_source=source,
                answer=utterance.text,
                answer_start_ms=utterance.start_ms,
                answer_end_ms=utterance.end_ms,
                probe_target=resolved_probe_target,
                question_utterance_id=question_utterance_id,
                answer_utterance_id=utterance.utterance_id,
                probe_chain_id=probe_chain_id,
                asked_option_id=asked_option_id,
                question_origin=question_origin,
            )
        )
        return result

    def _resolve_question(
        self,
        fallback_question: str,
        question_source: Literal["interviewer", "ai_probe"],
        probe_target: str | None,
    ) -> tuple[
        str,
        Literal["interviewer", "ai_probe"],
        str | None,
        str | None,
        str | None,
        Literal["system_suggested", "interviewer_custom"] | None,
    ]:
        if question_source == "ai_probe":
            return fallback_question, "ai_probe", None, probe_target or "AI 追问建议", None, None
        if self._pending_question is not None:
            question = self._pending_question
            annotated_question = self._pending_question_text or question.text
            asked_option_id = self._pending_asked_option_id
            question_origin = self._pending_question_origin
            self._pending_question = None
            self._pending_question_text = None
            self._pending_asked_option_id = None
            self._pending_question_origin = None
            return (
                annotated_question,
                "interviewer",
                question.utterance_id,
                None,
                asked_option_id,
                question_origin,
            )
        return fallback_question, "interviewer", None, None, None, None

    def _build_utterance(self) -> Utterance:
        start_ms = min(segment.start_ms for segment in self._buffer)
        end_ms = max(segment.end_ms for segment in self._buffer)
        return Utterance(
            speaker=self._buffer[0].speaker,
            text=_join_segment_text(segment.text for segment in self._buffer),
            start_ms=start_ms,
            end_ms=end_ms,
            sentence_count=len(self._buffer),
        )


def _clean_question(value: str | None) -> str:
    if value is None:
        return FALLBACK_QUESTION
    clean = value.strip()
    if not clean or clean == "实时输入片段":
        return FALLBACK_QUESTION
    return clean


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    return clean or None


def _join_segment_text(values) -> str:
    return " ".join(value.strip() for value in values if value.strip())
