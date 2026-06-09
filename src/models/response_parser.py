from __future__ import annotations

import re

from src.datasets.schema import ParsedResponse

_YES_PATTERNS = (
    re.compile(r"^\s*yes\b", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*yes\b", re.IGNORECASE),
    re.compile(r"答案\s*[:：]\s*是"),
)
_NO_PATTERNS = (
    re.compile(r"^\s*no\b", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*no\b", re.IGNORECASE),
    re.compile(r"答案\s*[:：]\s*否"),
)
_UNCERTAIN_PATTERN = re.compile(
    r"\b(uncertain|cannot be determined|not sure|insufficient evidence)\b",
    re.IGNORECASE,
)


def normalize_yes_no(text: str) -> str:
    cleaned = text.strip()
    for pattern in _YES_PATTERNS:
        if pattern.search(cleaned):
            return "yes"
    for pattern in _NO_PATTERNS:
        if pattern.search(cleaned):
            return "no"
    if _UNCERTAIN_PATTERN.search(cleaned):
        return "unclear"
    lowered = cleaned.lower()
    if lowered in {"yes", "y", "是"}:
        return "yes"
    if lowered in {"no", "n", "否"}:
        return "no"
    return "unclear"


def parse_direct_response(raw_response: str) -> ParsedResponse:
    final_answer = raw_response.strip()
    if not final_answer:
        return ParsedResponse(parse_status="failed")
    return ParsedResponse(final_answer=final_answer, parse_status="ok")


def parse_cot_response(raw_response: str) -> ParsedResponse:
    text = raw_response.strip()
    if not text:
        return ParsedResponse(parse_status="failed")

    visual = _extract_section(text, "Visual Evidence", ("Reasoning", "Final Answer"))
    reasoning = _extract_section(text, "Reasoning", ("Final Answer",))
    final = _extract_section(text, "Final Answer", ())

    if final:
        return ParsedResponse(
            visual_evidence=visual,
            reasoning=reasoning,
            final_answer=final,
            parse_status="ok",
        )

    fallback = _last_nonempty_line(text)
    if fallback:
        return ParsedResponse(
            visual_evidence=visual,
            reasoning=reasoning,
            final_answer=fallback,
            parse_status="fallback",
        )

    return ParsedResponse(
        visual_evidence=visual, reasoning=reasoning, parse_status="failed"
    )


def parse_response(raw_response: str, prompt_type: str) -> ParsedResponse:
    if prompt_type == "evidence_grounded_cot":
        return parse_cot_response(raw_response)
    return parse_direct_response(raw_response)


def _extract_section(text: str, start_label: str, end_labels: tuple[str, ...]) -> str:
    start_pattern = _section_label_pattern(start_label)
    start_match = start_pattern.search(text)
    if not start_match:
        return ""

    start = start_match.end()
    end = len(text)
    for label in end_labels:
        end_pattern = _section_label_pattern(label)
        end_match = end_pattern.search(text, start)
        if end_match:
            end = min(end, end_match.start())
    return text[start:end].strip()


def _section_label_pattern(label: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?(?:\d+\.\s*)?(?:\*\*)?"
        rf"{re.escape(label)}\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?",
        re.IGNORECASE,
    )


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
