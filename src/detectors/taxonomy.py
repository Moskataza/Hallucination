from __future__ import annotations

FACTUAL_TYPES = {"OBJ", "ATT", "SPA"}
LOGICAL_TYPES = {"IR", "CI", "INC", "SO"}
FINE_TYPES = FACTUAL_TYPES | LOGICAL_TYPES | {"None", "Unclear"}
COARSE_TYPES = {"Factual", "Logical", "None", "Unclear"}

_FINE_CANONICAL = {label.upper(): label for label in FINE_TYPES}
_COARSE_CANONICAL = {label.upper(): label for label in COARSE_TYPES}


def infer_coarse_from_fine(fine: str) -> str:
    normalized = normalize_fine_label(fine)
    if normalized in FACTUAL_TYPES:
        return "Factual"
    if normalized in LOGICAL_TYPES:
        return "Logical"
    if normalized == "Unclear":
        return "Unclear"
    return "None"


def normalize_fine_label(value: str | None) -> str:
    if value is None:
        return "None"
    normalized = value.strip().upper()
    return _FINE_CANONICAL.get(normalized, "Unclear")


def normalize_coarse_label(value: str | None, fine: str | None = None) -> str:
    if value is None or not value.strip():
        return infer_coarse_from_fine(fine or "None")
    normalized = value.strip().upper()
    return _COARSE_CANONICAL.get(normalized, infer_coarse_from_fine(fine or "None"))
