"""Prediction-side normalization required by the case contract.

The model supplies facts and turn indices. Diarization is authoritative for
speaker, so code fills speaker from the source turn. Dates that cannot be
parsed, or intervals whose end precedes their start, become missing dates
instead of discarding every fact in the call. Gold validation stays strict.
"""
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from schema import (Attribute, CallInput, Facts, ObjectRef, Source, Speaker, StrictModel,
                    load_answer_payload, schema_error)


class RawFact(StrictModel):
    object: ObjectRef
    attribute: Attribute
    value: str = Field(min_length=1)
    polarity: Literal[-1, 1]
    valid_from: str | None
    valid_to: str | None
    source: Source
    speaker: Speaker | None = None
    confirmed: bool | None = None

    @field_validator("polarity", mode="before")
    @classmethod
    def strict_polarity(cls, value):
        if type(value) is not int or value not in (-1, 1):
            raise ValueError("polarity must be the integer -1 or 1")
        return value


class RawFacts(StrictModel):
    facts: list[RawFact]


def _parse_datetime(value):
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None


def sanitize_interval(valid_from, valid_to):
    changes = {}
    start = _parse_datetime(valid_from)
    if valid_from is not None and start is None:
        valid_from = None
        changes["invalid_dates_dropped"] = changes.get("invalid_dates_dropped", 0) + 1
    end = _parse_datetime(valid_to)
    if valid_to is not None and end is None:
        valid_to = None
        changes["invalid_dates_dropped"] = changes.get("invalid_dates_dropped", 0) + 1
    if valid_from is not None and valid_to is not None:
        try:
            if end < start:
                valid_to = None
                changes["reversed_intervals_opened"] = changes.get(
                    "reversed_intervals_opened", 0) + 1
        except TypeError:
            valid_to = None
            changes["date_timezone_mismatch"] = changes.get("date_timezone_mismatch", 0) + 1
    return valid_from, valid_to, changes


def normalize_prediction(raw: RawFacts, call: CallInput):
    """Fill diarization speaker and make date strings structurally valid."""
    turns = {turn.idx: turn for turn in call.turns}
    facts, summary = [], {}
    for fact in raw.facts:
        item = fact.model_dump(mode="json")
        turn = turns.get(item["source"]["turn_idx"])
        if turn is not None and item.get("speaker") != turn.speaker:
            item["speaker"] = turn.speaker
            summary["speaker_from_turn"] = summary.get("speaker_from_turn", 0) + 1
        item["valid_from"], item["valid_to"], changes = sanitize_interval(
            item["valid_from"], item["valid_to"])
        for key, value in changes.items():
            summary[key] = summary.get(key, 0) + value
        facts.append(item)
    return Facts.model_validate({"facts": facts}), summary


def parse_prediction(raw: str, call: CallInput):
    """Parse a model answer and normalize prediction-only fields.

    Returns (Facts | None, error | None, normalization summary | None).
    """
    payload, error = load_answer_payload(raw)
    if error:
        return None, error, None
    try:
        raw_facts = RawFacts.model_validate(payload)
    except ValueError as raw_error:
        try:
            Facts.model_validate(payload)
        except ValueError as exc:
            return None, schema_error(exc), None
        return None, schema_error(raw_error), None
    try:
        facts, summary = normalize_prediction(raw_facts, call)
    except ValueError as exc:
        return None, schema_error(exc), None
    return facts, None, summary or None
