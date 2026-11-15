"""Prediction-side normalization required by the case contract.

The model supplies facts and turn indices. Diarization is authoritative for
speaker, so code fills speaker from the source turn. Dates that cannot be
parsed, or intervals whose end precedes their start, become missing dates
instead of discarding every fact in the call. Gold validation stays strict.
"""
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from config import (CATALOG_PATH, TEMPORAL_NORMALIZATION, TEMPORAL_MODES,
                    TEMPORAL_CATALOG_PATH, FACT_RECOVERY)
from catalog_normalization import load_catalog, normalize_catalog_facts
from schema import (Attribute, CallInput, Facts, ObjectRef, Source, Speaker, StrictModel,
                    load_answer_payload, schema_error, source_problems)
from temporal import normalize_fact_dates


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


def normalize_prediction(raw: RawFacts, call: CallInput, *, temporal_mode=None,
                         temporal_decisions=None, catalog_path=None, date_catalog_path=None,
                         fact_recovery=None):
    """Fill diarization speaker, make dates valid and remove exact duplicates."""
    turns = {turn.idx: turn for turn in call.turns}
    facts, seen, summary = [], set(), {}
    items = [fact.model_dump(mode="json") for fact in raw.facts]
    catalog_path = CATALOG_PATH if catalog_path is None else catalog_path
    if catalog_path:
        items, catalog_changes = normalize_catalog_facts(items, load_catalog(catalog_path))
        if catalog_changes:
            summary['catalog_fields_normalized'] = len(catalog_changes)
    temporal_mode = TEMPORAL_NORMALIZATION if temporal_mode is None else temporal_mode
    if temporal_mode not in TEMPORAL_MODES:
        raise ValueError('Unknown temporal normalization mode')
    if temporal_mode != 'off':
        layers = None
        if temporal_mode == 'case_v2':
            from temporal_case_v2 import load_layers
            layers = load_layers(TEMPORAL_CATALOG_PATH if date_catalog_path is None else date_catalog_path)
        items, decisions = normalize_fact_dates(items, call, mode=temporal_mode, layers=layers)
        if temporal_decisions is not None:
            temporal_decisions.extend(decisions)
        changed = sum(len(d['changes']) for d in decisions)
        if changed:
            summary['temporal_dates_repaired'] = changed
    recovery_enabled = FACT_RECOVERY if fact_recovery is None else fact_recovery
    if recovery_enabled:
        from fact_recovery import recover_facts
        items, recovered = recover_facts(items, call, catalog_path)
        if recovered:
            summary['recovered_facts'] = len(recovered)
            if temporal_decisions is not None:
                temporal_decisions.extend(recovered)
    for item in items:
        turn = turns.get(item["source"]["turn_idx"])
        if turn is not None and item.get("speaker") != turn.speaker:
            item["speaker"] = turn.speaker
            summary["speaker_from_turn"] = summary.get("speaker_from_turn", 0) + 1
        item["valid_from"], item["valid_to"], changes = sanitize_interval(
            item["valid_from"], item["valid_to"])
        for key, value in changes.items():
            summary[key] = summary.get(key, 0) + value
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            summary["exact_duplicates_removed"] = summary.get(
                "exact_duplicates_removed", 0) + 1
            continue
        seen.add(key)
        facts.append(item)
    return Facts.model_validate({"facts": facts}), summary


def salvage_sources(facts: Facts, call: CallInput):
    """Keep valid facts when only some sources are invalid; never repair a source."""
    valid, dropped = [], 0
    for fact in facts.facts:
        if source_problems(Facts(facts=[fact]), call):
            dropped += 1
        else:
            valid.append(fact)
    if not valid:
        return None, dropped
    return Facts(facts=valid), dropped


def parse_prediction(raw: str, call: CallInput, *, temporal_mode=None,
                     temporal_decisions=None, catalog_path=None, date_catalog_path=None,
                     fact_recovery=None):
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
        facts, summary = normalize_prediction(raw_facts, call, temporal_mode=temporal_mode,
                                              temporal_decisions=temporal_decisions,
                                              catalog_path=catalog_path, date_catalog_path=date_catalog_path,
                                              fact_recovery=fact_recovery)
    except ValueError as exc:
        if not hasattr(exc, 'errors'):
            raise  # Configuration/catalog failures are not model schema errors.
        return None, schema_error(exc), None
    return facts, None, summary or None
