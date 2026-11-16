"""Validated, call-ID-aligned wrapper around the unmodified organizer scorer."""
import hashlib
from pathlib import Path

from scoring import make_scorer
from schema import Facts, as_call, source_problems

KNOWN_ISSUES = [
    "Organizer state_accuracy may mix objects; gold vs gold is not necessarily 1.0.",
    "Organizer object-name matching can accept different objects.",
    "temporal_recall and grounding are conditional on matched facts; timestamps compare by day.",
]
ATTRIBUTES = ("service_state", "tariff", "ticket_status", "action", "agreement")
TEMPORAL_ATTRIBUTES = ("service_state", "tariff", "ticket_status", "agreement")


def _day(value):
    return value[:10] if isinstance(value, str) and value else None


def attribute_error_counts(gold_facts, pred_facts, *, scorer=None, call=None):
    """Error-type counts with the organizer matcher; diagnostics only, not official metrics.

    Dates compare by day like the official temporal_recall, sources by turn_idx like
    grounding_turn, so the counts explain those metrics instead of redefining them.
    """
    scorer = scorer or make_scorer()
    matches = scorer.match_facts(gold_facts, pred_facts)
    counts = {"matched_facts": len(matches), "missed_facts": len(gold_facts) - len(matches),
              "extra_facts": len(pred_facts) - len(matches), "valid_from_errors": 0,
              "valid_to_errors": 0, "date_errors": 0, "source_turn_errors": 0,
              "temporal_eligible_matched_facts": 0, "temporal_correct_pairs": 0,
              "temporal_date_errors": 0}
    for gold, pred, _score in matches:
        wrong_from = _day(gold.get("valid_from")) != _day(pred.get("valid_from"))
        wrong_to = _day(gold.get("valid_to")) != _day(pred.get("valid_to"))
        counts["valid_from_errors"] += wrong_from
        counts["valid_to_errors"] += wrong_to
        counts["date_errors"] += wrong_from or wrong_to
        if call is not None and scorer.temporal_eligible(gold, call):
            counts["temporal_eligible_matched_facts"] += 1
            counts["temporal_correct_pairs"] += not (wrong_from or wrong_to)
            counts["temporal_date_errors"] += wrong_from or wrong_to
        counts["source_turn_errors"] += ((pred.get("source") or {}).get("turn_idx")
                                         != gold["source"]["turn_idx"])
    return counts


def index_calls(rows):
    result = {}
    for row in rows:
        cid = row["call_id"]
        if cid in result:
            raise ValueError(f"Duplicate call_id: {cid}")
        result[cid] = row
    return result


def scorer_can_read(facts):
    """Only structural safety; invalid date intervals must still reach official scoring."""
    if not isinstance(facts, list):
        return False
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get('value'), str):
            return False
        if not isinstance(fact.get('attribute'), str) or not isinstance(fact.get('object'), dict):
            return False
        if not isinstance(fact['object'].get('name'), str):
            return False
        if not isinstance(fact.get('polarity', 1), (int, float)):
            return False
        if any(fact.get(key) is not None and not isinstance(fact[key], str)
               for key in ('valid_from', 'valid_to')):
            return False
        source = fact.get('source') or {}
        if not isinstance(source, dict):
            return False
        span = source.get('span')
        if span is not None and (not isinstance(span, (list, tuple)) or len(span) != 2
                                 or any(type(x) is not int for x in span)):
            return False
    return True


def evaluate_predictions(gold_rows, pred_rows, *, scorer_version=None, catalog_path=None, scorer=None):
    scorer = scorer or make_scorer(scorer_version, catalog_path)
    gold, pred = index_calls(gold_rows), index_calls(pred_rows)
    if not gold:
        raise ValueError("Gold file is empty")
    extra = sorted(pred.keys() - gold.keys())
    if extra:
        raise ValueError(f"Predictions contain unknown call IDs: {extra[:5]}")
    scoring, errors = {}, []
    invalid, bad_sources, missing = 0, 0, 0
    for cid, row in gold.items():
        call = as_call(row)
        Facts.model_validate({"facts": row["facts"]})
        prediction = pred.get(cid)
        failure = None
        if prediction is None:
            missing += 1
            failure = "missing_prediction"
        elif prediction.get("error"):
            failure = prediction["error"]
        else:
            try:
                parsed = Facts.model_validate({"facts": prediction["facts"]})
            except (KeyError, TypeError, ValueError):
                # Published baseline has reversed intervals. Dropping those facts
                # would silently change the official comparison and its denominator.
                invalid += 1
                if scorer_can_read(prediction.get('facts')):
                    scoring[cid] = prediction
                else:
                    scoring[cid] = {"call_id": cid, "facts": []}
                errors.append({"call_id": cid, "error": "invalid_schema"})
                continue
        if failure:
            invalid += 1
            scoring[cid] = {"call_id": cid, "facts": []}
            errors.append({"call_id": cid, "error": failure})
            continue
        # Preserve raw fields: the wrapper must not change official scoring semantics.
        scoring[cid] = prediction
        problems = source_problems(parsed, call)
        bad_sources += bool(problems)
        matches = scorer.match_facts(row["facts"], prediction["facts"])
        mismatched = len(matches) != len(row["facts"]) or len(matches) != len(prediction["facts"])
        evidence_or_time = any(
            (g.get("valid_from") or "")[:10] != (p.get("valid_from") or "")[:10]
            or (g.get("valid_to") or "")[:10] != (p.get("valid_to") or "")[:10]
            or g["source"]["turn_idx"] != p["source"]["turn_idx"]
            for g, p, _ in matches
        )
        if mismatched or evidence_or_time or problems:
            errors.append({
                "call_id": cid, "gold_facts": len(row["facts"]),
                "pred_facts": len(prediction["facts"]), "matched": len(matches),
                "source_problems": problems,
            })
    official = scorer.evaluate(gold, scoring)
    from grounding_audit import diagnose_grounding
    grounding, _ = diagnose_grounding(list(gold.values()), list(scoring.values()), scorer=scorer)
    per_attribute = {}
    for attr in ATTRIBUTES:
        filtered_gold = {cid: {**row, "facts": [f for f in row["facts"] if f["attribute"] == attr]}
                         for cid, row in gold.items()}
        filtered_pred = {cid: {"facts": [f for f in row.get("facts", []) if f.get("attribute") == attr]}
                         for cid, row in scoring.items()}
        metrics = scorer.evaluate(filtered_gold, filtered_pred)
        counts = dict.fromkeys(("matched_facts", "missed_facts", "extra_facts", "valid_from_errors",
                                "valid_to_errors", "date_errors", "source_turn_errors",
                                "temporal_eligible_matched_facts", "temporal_correct_pairs",
                                "temporal_date_errors"), 0)
        for cid, row in filtered_gold.items():
            pred_facts = filtered_pred.get(cid, {"facts": []})["facts"]
            for key, value in attribute_error_counts(row["facts"], pred_facts, scorer=scorer, call=row).items():
                counts[key] += value
        per_attribute[attr] = {
            "attribute_kind": "temporal" if attr in TEMPORAL_ATTRIBUTES else "static",
            **{k: metrics[k] for k in ("n_gold_facts", "n_pred_facts", "fact_precision", "fact_recall",
                                       "temporal_recall", "n_temporal", "state_accuracy", "n_state", "grounding_turn")},
            **counts,
        }
    known = KNOWN_ISSUES if scorer.version == '1' else [
        'Organizer v2.1 gold-vs-gold state_accuracy is not necessarily 1.0.',
        'Temporal recall is conditional on matched facts and recoverable dates; action is excluded.',
        'Official v2.1 has two known date-set.update TypeErrors; use explicitly labelled 2.1-datefix for those inputs.']
    return {
        "official": official,
        "diagnostics": {
            "invalid_or_failed_calls": invalid, "missing_calls": missing,
            "format_valid_rate": (len(gold) - invalid) / len(gold),
            "calls_with_invalid_sources": bad_sources,
        },
        "per_attribute": per_attribute,
        "grounding_diagnostics": grounding,
        **scorer.metadata(),
        "known_scorer_issues": known,
    }, errors
