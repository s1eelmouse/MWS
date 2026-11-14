"""Validated, call-ID-aligned wrapper around the unmodified organizer scorer."""
import hashlib
from pathlib import Path

from official_case_metrics import evaluate, match_facts
from schema import Facts, as_call, source_problems

KNOWN_ISSUES = [
    "Organizer state_accuracy may mix objects; gold vs gold is not necessarily 1.0.",
    "Organizer object-name matching can accept different objects.",
    "temporal_recall and grounding are conditional on matched facts; timestamps compare by day.",
]


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


def evaluate_predictions(gold_rows, pred_rows):
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
        matches = match_facts(row["facts"], prediction["facts"])
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
    official = evaluate(gold, scoring)
    per_attribute = {}
    for attr in ("service_state", "tariff", "ticket_status", "action", "agreement"):
        filtered_gold = {cid: {"facts": [f for f in row["facts"] if f["attribute"] == attr]}
                         for cid, row in gold.items()}
        filtered_pred = {cid: {"facts": [f for f in row["facts"] if f["attribute"] == attr]}
                         for cid, row in scoring.items()}
        metrics = evaluate(filtered_gold, filtered_pred)
        per_attribute[attr] = {k: metrics[k] for k in
                               ("n_gold_facts", "n_pred_facts", "fact_precision", "fact_recall")}
    scorer_path = Path(__file__).with_name("official_case_metrics.py")
    return {
        "official": official,
        "diagnostics": {
            "invalid_or_failed_calls": invalid, "missing_calls": missing,
            "format_valid_rate": (len(gold) - invalid) / len(gold),
            "calls_with_invalid_sources": bad_sources,
        },
        "per_attribute": per_attribute,
        "scorer_sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
        "known_scorer_issues": KNOWN_ISSUES,
    }, errors
