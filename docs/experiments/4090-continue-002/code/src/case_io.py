"""Strict JSONL boundaries and stable call-ID alignment."""
import json
from pathlib import Path

from schema import Facts, as_call, source_problems


def read_jsonl(path, limit=None) -> list[dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows, seen = [], set()
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be a JSON object")
                cid = row["call_id"]
                if not isinstance(cid, str) or not cid:
                    raise ValueError("call_id must be a nonempty string")
                if cid in seen:
                    raise ValueError(f"duplicate call_id: {cid}")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            seen.add(cid)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def validate_gold(row: dict):
    call = as_call(row)
    facts = Facts.model_validate({"facts": row["facts"]})
    problems = source_problems(facts, call)
    if problems:
        raise ValueError("; ".join(problems))
    return call, facts


def prediction_row(row: dict, result: dict) -> dict:
    """Failures stay explicit instead of becoming successful empty extraction."""
    call = as_call(row)
    if "error" in result:
        return {"call_id": call.call_id, "facts": [], "error": result["error"]}
    facts = Facts.model_validate(result)
    problems = source_problems(facts, call)
    if problems:
        return {"call_id": call.call_id, "facts": [], "error": "invalid_source", "details": problems}
    return {"call_id": call.call_id, **facts.model_dump(mode="json")}


def write_jsonl(path, rows, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
