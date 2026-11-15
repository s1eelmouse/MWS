"""Validate organizer calls without changing labels or requiring literal ASR quotes."""
import argparse
import json

from case_io import read_jsonl, validate_gold


def find_problems(row: dict) -> list[str]:
    try:
        validate_gold(row)
        return []
    except (KeyError, TypeError, ValueError) as err:
        return [" ".join(str(err).split())[:500]]


def inspect_rows(rows: list[dict]) -> dict:
    bad, quotes, nonliteral, empty_names, spans = [], 0, 0, 0, 0
    for row in rows:
        problems = find_problems(row)
        if problems:
            bad.append({"call_id": row["call_id"], "problems": problems})
            continue
        turns = {turn["idx"]: turn["text"] for turn in row["turns"]}
        for fact in row["facts"]:
            empty_names += fact['object']['name'] == ''
            source = fact["source"]
            spans += source.get('span') is not None
            if source.get("quote"):
                quotes += 1
                nonliteral += source["quote"] not in turns[source["turn_idx"]]
    return {
        "calls": len(rows), "facts": sum(len(row.get("facts", [])) for row in rows),
        "invalid_calls": len(bad), "errors": bad,
        "nonliteral_quote_warnings": nonliteral, "nonempty_quotes": quotes,
        'empty_object_names': empty_names, 'facts_with_spans': spans,
        'contract_version': '2.1',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    args = parser.parse_args()
    report = inspect_rows(read_jsonl(args.input))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["invalid_calls"] else 0)


if __name__ == "__main__":
    main()
