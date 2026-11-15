"""Make a deterministic copy of val with extra ASR-like noise for robustness checks.

Stand-in for the hidden adversarial subset: the original file is never modified.
Only turn texts change; gold facts, values, dates and source turn_idx stay as they are.
Character spans and quotes no longer point into the changed text, so they are set to
null (grounding_turn stays comparable, grounding_span does not apply).
"""
import argparse
import copy
import difflib
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

from case_io import read_jsonl, validate_gold

# Frequent Russian ASR confusions: vowel reduction, voicing, similar sounds.
CONFUSIONS = {
    "о": "а", "а": "о", "е": "и", "и": "е", "я": "е", "ё": "е", "э": "е",
    "т": "д", "д": "т", "с": "з", "з": "с", "б": "п", "п": "б", "в": "ф", "ф": "в",
    "г": "к", "к": "г", "ж": "ш", "ш": "ж", "щ": "ш", "ц": "с", "ы": "и",
}
WORD = re.compile(r"\w+", re.UNICODE)
OPERATIONS = ("confuse", "drop_char", "swap", "repeat", "drop_word", "merge")


def corrupt_word(word: str, rng: random.Random) -> str:
    operation = rng.choice(OPERATIONS)
    if operation == "drop_word":
        return ""
    if operation == "merge" or len(word) < 3:
        return word + "\x00"  # marker: glue to the next word
    position = rng.randrange(len(word))
    if operation == "confuse":
        letter = word[position].lower()
        replacement = CONFUSIONS.get(letter)
        if replacement is None:
            return word[:position] + word[position + 1:]
        if word[position].isupper():
            replacement = replacement.upper()
        return word[:position] + replacement + word[position + 1:]
    if operation == "drop_char":
        return word[:position] + word[position + 1:]
    if operation == "swap" and position < len(word) - 1:
        return word[:position] + word[position + 1] + word[position] + word[position + 2:]
    return word[:position] + word[position] + word[position:]


def corrupt_text(text: str, level: float, rng: random.Random) -> str:
    if level <= 0:
        return text
    pieces, last = [], 0
    for match in WORD.finditer(text):
        pieces.append(text[last:match.start()])
        word = match.group()
        pieces.append(corrupt_word(word, rng) if rng.random() < level else word)
        last = match.end()
    pieces.append(text[last:])
    result = "".join(pieces)
    result = re.sub(r"\x00\s*", "", result)
    result = re.sub(r"[ \t]{2,}", " ", result).strip()
    words = result.split(" ")
    # Cut-off utterance, as in the organizer corpus: "…, а потом —". At least 60 % of
    # the words stay, so the check measures noise rather than deleted evidence.
    if len(words) > 4 and rng.random() < level / 2:
        keep = rng.randrange(max(2, int(len(words) * .6)), len(words))
        result = " ".join(words[:keep]) + " —"
    return result or "—"


def make_noisy(rows: list[dict], level: float, seed: int) -> list[dict]:
    noisy = []
    for row in rows:
        row = copy.deepcopy(row)
        for turn in row["turns"]:
            rng = random.Random(f"{seed}:{row['call_id']}:{turn['idx']}")
            turn["text"] = corrupt_text(turn["text"], level, rng)
        for fact in row.get("facts", []):
            fact["source"] = {**fact["source"], "span": None, "quote": None}
        if row.get("noise_level") is not None or level > 0:
            row["noise_level"] = round(min(1.0, (row.get("noise_level") or 0) + level), 4)
        validate_gold(row)
        noisy.append(row)
    return noisy


def changed_word_share(original: list[dict], noisy: list[dict]) -> float:
    """Share of original words that did not survive unchanged, by sequence alignment."""
    total = kept = 0
    for a, b in zip(original, noisy):
        for ta, tb in zip(a["turns"], b["turns"]):
            words_a, words_b = ta["text"].split(), tb["text"].split()
            total += len(words_a)
            matcher = difflib.SequenceMatcher(a=words_a, b=words_b, autojunk=False)
            kept += sum(block.size for block in matcher.get_matching_blocks())
    return 1 - kept / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from config import CASE_DATA_DIR
    parser.add_argument("--input", default=str(Path(CASE_DATA_DIR) / 'val.jsonl'))
    parser.add_argument("--output", required=True)
    parser.add_argument("--level", type=float, default=0.15,
                        help="share of words to corrupt, added on top of the existing noise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.level <= 1:
        parser.error("--level must be within [0, 1]")
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if source == output:
        parser.error("The original file must stay untouched; choose another --output")
    if output.exists() or output.with_suffix('.manifest.json').exists():
        parser.error('Output or manifest exists; choose a fresh output')
    catalog, target_catalog = source.with_name('catalog.json'), output.with_name('catalog.json')
    if catalog.is_file() and target_catalog.exists() and catalog.read_bytes() != target_catalog.read_bytes():
        parser.error('Output directory contains a different catalog; choose a fresh directory')
    rows = read_jsonl(source, args.limit)
    noisy = make_noisy(rows, args.level, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in noisy:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if catalog.is_file() and not target_catalog.exists():
        shutil.copyfile(catalog, target_catalog)
    manifest = {
        "input": str(args.input), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "calls": len(noisy), "level": args.level, "seed": args.seed, "limit": args.limit,
        "changed_word_share": round(changed_word_share(rows, noisy), 4),
        "note": "Turn texts only; facts unchanged; source span/quote set to null.",
        'catalog_next_to_output': target_catalog.is_file(),
        'catalog_sha256': hashlib.sha256(target_catalog.read_bytes()).hexdigest() if target_catalog.is_file() else None,
    }
    with output.with_suffix(".manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
