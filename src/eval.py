"""Оценка качества и скорости.

    python src/eval.py --gold data/test.jsonl                        # сервис на 127.0.0.1:8000
    python src/eval.py --gold data/test.jsonl --workers 8             # 8 параллельных запросов
    python src/eval.py --gold data/test.jsonl --pred out/pred.jsonl   # файл от predict.py

Печатает F1 по полям, долю валидных ответов, задержку p50 и p95, обращений в секунду.
Расхождения с эталоном пишет в out/errors.jsonl: по ним видно, чему доучивать модель.

В адресе пишите 127.0.0.1, а не localhost: в Windows каждый запрос на localhost идёт
на 2 секунды дольше (сначала пробуется IPv6), и замер задержки получается неверным.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import config
from metrics import score


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ask(endpoint: str, text: str):
    """Один запрос к сервису: (ответ, секунды). Сбой сети — тоже ответ, а не падение скрипта."""
    start = time.perf_counter()
    try:
        reply = requests.post(endpoint, json={"text": text}, timeout=120)
        reply.raise_for_status()
        pred = reply.json()
    except (requests.RequestException, ValueError) as err:
        pred = {"error": "request_failed", "raw": str(err)}
    return pred, time.perf_counter() - start


def facts_of(pred: dict) -> dict:
    """Ответ без валидных фактов считается пустым: все эталонные факты пропущены."""
    return {} if "error" in pred else pred


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="data/test.jsonl", help="файл с эталонной разметкой")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="адрес сервиса")
    parser.add_argument("--pred", help="готовые ответы от predict.py вместо запросов к сервису")
    parser.add_argument("--workers", type=int, default=1, help="сколько запросов слать параллельно")
    parser.add_argument("--errors", default="out/errors.jsonl", help="куда записать расхождения")
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    if not gold:
        raise SystemExit(f"В файле {args.gold} нет строк")

    latencies, wall = [], None
    if args.pred:
        rows = read_jsonl(args.pred)
        if len(rows) != len(gold):
            raise SystemExit(f"В {args.pred} строк: {len(rows)}, а в {args.gold}: {len(gold)}")
        preds = [row["facts"] if "facts" in row else {"error": row.get("error", "no_facts")} for row in rows]
    else:
        endpoint = args.url.rstrip("/") + "/extract"
        ask(endpoint, gold[0]["text"])  # прогрев: первый запрос в замеры не входит
        start = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda row: ask(endpoint, row["text"]), gold))
        wall = time.perf_counter() - start
        preds = [pred for pred, _ in results]
        latencies = sorted(seconds for _, seconds in results)

    failed = sum("error" in pred for pred in preds)
    result = score([(row["facts"], facts_of(pred)) for row, pred in zip(gold, preds)])
    micro = result["micro"]
    print(f"Обращений: {len(gold)}, без валидного ответа: {failed} "
          f"(валидный JSON: {100 * (1 - failed / len(gold)):.1f}%)")
    print(f"F1={micro['f1']:.3f}  точность={micro['precision']:.3f}  полнота={micro['recall']:.3f}")
    print(f"{'поле':<12}{'точность':>10}{'полнота':>10}{'F1':>8}{'эталонов':>10}")
    for name, field in result["fields"].items():
        print(f"{name:<12}{field['precision']:>10.3f}{field['recall']:>10.3f}"
              f"{field['f1']:>8.3f}{field['tp'] + field['fn']:>10}")

    if latencies:
        p50 = statistics.median(latencies)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"Задержка при параллельных запросах ({args.workers}): "
              f"p50={p50 * 1000:.0f} мс, p95={p95 * 1000:.0f} мс")
        print(f"Пропускная способность: {len(gold) / wall:.1f} обращений в секунду")

    model_dir = Path(config.MERGED_DIR)
    if model_dir.is_dir():
        size = sum(file.stat().st_size for file in model_dir.rglob("*") if file.is_file())
        print(f"Размер модели в {model_dir}: {size / 1e9:.2f} ГБ")

    wrong = []
    for row, pred in zip(gold, preds):
        one = score([(row["facts"], facts_of(pred))])["micro"]
        if one["fp"] or one["fn"] or "error" in pred:
            wrong.append({"text": row["text"], "gold": row["facts"], "pred": pred})
    Path(args.errors).parent.mkdir(parents=True, exist_ok=True)
    with open(args.errors, "w", encoding="utf-8") as f:
        for item in wrong:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Обращений с расхождениями: {len(wrong)}, записаны в {args.errors}")


if __name__ == "__main__":
    main()
