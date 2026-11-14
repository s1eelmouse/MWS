"""Run tests, training and three inference measurements into an exclusive experiment directory."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from measurements import environment, write_json

ROOT = Path(__file__).resolve().parents[1]


def stage(directory, name, argv):
    start = time.perf_counter()
    print(f"Starting {name}: {' '.join(argv)}", flush=True)
    code = None
    error = None
    try:
        with (directory / f"{name}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (directory / "stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "argv": argv, "exit_code": code,
                                     "error": error, "seconds": time.perf_counter() - start}) + "\n")
    if code:
        raise RuntimeError(f"Stage {name} failed (exit {code}); inspect {name}.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", help="existing merged model: skip training and merge")
    parser.add_argument("--train", default=str(ROOT / "data/organizer/train.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "data/organizer/val.jsonl"))
    parser.add_argument("--baseline-pred", default=str(ROOT / "data/organizer/baseline_pred.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--modes", nargs="+", default=["cuda:1", "cuda:4", "cpu:1"])
    args = parser.parse_args()
    if len(args.modes) < 3 or len(set(args.modes)) != len(args.modes):
        parser.error("Provide at least three distinct device:batch configurations")
    for mode in args.modes:
        device, sep, batch = mode.partition(":")
        if device not in ("cpu", "cuda") or not sep or not batch.isdigit() or int(batch) < 1:
            parser.error("Modes must be cpu:N or cuda:N with N > 0")
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    status = "failed"
    error = None
    py = [sys.executable, "-u"]
    def script(name, *argv):
        return py + [str(ROOT / "src" / (name + ".py")), *map(str, argv)]
    try:
        write_json(directory / "environment.json", environment())
        write_json(directory / "settings.json", {
            **vars(args), "environment": {key: os.getenv(key) for key in
                ("BASE_MODEL", "MAX_INPUT_TOKENS", "MAX_NEW_TOKENS", "MAX_SEQUENCE_TOKENS", "PROMPT_LOOKUP_TOKENS")}})
        stage(directory, "tests", py + ["-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"])
        stage(directory, "compile", py + ["-m", "compileall", "-q", str(ROOT / "src"), str(ROOT / "tests")])
        stage(directory, "check-data", script("train", "--check-data", "--train", args.train, "--val", args.val))
        model = args.model
        if not model:
            extra = ["--limit", str(args.train_limit)] if args.train_limit else []
            stage(directory, "train", script("train", "--train", args.train, "--val", args.val,
                  "--epochs", args.epochs, "--batch-size", 1, "--save-steps", 50,
                  "--max-steps", args.max_steps, "--output-dir", directory / "adapter", *extra))
            model = str(directory / "merged")
            stage(directory, "merge", script("merge", "--adapter-dir", directory / "adapter", "--output", model))
        reports = []
        for mode in args.modes:
            device, batch = mode.split(":")
            name = f"{device}-batch{batch}"
            destination = directory / "benchmark" / name
            extra = ["--limit", str(args.eval_limit)] if args.eval_limit else []
            stage(directory, name, script("benchmark", "--input", args.val, "--model", model,
                  "--device", device, "--batch-size", batch, "--output-dir", destination, *extra))
            reports.append(str(destination / "report.json"))
        stage(directory, "compare", script("compare_runs", "--reports", *reports,
              "--gold", args.val, "--baseline-pred", args.baseline_pred,
              "--output-dir", directory / "comparison"))
        status = "complete"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(directory / "status.json", {"status": status, "error": error})


if __name__ == "__main__":
    main()
