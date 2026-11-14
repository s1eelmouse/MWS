"""Local measurement helpers. No model prompts, credentials or dataset contents logged."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import threading
import time
from pathlib import Path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def model_artifacts(model):
    """Hash local weights outside the measured interval without loading them into RAM."""
    root = Path(model)
    if not root.is_dir():
        return None
    result = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".model"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def percentile(values, quantile):
    """Nearest-rank percentile, explicitly shared by all local benchmark modes."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def distribution(values):
    known = [v for v in values if v is not None]
    return {"count": len(known), "unknown": len(values) - len(known),
            "p50": percentile(known, .5), "p95": percentile(known, .95),
            "max": max(known) if known else None}


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    import psutil
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in
                    Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
    root = Path(__file__).resolve().parents[1]
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": cpu, "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total, "packages": packages,
            "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                   if d.metadata.get("Name")},
            "git_commit": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]) or os.getenv("MWS_CODE_REVISION"),
            "git_status": command_output(["git", "-C", str(root), "status", "--porcelain"]),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root / "src").glob("*.py"))},
            "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                          "--format=csv"]),
            "image_reference": os.getenv("MWS_IMAGE_REFERENCE")}


class MemorySampler:
    """Sample process RSS (not machine RAM); CUDA allocator peaks come from PyTorch."""
    def __init__(self, path, interval=.05):
        import psutil
        self.process = psutil.Process()
        self.path, self.interval = Path(path), interval
        self.stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self.error = None

    def sample(self):
        rss = self.process.memory_info().rss
        self.peak = max(self.peak, rss)
        self.samples += 1
        self.writer.writerow([time.perf_counter() - self.start, rss])

    def loop(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except Exception as exc:
            self.error = str(exc)

    def __enter__(self):
        self.stream = self.path.open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["elapsed_seconds", "process_rss_bytes"])
        self.start = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        try:
            self.sample()
        finally:
            self.stream.close()

    def report(self):
        return {"process_rss_sampled_peak_bytes": self.peak, "rss_sample_interval_seconds": self.interval,
                "rss_samples": self.samples, "sampling_error": self.error}
