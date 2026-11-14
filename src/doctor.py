"""Read-only environment report; no model downloads and no system configuration changes."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="new report file")
    args = parser.parse_args()
    report = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "xgrammar",
                 "pydantic", "fastapi"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["cuda_available"] = False
    if importlib.util.find_spec("torch"):
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_build"] = torch.version.cuda
        if report["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = props.name
            report["vram_gib"] = props.total_memory / 2**30
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
