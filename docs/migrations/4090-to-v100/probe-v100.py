"""Tiny real-kernel checks for V100; no model downloads, training or full evaluation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from gpu_precision import native_bf16_supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', required=True)
    args = parser.parse_args()
    report = {'status': 'failed', 'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'xgrammar')}}
    path = Path(args.json)
    if path.exists():
        raise FileExistsError(path)
    try:
        import torch
        assert torch.cuda.is_available(), 'CUDA unavailable'
        report.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability()),
                      cuda_build=torch.version.cuda, arch_list=torch.cuda.get_arch_list(),
                      native_bf16=native_bf16_supported(torch))
        assert report['cuda_build'].startswith('12.'), 'Expected a CUDA 12.x PyTorch build'
        assert report['capability'] == [7, 0], 'Expected a V100-class sm_70 GPU; review device selection'
        assert any(x in report['arch_list'] for x in ('sm_70', 'compute_70')), 'Wheel lacks sm_70; do not proceed'
        free, total = torch.cuda.mem_get_info()
        report.update(free_vram_before=free, total_vram=total)
        assert free >= 2 * 2**30, 'Wait for at least 2 GiB free before the tiny kernel probe'
        torch.cuda.set_per_process_memory_fraction(min(1.0, 2 * 2**30 / total), 0)
        with torch.inference_mode():
            a = torch.ones((128, 128), device='cuda', dtype=torch.float16)
            assert torch.isfinite(a @ a).all().item()
            q = torch.ones((1, 2, 128, 64), device='cuda', dtype=torch.float16)
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            assert torch.isfinite(out).all().item()
            report['fp16_matmul_and_sdpa'] = 'passed'
            import xgrammar as xgr
            mask = xgr.allocate_token_bitmask(1, 16)
            mask.fill_(-1)
            scores = torch.zeros((1, 16), device='cuda', dtype=torch.float32)
            xgr.apply_token_bitmask_inplace(scores, mask.to('cuda'), backend='torch_native')
            assert torch.isfinite(scores).all().item()
            report['xgrammar_torch_native_mask'] = 'passed'
            import bitsandbytes as bnb
            layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16,
                                     quant_type='nf4').to('cuda')
            assert torch.isfinite(layer(torch.ones((1, 64), device='cuda', dtype=torch.float16))).all().item()
            report['nf4_kernel'] = 'passed'
        torch.cuda.synchronize()
        report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
