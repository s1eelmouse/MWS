"""Select native CUDA precision, excluding BF16 emulation on older devices."""


def native_bf16_supported(torch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())
