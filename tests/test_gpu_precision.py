import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gpu_precision import native_bf16_supported


class PrecisionTest(unittest.TestCase):
    def test_volta_emulated_bf16_is_not_selected(self):
        cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (7, 0),
                               is_bf16_supported=Mock(return_value=True))
        self.assertFalse(native_bf16_supported(SimpleNamespace(cuda=cuda)))
        cuda.is_bf16_supported.assert_not_called()

    def test_modern_gpu_requests_native_support(self):
        cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 9),
                               is_bf16_supported=Mock(return_value=True))
        self.assertTrue(native_bf16_supported(SimpleNamespace(cuda=cuda)))
        cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)

    def test_cpu_and_legacy_api(self):
        self.assertFalse(native_bf16_supported(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))))
        cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 0),
                               is_bf16_supported=lambda: True)
        self.assertTrue(native_bf16_supported(SimpleNamespace(cuda=cuda)))
