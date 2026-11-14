"""Starting a new cycle from a local adapter must be explicit, separate and hashed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from schema import contract_signature
from training_data import adapter_origin


class AdapterOriginTest(unittest.TestCase):
    def make_adapter(self, root):
        source = root / "adapter"
        source.mkdir()
        (source / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test/base", "peft_type": "LORA"}))
        (source / "case_contract.json").write_text(json.dumps({
            "signature": contract_signature()}))
        (source / "adapter_model.safetensors").write_bytes(b"original weights")
        return source

    def test_separate_output_contract_base_and_exact_weight_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_adapter(root)
            first = adapter_origin(source, root / "new", "test/base")
            (source / "adapter_model.safetensors").write_bytes(b"changed weights")
            self.assertNotEqual(first["files"],
                                adapter_origin(source, root / "new", "test/base")["files"])
            with self.assertRaises(ValueError):
                adapter_origin(source, source, "test/base")
            with self.assertRaises(ValueError):
                adapter_origin(source, source / "child", "test/base")
            with self.assertRaises(ValueError):
                adapter_origin(source, root / "new", "wrong/base")
            (source / "case_contract.json").write_text('{"signature":"wrong"}')
            with self.assertRaises(ValueError):
                adapter_origin(source, root / "new", "test/base")


if __name__ == "__main__":
    unittest.main()
