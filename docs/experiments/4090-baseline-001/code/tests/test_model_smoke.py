"""Optional real-PyTorch smoke: tiny random Qwen, no downloaded model or CUDA needed."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_ML = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "accelerate"))


@unittest.skipUnless(HAS_ML, "GPU/ML dependencies are optional for CPU contract tests")
class ModelSmokeTest(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.torch = torch
        self.model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=64, pad_token_id=0, eos_token_id=2,
            attention_dropout=0.0,
        ))
        self.rows = [
            {"input_ids": [1, 3, 4, 5, 6, 7, 2], "labels": [-100, -100, -100, -100, 6, 7, 2]},
            {"input_ids": [1, 4, 5, 8, 2], "labels": [-100, -100, -100, 8, 2]},
        ]

    def test_tail_loss_matches_full_loss_on_real_model(self):
        from train import make_batch
        self.model.eval()
        with self.torch.no_grad():
            full = self.model(**make_batch(self.rows, 0, tail_only=False)).loss
            tail = self.model(**make_batch(self.rows, 0, tail_only=True)).loss
        self.assertTrue(self.torch.allclose(full, tail, atol=1e-6), (full, tail))

    def test_checkpoint_retains_optimizer_and_can_resume(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Trainer, TrainingArguments
        from train import Rows, make_batch
        model = get_peft_model(self.model, LoraConfig(
            r=2, lora_alpha=4, target_modules="all-linear", task_type="CAUSAL_LM"))
        with tempfile.TemporaryDirectory() as temp:
            def trainer(steps):
                return Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=temp, use_cpu=True, max_steps=steps,
                        per_device_train_batch_size=1, learning_rate=1e-3,
                        gradient_accumulation_steps=2, gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={'use_reentrant': False},
                        save_strategy="steps", save_steps=1, save_total_limit=2,
                        report_to="none", disable_tqdm=True, logging_strategy="no",
                        label_names=["labels"], prediction_loss_only=True),
                    train_dataset=Rows(self.rows),
                    data_collator=lambda rows: make_batch(rows, 0, True),
                )
            first = trainer(2)
            first.train()
            checkpoint = Path(temp) / "checkpoint-2"
            self.assertTrue((checkpoint / "trainer_state.json").exists())
            self.assertTrue((checkpoint / "optimizer.pt").exists())
            second = trainer(3)
            second.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(second.state.global_step, 3)
            self.assertTrue((Path(temp) / "checkpoint-3").exists())


if __name__ == "__main__":
    unittest.main()
