"""Static container checks; no Docker daemon or model weights are required."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ignored_entries():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


class DockerAssetsTest(unittest.TestCase):
    def test_build_context_excludes_gold_weights_and_secrets(self):
        ignored = ignored_entries()
        for entry in ("data", "case_data", "out", ".env", ".venv", ".git"):
            self.assertIn(entry, ignored)

    def test_image_serves_the_organizer_http_contract(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('"serve:app"', dockerfile)
        self.assertIn('"serve_vllm:app"', dockerfile)
        self.assertIn("/health", dockerfile)
        self.assertRegex(dockerfile, r"(?m)^USER 10001:10001$")
        self.assertRegex(dockerfile, r"(?m)^HEALTHCHECK ")

    def test_compose_mounts_the_model_read_only(self):
        for name in ("docker-compose.yml", "docker-compose.vllm.yml"):
            compose = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("read_only: true", compose)
            self.assertIn("create_host_path: false", compose)
            self.assertIn("${MODEL_PATH:-./out/merged}", compose)

    def test_serving_requirements_exclude_the_training_stack(self):
        requirements = (ROOT / "requirements-serve.txt").read_text(encoding="utf-8")
        for package in ("peft", "bitsandbytes"):
            self.assertNotIn(package, requirements)


if __name__ == "__main__":
    unittest.main()
