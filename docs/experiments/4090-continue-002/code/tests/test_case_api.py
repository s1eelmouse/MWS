"""Exercise actual HTTP validation and serialization with an injected model backend."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from schema import as_call, to_answer
from serve import create_app
from test_case_schema import CALL, FACT
import json


class Backend:
    def __init__(self):
        self.calls = []

    def extract(self, calls, batch_size):
        self.calls.extend(calls)
        return [json.loads(to_answer([FACT])) for _ in calls]


class ApiTest(unittest.TestCase):
    def test_metadata_and_source_survive_http_round_trip(self):
        backend = Backend()
        with TestClient(create_app(backend)) as client:
            response = client.post("/extract", json=as_call(CALL).model_dump(mode="json"))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["call_id"], CALL["call_id"])
            self.assertEqual(response.json()["facts"][0]["source"]["turn_idx"], 7)
            self.assertEqual(backend.calls[0].started_at, CALL["started_at"])
            self.assertEqual(client.get("/health").status_code, 200)

    def test_gold_labels_cannot_leak_through_http(self):
        backend = Backend()
        with TestClient(create_app(backend)) as client:
            self.assertEqual(client.post("/extract", json=CALL).status_code, 422)
        self.assertFalse(backend.calls)

    def test_legacy_text_only_requests_are_rejected(self):
        with TestClient(create_app(Backend())) as client:
            self.assertEqual(client.post("/extract", json={"text": "привет"}).status_code, 422)

    def test_model_failures_do_not_become_successful_empty_facts(self):
        class Failing:
            def extract(self, calls, batch_size):
                return [{"error": "invalid_json"}]
        with TestClient(create_app(Failing())) as client:
            self.assertEqual(client.post("/extract", json=as_call(CALL).model_dump(mode="json")).status_code, 502)

    def test_runtime_failure_returns_service_unavailable(self):
        class Failing:
            def extract(self, calls, batch_size):
                raise RuntimeError("internal details must not leak")
        with TestClient(create_app(Failing())) as client:
            response = client.post("/extract", json=as_call(CALL).model_dump(mode="json"))
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("internal details", response.text)

    def test_batch_preserves_order_and_ids(self):
        first = as_call(CALL).model_dump(mode="json")
        second = {**copy.deepcopy(first), "call_id": "synthetic:call0001"}
        with TestClient(create_app(Backend())) as client:
            response = client.post("/extract_batch", json={"calls": [second, first]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual([r["call_id"] for r in response.json()], [second["call_id"], first["call_id"]])

    def test_wrong_source_from_backend_is_rejected(self):
        class BadSource:
            def extract(self, calls, batch_size):
                result = json.loads(to_answer([FACT]))
                result["facts"][0]["source"]["turn_idx"] = 0
                return [result]
        with TestClient(create_app(BadSource())) as client:
            self.assertEqual(client.post("/extract", json=as_call(CALL).model_dump(mode="json")).status_code, 502)


if __name__ == "__main__":
    unittest.main()
