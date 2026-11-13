"""HTTP interface for organizer calls; model is loaded at application startup."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import Field

from case_io import prediction_row
from config import MERGED_DIR
from schema import CallInput, Fact, StrictModel

MODEL_DIR = os.getenv("MODEL_DIR", MERGED_DIR)
MAX_BATCH = int(os.getenv("MAX_BATCH", "1"))


class CallPrediction(StrictModel):
    call_id: str
    facts: list[Fact]


class Batch(StrictModel):
    calls: list[CallInput] = Field(min_length=1, max_length=32)


def create_app(backend=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.extractor is None:
            from extractor import Extractor
            app.state.extractor = Extractor(MODEL_DIR, with_schema=os.getenv("WITH_SCHEMA") == "1")
        yield

    app = FastAPI(title="CompactFactExtraction", lifespan=lifespan)
    app.state.extractor = backend

    def run(calls):
        if app.state.extractor is None:
            raise HTTPException(503, "Model is not loaded")
        try:
            results = app.state.extractor.extract(calls, MAX_BATCH)
            if len(results) != len(calls):
                raise ValueError("Backend returned a different result count")
            predictions = [prediction_row(call.model_dump(mode="json"), result)
                           for call, result in zip(calls, results)]
        except Exception as exc:
            raise HTTPException(503, "Extraction backend failed") from exc
        failures = [{"call_id": row["call_id"], "error": row["error"]}
                    for row in predictions if "error" in row]
        if failures:
            raise HTTPException(502, detail=failures)
        return predictions

    @app.post("/extract", response_model=CallPrediction)
    def extract(call: CallInput):
        return run([call])[0]

    @app.post("/extract_batch", response_model=list[CallPrediction])
    def extract_batch(batch: Batch):
        return run(batch.calls)

    @app.get("/health")
    def health():
        if app.state.extractor is None:
            raise HTTPException(503, "Model is not loaded")
        return {"status": "ok", "model": MODEL_DIR, "schema": "organizer-call-v1"}

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    return app


app = create_app()
