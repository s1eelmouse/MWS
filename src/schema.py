"""CompactFactExtraction contract shared by training, prediction and HTTP serving."""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import MAX_TEXT_CHARS

Attribute = Literal["service_state", "tariff", "ticket_status", "action", "agreement"]
Speaker = Literal["client", "operator"]


def iso_date(value: str | None) -> str | None:
    if value is not None:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectRef(StrictModel):
    name: str = Field(min_length=1)
    id: str | None = None
    # 'other' occurs in organizer gold despite not being listed in schema.md.
    type: Literal["service", "device", "ticket", "client", "other"] | None = None


class Source(StrictModel):
    turn_idx: int = Field(ge=0, strict=True)
    span: tuple[int, int] | None = None
    quote: str | None = None

    @field_validator("span")
    @classmethod
    def valid_span(cls, value):
        if value is not None and not (0 <= value[0] < value[1]):
            raise ValueError("span must be a nonempty, nonnegative [start, end) interval")
        return value


class Fact(StrictModel):
    object: ObjectRef
    attribute: Attribute
    value: str = Field(min_length=1)
    polarity: Literal[-1, 1]
    valid_from: str | None
    valid_to: str | None
    source: Source
    speaker: Speaker | None = None
    confirmed: bool | None = None

    @field_validator("polarity", mode="before")
    @classmethod
    def strict_polarity(cls, value):
        if type(value) is not int or value not in (-1, 1):
            raise ValueError("polarity must be the integer -1 or 1")
        return value

    @field_validator("valid_from", "valid_to")
    @classmethod
    def valid_date(cls, value):
        return iso_date(value)

    @model_validator(mode="after")
    def ordered_interval(self):
        if self.valid_from and self.valid_to:
            try:
                if datetime.fromisoformat(self.valid_to.replace("Z", "+00:00")) < datetime.fromisoformat(self.valid_from.replace("Z", "+00:00")):
                    raise ValueError("valid_to precedes valid_from")
            except TypeError as exc:
                raise ValueError("date interval mixes timezone-aware and naive timestamps") from exc
        return self


class Facts(StrictModel):
    facts: list[Fact]


class Turn(StrictModel):
    idx: int = Field(ge=0, strict=True)
    speaker: Speaker
    text: str


class CallInput(StrictModel):
    call_id: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    turns: list[Turn] = Field(min_length=1)
    duration_sec: int | None = Field(default=None, ge=0)
    series: dict | None = None
    noise_level: float | None = Field(default=None, ge=0, le=1)

    @field_validator("started_at")
    @classmethod
    def valid_started_at(cls, value):
        return iso_date(value)

    @model_validator(mode="after")
    def unique_turns(self):
        indices = [turn.idx for turn in self.turns]
        if len(set(indices)) != len(indices):
            raise ValueError("turn indices must be unique")
        return self


SYSTEM = (
    "Извлеки подтверждённые факты из диалога поддержки. Ответь только JSON с ключом facts. "
    "Факт: object{name,type}, attribute, value, polarity, valid_from, valid_to, source{turn_idx}. "
    "attribute: service_state, tariff, ticket_status, action, agreement. "
    "Нормализуй названия и значения, не теряя результат действия. "
    "Предложения и догадки без подтверждения не являются фактами. "
    "Не ставь polarity=-1 только из-за слова «не»: отрицательный результат не отменяет выполненное действие. "
    "Даты ISO рассчитывай относительно started_at; закрытые состояния имеют valid_to. "
    "Указывай исходный idx реплики, не её позицию в списке. Не выдумывай факты и даты."
)
SYSTEM_WITH_SCHEMA = SYSTEM + " JSON-схема: " + json.dumps(Facts.model_json_schema(), ensure_ascii=False)


def contract_signature():
    payload = {'system': SYSTEM, 'schema': Facts.model_json_schema(), 'target_version': 1}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def check_model_contract(model_dir, with_schema=False):
    path = Path(model_dir)
    if not path.is_dir() or with_schema:
        return
    manifest = path / 'case_contract.json'
    if not manifest.is_file():
        raise ValueError('Local model lacks case_contract.json; it may use the legacy schema')
    if json.loads(manifest.read_text(encoding='utf-8')).get('signature') != contract_signature():
        raise ValueError('Model was trained with a different case contract')


def as_call(row: dict | CallInput) -> CallInput:
    """Gold labels and prediction metadata never enter model input."""
    if isinstance(row, CallInput):
        return row
    return CallInput.model_validate({k: v for k, v in row.items() if k in CallInput.model_fields})


def build_messages(call: dict | CallInput, with_schema: bool = False) -> list[dict]:
    call = as_call(call)
    if MAX_TEXT_CHARS and sum(len(t.text) for t in call.turns) > MAX_TEXT_CHARS:
        raise ValueError("Call exceeds MAX_TEXT_CHARS; refusing to truncate transcript")
    payload = call.model_dump(mode="json", exclude_none=True)
    return [
        {"role": "system", "content": SYSTEM_WITH_SCHEMA if with_schema else SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def to_answer(facts: list | dict) -> str:
    """Compact case targets exclude noisy gold quotes and account-specific IDs."""
    parsed = Facts.model_validate({"facts": facts} if isinstance(facts, list) else facts)
    result = []
    for fact in parsed.facts:
        item = {
            "object": {"name": fact.object.name}, "attribute": fact.attribute,
            "value": fact.value, "polarity": fact.polarity,
            "valid_from": fact.valid_from, "valid_to": fact.valid_to,
            "source": {"turn_idx": fact.source.turn_idx},
        }
        if fact.object.type is not None:
            item["object"]["type"] = fact.object.type
        if fact.speaker is not None:
            item["speaker"] = fact.speaker
        result.append(item)
    return json.dumps({"facts": result}, ensure_ascii=False, separators=(",", ":"))


def parse_answer(raw: str) -> Facts | None:
    raw = raw.rsplit("</think>", 1)[-1]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return Facts.model_validate_json(raw[start:end + 1])
    except ValueError:
        return None


def source_problems(facts: Facts, call: CallInput) -> list[str]:
    turns = {turn.idx: turn for turn in call.turns}
    problems = []
    for index, fact in enumerate(facts.facts):
        turn = turns.get(fact.source.turn_idx)
        if turn is None:
            problems.append(f"facts[{index}]: source turn does not exist")
        elif fact.source.span and fact.source.span[1] > len(turn.text):
            problems.append(f"facts[{index}]: source span exceeds original turn")
        elif fact.speaker is not None and fact.speaker != turn.speaker:
            problems.append(f"facts[{index}]: speaker differs from source turn")
    return problems
