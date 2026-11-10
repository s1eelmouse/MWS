"""Схема фактов и всё, что обязано совпадать при обучении и при работе сервиса.

Поля ниже — пример. Когда организаторы назовут свои поля, меняйте класс Facts:
остальные скрипты подхватят изменения сами.
"""
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import MAX_TEXT_CHARS

Category = Literal["сбой", "вопрос", "заявка", "жалоба", "другое"]
Urgency = Literal["низкая", "средняя", "высокая"]


class Facts(BaseModel):
    # Числа вместо строк и лишние пробелы в ответе модели не считаем ошибкой.
    model_config = ConfigDict(coerce_numbers_to_str=True, str_strip_whitespace=True)

    product: Optional[str] = Field(None, description="Продукт или услуга, о которой идёт речь")
    category: Category = Field(description="Тип обращения")
    problem: Optional[str] = Field(None, description="Что именно не работает или не устраивает")
    request: Optional[str] = Field(None, description="Что клиент просит сделать или объяснить")
    ids: list[str] = Field(default_factory=list, description="Номера договоров, заказов, заявок, счетов")
    error_code: Optional[str] = Field(None, description="Код или текст ошибки, если он назван")
    urgency: Urgency = Field(description="Срочность по тону и последствиям")

    @field_validator("product", "problem", "request", "error_code")
    @classmethod
    def _empty_to_none(cls, value):
        return value or None

    @field_validator("ids")
    @classmethod
    def _drop_empty_ids(cls, value):
        return [item for item in value if item]


# Короткая инструкция: её видит дообученная модель. Чем короче запрос, тем быстрее ответ.
SYSTEM = "Извлеки факты из обращения в техподдержку. Ответь только JSON."

# Длинная инструкция со схемой: для модели-учителя и для модели без дообучения.
RULES = (
    "Правила. Поля product, problem, request, error_code и элементы ids — дословные фрагменты "
    "обращения, без пересказа и исправления опечаток. Если факта в тексте нет, ставь null или "
    "пустой список. Значения category: сбой — что-то не работает; вопрос — просят объяснить; "
    "заявка — просят что-то сделать; жалоба — недовольны качеством или сроками; "
    "другое — всё остальное."
)
SYSTEM_WITH_SCHEMA = (
    f"{SYSTEM} {RULES} JSON-схема ответа: "
    f"{json.dumps(Facts.model_json_schema(), ensure_ascii=False)}"
)


def build_messages(text: str, with_schema: bool = False) -> list[dict]:
    """Запрос к модели. Один и тот же код вызывается при обучении и в сервисе."""
    system = SYSTEM_WITH_SCHEMA if with_schema else SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text[:MAX_TEXT_CHARS]},
    ]


def to_answer(facts: dict) -> str:
    """Эталонный ответ для обучения: JSON без пробелов и без пустых полей."""
    clean = Facts.model_validate(facts).model_dump(exclude_defaults=True)
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def parse_answer(raw: str) -> Optional[Facts]:
    """Разбирает ответ модели. Возвращает None, если это не JSON по схеме."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return Facts.model_validate_json(raw[start : end + 1])
    except ValueError:
        return None
