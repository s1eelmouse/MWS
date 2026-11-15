"""Вызов модели по API в формате OpenAI обычным HTTP-запросом.

Подходит и для модели-учителя, и для своего сервера vLLM: формат запроса у них общий.
"""
import time

import requests


def chat(base_url: str, api_key: str, model: str, messages: list, retries: int = 3,
         timeout: int = 120, **params) -> str:
    """Возвращает текст ответа модели.

    base_url — адрес с версией в конце, например http://localhost:8001/v1.
    params — любые поля запроса: temperature, max_tokens, response_format и другие.
    Временные сбои (сеть, 429, 5xx) повторяются с паузой; ошибки 4xx сразу поднимаются.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": messages, **params}
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(retries):
        try:
            reply = requests.post(url, json=body, headers=headers, timeout=timeout)
            if reply.status_code in (408, 429) or reply.status_code >= 500:
                raise requests.ConnectionError(f"HTTP {reply.status_code}: {reply.text[:200]}")
            reply.raise_for_status()  # 4xx: ошибка в запросе или ключе, повторять бессмысленно
            return reply.json()["choices"][0]["message"]["content"] or ""
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("retries должен быть больше нуля")
