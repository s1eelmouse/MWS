"""MCP-сервер: даёт агенту (например, nanobot) инструмент extract_facts.

Нужен пакет mcp версии 2 или новее:  uv pip install "mcp>=2"
Сервис из serve.py или serve_vllm.py должен быть уже запущен.
Агент сам стартует этот файл командой:  python путь/к/src/mcp_server.py

В этом файле нельзя ничего печатать через print: по stdout идёт обмен с агентом.
"""
import os

import requests
from mcp.server import MCPServer

# 127.0.0.1, а не localhost: в Windows запрос на localhost идёт на 2 секунды дольше.
SERVICE_URL = os.getenv("EXTRACTOR_URL", "http://127.0.0.1:8000")
mcp = MCPServer("extractor")


@mcp.tool()
def extract_facts(text: str) -> dict:
    """Извлечь факты из обращения в техподдержку: продукт, проблема, запрос, номера, срочность."""
    try:
        reply = requests.post(SERVICE_URL + "/extract", json={"text": text}, timeout=60)
        reply.raise_for_status()
        return reply.json()
    except (requests.RequestException, ValueError) as err:
        return {"error": "service_unavailable", "raw": str(err)}


if __name__ == "__main__":
    mcp.run()  # без аргументов сервер общается через stdin и stdout: так его запускает агент
