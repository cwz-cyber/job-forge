from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .prompts import PromptPackage


class ModelError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete_json(self, prompt: PromptPackage) -> dict[str, Any]: ...


@dataclass(slots=True)
class OpenAICompatibleClient:
    """Minimal dependency-free adapter for Chat Completions compatible endpoints."""

    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0

    def complete_json(self, prompt: PromptPackage) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt.system},
                {
                    "role": "user",
                    "content": (
                        prompt.user
                        + "\n\n输出 JSON Schema：\n"
                        + json.dumps(prompt.schema_hint, ensure_ascii=False)
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelError(f"模型请求失败：{exc}") from exc

        try:
            content = raw["choices"][0]["message"]["content"]
            if isinstance(content, dict):
                return content
            return _parse_json_text(str(content))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelError("模型返回格式不符合 Chat Completions JSON 约定") from exc


class ScriptedLLM:
    """Offline test double that returns preconfigured responses in order."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.prompts: list[PromptPackage] = []

    def complete_json(self, prompt: PromptPackage) -> dict[str, Any]:
        self.prompts.append(prompt)
        if not self.responses:
            raise ModelError("ScriptedLLM 没有剩余响应")
        return self.responses.pop(0)


def _parse_json_text(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected object", text, 0)
    return parsed

