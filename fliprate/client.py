"""Minimal OpenAI-compatible chat client (requests only).

Ollama:    base_url=http://localhost:11434/v1   (api key ignored)
LM Studio: base_url=http://localhost:1234/v1    (api key ignored)
"""
from __future__ import annotations
import os, time
from pathlib import Path
import requests

OLLAMA_URL = "http://localhost:11434/v1"
LMSTUDIO_URL = "http://localhost:1234/v1"


def load_dotenv() -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass
            break


class ChatClient:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = "", api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 300, timeout: float = 300.0,
                 retries: int = 3, seed: int | None = 0):
        load_dotenv()
        self.base_url = base_url.rstrip("/")
        self.model = model
        if not api_key:
            if "googleapis.com" in self.base_url:
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            elif "nvidia.com" in self.base_url:
                api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key or "not-needed"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.seed = seed

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json().get("data") or []
        return [m["id"] for m in data]

    def chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, "stream": False}
        if self.seed is not None:
            payload["seed"] = self.seed
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers=self._headers(), timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"chat failed after {self.retries} attempts: {last}")
