"""Minimal OpenAI-compatible chat client (requests only).

Ollama:    base_url=http://localhost:11434/v1   (api key ignored)
LM Studio: base_url=http://localhost:1234/v1    (api key ignored)
"""
from __future__ import annotations
import os, re, time
from pathlib import Path
import requests

OLLAMA_URL = "http://localhost:11434/v1"
LMSTUDIO_URL = "http://localhost:1234/v1"


def _extract_retry_delay(r: requests.Response | None) -> float | None:
    if r is None:
        return None
    ra = r.headers.get("Retry-After")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    try:
        data = r.json()
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            err = data.get("error", {})
            for detail in err.get("details", []):
                delay_str = detail.get("retryDelay")
                if delay_str and delay_str.endswith("s"):
                    return float(delay_str[:-1]) + 1.0
            msg = err.get("message", "")
            m = re.search(r"(?:retry|try again) in ([\d\.]+)s", msg, re.I)
            if m:
                return float(m.group(1)) + 1.0
    except Exception:
        pass
    return None


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
                 retries: int = 6, seed: int | None = 0):
        load_dotenv()
        self.base_url = base_url.rstrip("/")
        self.model = model
        if not api_key:
            if "googleapis.com" in self.base_url:
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            elif "nvidia.com" in self.base_url:
                api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
            elif "groq.com" in self.base_url:
                api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key or "not-needed"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.seed = seed
        # Gemini free tier has a strict 15 RPM limit (pace to ~14.6 RPM):
        self.min_interval = 4.1 if "googleapis.com" in self.base_url else 0.0
        self._last_call = 0.0

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
        if self.seed is not None and "googleapis.com" not in self.base_url:
            payload["seed"] = self.seed
        last = None
        for attempt in range(self.retries):
            if self.min_interval > 0:
                elapsed = time.time() - self._last_call
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers=self._headers(), timeout=self.timeout)
                self._last_call = time.time()
                if r.status_code == 429:
                    delay = _extract_retry_delay(r) or (15.0 * (attempt + 1))
                    print(f"\n[Rate limited (429): waiting {delay:.1f}s before retry...]", flush=True)
                    time.sleep(delay)
                    last = f"HTTP 429: {r.text}"
                    continue
                if r.status_code >= 400:
                    try:
                        err_detail = r.json()
                    except Exception:
                        err_detail = r.text
                    last = f"HTTP {r.status_code}: {err_detail}"
                    r.raise_for_status()
                msg_obj = r.json()["choices"][0]["message"]
                return msg_obj.get("content") or msg_obj.get("reasoning_content") or ""
            except Exception as e:  # noqa: BLE001
                if last is None:
                    last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"chat failed after {self.retries} attempts: {last}")
