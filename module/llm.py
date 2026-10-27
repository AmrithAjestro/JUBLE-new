"""Shared Groq client, model fallback, and JSON-extraction helpers."""
import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
LLM_MODEL = (os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL") or "openai/gpt-oss-120b").strip()
CHAT_MODEL = (os.getenv("CHAT_MODEL") or LLM_MODEL).strip()
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

# Error fragments meaning "this model can't be used" -> try the next one
_MODEL_ERRORS = ("decommissioned", "does not exist", "not found", "unsupported model", "unavailable")

_client = None
_unavailable: set[str] = set()  # models that failed with a model error this process; skipped afterwards


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("❌ GROQ_API_KEY not found. Add it to your .env file.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def friendly_error(e: Exception) -> str:
    msg = str(e).lower()
    if "rate_limit" in msg or "rate limit" in msg:
        return "⚠️ Groq API rate limit reached. Please try again shortly."
    if "invalid_api_key" in msg or "invalid api key" in msg:
        return "❌ Invalid Groq API key. Check your .env file."
    return f"⚠️ Groq API error: {e}"


def chat(messages: list[dict], model: str | None = None,
         temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Run a chat completion, falling back to other models if the requested one is unavailable."""
    candidates = list(dict.fromkeys(m for m in (model or LLM_MODEL, *FALLBACK_MODELS) if m))
    last_err = None
    for mdl in [m for m in candidates if m not in _unavailable] or candidates:
        try:
            resp = get_client().chat.completions.create(
                model=mdl, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if any(k in str(e).lower() for k in _MODEL_ERRORS):
                _unavailable.add(mdl)
                last_err = e
                continue
            raise RuntimeError(friendly_error(e)) from e
    raise RuntimeError(
        f"❌ No available Groq model (tried {', '.join(candidates)}). "
        f"Update LLM_MODEL in .env — see https://console.groq.com/docs/models. Last error: {last_err}"
    )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.S)
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (fenced or bare). Returns None if none parses."""
    if not text:
        return None
    for candidate in (text, text.translate(_SMART_QUOTES)):
        for pattern in (_JSON_FENCE, _FIRST_OBJECT):
            m = pattern.search(candidate)
            if not m:
                continue
            try:
                data = json.loads(m.group(1) if m.groups() else m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None
