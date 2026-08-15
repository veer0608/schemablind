"""The model seam, with tool calling.

Lifted from moneytrail's, which is deliberate -- the two projects should tell
one story about how a model gets called -- and extended in the one way this
project needs: the agent holds a conversation and calls tools, where that one
asked a single question and got a single answer back.

Everything here that looks defensive was paid for once already:

  the client names itself      Cloudflare answers "Python-urllib" with a 403
                               and "error code: 1010", which never reaches the
                               API and reads exactly like an auth failure
  a daily 429 is its own error the limit that binds on a free tier is tokens
                               per day, it appears in NO header, and retrying
                               one burns an hour to fail anyway
  pacing reads the headers     x-ratelimit-remaining-tokens is the per-minute
                               bucket; pacing on request count overruns it
  .env is decoded, not assumed PowerShell writes UTF-16, and read as UTF-8 the
                               key silently does not load -- indistinguishable
                               from having no key at all
  unpriced means None          never 0.0, or an unpriced model shows up free
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

PROVIDER_ENV = "SCHEMABLIND_PROVIDER"
MODEL_ENV = "SCHEMABLIND_MODEL"
USER_AGENT = "schemablind/0.1"


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str | None
    default_model: str


PROVIDERS: dict[str, Provider] = {
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-20b"),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "gemini": Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "gemini-3.7-flash",
    ),
    "together": Provider(
        "together", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    "openrouter": Provider(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct",
    ),
    "ollama": Provider("ollama", "http://localhost:11434/v1", None, "llama3.1"),
}

#: USD per million tokens, (prompt, completion). Verified against the provider's
#: own docs on the date below, because a price written from memory is a
#: confident wrong number in a column labelled cost.
PRICES_READ_ON = "2026-08-13"
PRICES: dict[str, tuple[float, float]] = {
    # groq -- console.groq.com/docs/models
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    # openai -- openai.com/api/pricing
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # gemini -- ai.google.dev/pricing. gemini-2.0-flash and 2.5-flash were
    # retired; the API returns 404 for them. Current models are unpriced
    # here on purpose until the figures are checked, so cost reports as
    # unknown rather than as a number nobody verified.
}


@dataclass(frozen=True)
class Usage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    latency_ms: float

    @property
    def priced(self) -> bool:
        return self.cost_usd is not None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    #: The assistant message verbatim, to append to history for the next turn.
    message: dict = field(default_factory=dict)
    usage: Usage | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    model: str

    def chat(
        self, *, messages: list[dict], tools: list[dict], force: str | None = None
    ) -> Reply: ...


class LLMError(RuntimeError):
    """A call that did not come back with an answer."""


class QuotaExhausted(LLMError):
    """The daily allowance is gone; waiting inside this run will not help.

    Separate from an ordinary rate limit because the two want opposite
    handling. A per-minute limit is worth sleeping through. A per-day one is
    not -- and every question left unasked after it scores zero, which a
    scorecard cannot tell apart from a model getting them wrong.
    """


#: Every spelling of "you are out for today" seen in the wild. Groq writes
#: "tokens per day (TPD)"; Gemini writes the quota id
#: "GenerateRequestsPerDayPerProjectPerModel-FreeTier", with no spaces, which
#: the space-separated pattern missed -- so a 60-question run spent every
#: single question on four retries of a limit that was never going to lift.
_DAILY_LIMIT = re.compile(r"per[ _-]?day|\bTPD\b|\bRPD\b", re.I)
_DURATION = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$"
)


def parse_duration(text: str) -> float | None:
    """Seconds from the shapes these headers use: 185ms, 6.5s, 1h50m52.8s."""
    match = _DURATION.match(text.strip())
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = (float(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def price_of(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    rates = PRICES.get(model)
    if rates is None:
        return None
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1e6


def as_tool_schema(tools: list[dict]) -> list[dict]:
    """Wrap plain definitions in the envelope the chat API expects."""
    return [{"type": "function", "function": tool} for tool in tools]


class OpenAICompatibleClient:
    """One client for every provider that speaks /chat/completions."""

    RETRY_ON = frozenset({408, 409, 429, 500, 502, 503, 504})
    HEADROOM_TOKENS = 2500

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 120.0,
        attempts: int = 4,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._attempts = attempts
        self._temperature = temperature
        #: Rate-limit headers from the previous call. Pacing happens before the
        #: next request rather than after the last one, so that the latency this
        #: reports is the model's response time and not this client's own sleep.
        #: A latency column measuring your own throttling is a wrong number.
        self._last_headers = None

    def chat(
        self, *, messages: list[dict], tools: list[dict], force: str | None = None
    ) -> Reply:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = as_tool_schema(tools)
            # Naming a function makes the API require that call rather than
            # merely offer it. Asking a model in words to submit its answer is
            # a request it is free to decline -- and on the first live run it
            # declined on every single question.
            payload["tool_choice"] = (
                {"type": "function", "function": {"name": force}} if force else "auto"
            )

        # Pacing happens out here, before the clock starts. Inside the timed
        # region -- which is where it was, and where moving it to the top of
        # _post did not get it out of -- the reported latency is this client's
        # own sleep rather than the model's response time.
        if self._last_headers is not None:
            self._pace(self._last_headers)
            self._last_headers = None

        started = time.perf_counter()
        body = self._post("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"no message in response: {str(body)[:300]}") from exc

        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=function.get("name") or "",
                    arguments=_arguments(function.get("arguments")),
                )
            )

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        served = body.get("model") or self.model
        return Reply(
            text=message.get("content") or "",
            tool_calls=tuple(calls),
            finish_reason=choice.get("finish_reason") or "stop",
            message=message,
            usage=Usage(
                model=served,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=price_of(served, prompt_tokens, completion_tokens),
                latency_ms=latency_ms,
            ),
        )

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(payload).encode()
        last = "no attempt was made"
        for attempt in range(1, self._attempts + 1):
            request = urllib.request.Request(
                f"{self.base_url}{path}", data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = json.loads(response.read().decode())
                    self._last_headers = response.headers
                    return body
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:1200]
                last = f"HTTP {exc.code}: {detail}"
                if exc.code == 429 and _DAILY_LIMIT.search(detail):
                    raise QuotaExhausted(last) from exc
                if exc.code not in self.RETRY_ON or attempt == self._attempts:
                    raise LLMError(last) from exc
                self._wait(exc, attempt)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt == self._attempts:
                    raise LLMError(last) from exc
                self._wait(None, attempt)
        raise LLMError(last)

    def _pace(self, headers) -> None:
        """Wait for the token bucket before it empties, not after."""
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = headers.get("x-ratelimit-reset-tokens")
        if remaining is None or reset is None:
            return
        try:
            left = float(remaining)
        except ValueError:
            return
        if left > self.HEADROOM_TOKENS:
            return
        delay = parse_duration(reset)
        if delay:
            time.sleep(min(delay + 0.2, 65.0))

    def _wait(self, exc: urllib.error.HTTPError | None, attempt: int) -> None:
        delay = min(2.0 ** (attempt - 1), 30.0)
        if exc is not None and exc.headers:
            header = exc.headers.get("Retry-After")
            if header:
                try:
                    delay = min(float(header), 60.0)
                except ValueError:
                    pass
        time.sleep(delay)


def _arguments(raw) -> dict:
    """Tool arguments arrive as a JSON string, and sometimes as broken JSON."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)


def _decode(raw: bytes) -> str:
    for bom, encoding in _BOMS:
        if raw.startswith(bom):
            return raw.decode(encoding, errors="replace")
    return raw.decode("utf-8", errors="replace")


def load_dotenv(root: Path | None = None) -> None:
    """Read `.env` without overwriting what is already set, whatever wrote it."""
    target = (root or Path.cwd()) / ".env"
    if not target.is_file():
        return
    try:
        text = _decode(target.read_bytes())
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def resolve(provider: str | None = None, model: str | None = None):
    name = (provider or os.environ.get(PROVIDER_ENV) or "").strip().lower()
    if not name:
        for candidate in PROVIDERS.values():
            if candidate.key_env and os.environ.get(candidate.key_env):
                name = candidate.name
                break
    if name not in PROVIDERS:
        return None
    chosen = PROVIDERS[name]
    return chosen, (model or os.environ.get(MODEL_ENV) or chosen.default_model)


def build_client(provider: str | None = None, model: str | None = None, **kwargs):
    """A client, or None when there is no key -- never one that cannot work."""
    load_dotenv()
    resolved = resolve(provider, model)
    if resolved is None:
        return None
    chosen, model_id = resolved
    api_key = os.environ.get(chosen.key_env) if chosen.key_env else None
    if chosen.key_env and not api_key:
        return None
    return OpenAICompatibleClient(
        base_url=chosen.base_url, api_key=api_key, model=model_id, **kwargs
    )
