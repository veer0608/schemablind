"""The transport, tested at the seam where a daily allowance runs out.

A free-tier daily quota is granted per project per model, which the refusal
itself says: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. So a key
belonging to a second project is a second day's worth of the *same* model, and
a long run can cross into it without the answers becoming incomparable. What
must not happen is a spare key being spent on a per-minute refusal, which is a
wait rather than an exhaustion.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from schemablind.llm import (
    LLMError,
    OpenAICompatibleClient,
    QuotaExhausted,
    build_client,
    spare_keys,
)

PER_DAY = json.dumps(
    {
        "error": {
            "code": 429,
            "message": (
                "You exceeded your current quota. Quota exceeded for metric: "
                "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "limit: 500"
            ),
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "violations": [
                        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                    ]
                }
            ],
        }
    }
)

PER_MINUTE = json.dumps(
    {"error": {"code": 429, "message": "rate limit reached, retry in 2.0s"}}
)

ANSWER = json.dumps(
    {
        "model": "gemini-3.5-flash-lite",
        "choices": [{"message": {"content": "SELECT 1"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
)


class Response:
    """What urlopen yields on success: a context manager that reads bytes."""

    def __init__(self, body: str) -> None:
        self._body = body.encode()
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def refusal(body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/chat/completions",
        429,
        "Too Many Requests",
        {},
        io.BytesIO(body.encode()),
    )


class Transport:
    """Serves a scripted sequence and records the key each call carried."""

    def __init__(self, *script) -> None:
        self._script = list(script)
        self.keys: list[str] = []

    def __call__(self, request, timeout=None):
        self.keys.append(request.headers.get("Authorization", ""))
        step = self._script.pop(0) if self._script else Response(ANSWER)
        if isinstance(step, BaseException):
            raise step
        return step


@pytest.fixture
def no_sleeping(monkeypatch):
    """Backoff is real seconds; a test should not pay them."""
    monkeypatch.setattr("schemablind.llm.time.sleep", lambda _seconds: None)


def client(transport, monkeypatch, **kwargs):
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    kwargs.setdefault("min_interval", 0.0)
    return OpenAICompatibleClient(
        base_url="https://example.invalid", model="gemini-3.5-flash-lite", **kwargs
    )


class TestSpareKeys:
    def test_a_spent_key_hands_over_to_the_spare(self, monkeypatch, no_sleeping):
        transport = Transport(refusal(PER_DAY), Response(ANSWER))
        llm = client(transport, monkeypatch, api_key="first", spare_keys=("second",))

        reply = llm.chat(messages=[{"role": "user", "content": "hi"}], tools=[])

        assert reply.text == "SELECT 1"
        assert transport.keys == ["Bearer first", "Bearer second"]

    def test_the_handover_does_not_spend_the_retry_budget(
        self, monkeypatch, no_sleeping
    ):
        """With one attempt allowed, the spare must still get its turn."""
        transport = Transport(refusal(PER_DAY), Response(ANSWER))
        llm = client(
            transport,
            monkeypatch,
            api_key="first",
            spare_keys=("second",),
            attempts=1,
        )

        assert llm.chat(messages=[], tools=[]).text == "SELECT 1"

    def test_every_spare_is_tried_before_the_run_stops(self, monkeypatch, no_sleeping):
        transport = Transport(
            refusal(PER_DAY), refusal(PER_DAY), Response(ANSWER)
        )
        llm = client(
            transport, monkeypatch, api_key="first", spare_keys=("second", "third")
        )

        assert llm.chat(messages=[], tools=[]).text == "SELECT 1"
        assert transport.keys[-1] == "Bearer third"

    def test_the_last_key_still_stops_the_run(self, monkeypatch, no_sleeping):
        """Without a spare, an exhausted day must abandon as it always did."""
        transport = Transport(refusal(PER_DAY))
        llm = client(transport, monkeypatch, api_key="only")

        with pytest.raises(QuotaExhausted):
            llm.chat(messages=[], tools=[])

    def test_an_exhausted_spare_stops_the_run_too(self, monkeypatch, no_sleeping):
        transport = Transport(refusal(PER_DAY), refusal(PER_DAY))
        llm = client(transport, monkeypatch, api_key="first", spare_keys=("second",))

        with pytest.raises(QuotaExhausted):
            llm.chat(messages=[], tools=[])

    def test_a_per_minute_refusal_waits_rather_than_burning_a_spare(
        self, monkeypatch, no_sleeping
    ):
        """A minute is a wait. Rotating on it would spend a day to save 2s."""
        transport = Transport(refusal(PER_MINUTE), Response(ANSWER))
        llm = client(transport, monkeypatch, api_key="first", spare_keys=("second",))

        assert llm.chat(messages=[], tools=[]).text == "SELECT 1"
        assert transport.keys == ["Bearer first", "Bearer first"]

    def test_an_ordinary_failure_is_not_a_quota_failure(self, monkeypatch, no_sleeping):
        transport = Transport(
            urllib.error.HTTPError(
                "https://example.invalid", 400, "Bad Request", {}, io.BytesIO(b"nope")
            )
        )
        llm = client(transport, monkeypatch, api_key="first", spare_keys=("second",))

        with pytest.raises(LLMError) as raised:
            llm.chat(messages=[], tools=[])
        assert not isinstance(raised.value, QuotaExhausted)
        assert transport.keys == ["Bearer first"]


class TestCollectingThem:
    def test_numbered_keys_are_read_in_order(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY_2", "second")
        monkeypatch.setenv("GEMINI_API_KEY_3", "third")

        assert spare_keys("GEMINI_API_KEY") == ("second", "third")

    def test_a_gap_in_the_numbering_is_not_a_full_stop(self, monkeypatch):
        """A deleted key should not silently hide the ones after it."""
        monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY_3", "third")

        assert spare_keys("GEMINI_API_KEY") == ("third",)

    def test_a_blank_line_in_env_is_not_a_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY_2", "   ")

        assert spare_keys("GEMINI_API_KEY") == ()

    def test_no_key_env_means_no_spares(self):
        assert spare_keys(None) == ()

    def test_a_built_client_carries_the_spares(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "first")
        monkeypatch.setenv("GEMINI_API_KEY_2", "second")

        llm = build_client("gemini", "gemini-3.5-flash-lite")

        assert llm is not None
        assert llm._keys == ["first", "second"]
