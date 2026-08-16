"""The checkpoint exists to survive the run not reaching its own end.

So these tests are mostly about the ugly middles: a run killed by the daily
token cap, a file with a half-written last line, a scorer that changed since the
answers were recorded.
"""

from __future__ import annotations

import json

import pytest

from evals.checkpoint import VERSION, Checkpoint, key
from evals.dataset import toy
from evals.runner import ORACLE, mute_solver, oracle_solver, score
from schemablind.agent import Transcript
from schemablind.llm import QuotaExhausted, Usage


@pytest.fixture(scope="module")
def toy_set():
    return toy()


def transcript_for(sql: str | None = "SELECT 1", cost: float | None = 0.002):
    return Transcript(
        sql=sql,
        turns=3,
        tool_calls=[("list_tables", {}), ("run_sql_readonly", {"sql": "SELECT 1"})],
        usage=[
            Usage(
                model="openai/gpt-oss-120b",
                prompt_tokens=800,
                completion_tokens=120,
                cost_usd=cost,
                latency_ms=450.0,
            )
        ],
        repairs=1,
        nudges=0,
        last_query="SELECT 1",
    )


class CountingSolver:
    """A solver that answers from gold, and can run out of allowance."""

    def __init__(self, quota: int | None = None):
        self.calls = 0
        self.quota = quota
        self.asked = []

    def __call__(self, question, sandbox):
        if self.quota is not None and self.calls >= self.quota:
            raise QuotaExhausted("daily allowance gone")
        self.calls += 1
        self.asked.append(question.question_id)
        return Transcript(
            sql=question.gold_sql,
            turns=2,
            usage=[
                Usage(
                    model="test",
                    prompt_tokens=100,
                    completion_tokens=10,
                    cost_usd=0.001,
                    latency_ms=10.0,
                )
            ],
        )


class TestItRemembersWhatWasPaidFor:
    def test_a_second_run_does_not_ask_the_model_again(self, toy_set, tmp_path):
        questions, databases = toy_set
        cache = Checkpoint(tmp_path / "run.jsonl").load()

        first = CountingSolver()
        score("agent", first, questions, databases, cache=cache)

        second = CountingSolver()
        reloaded = Checkpoint(tmp_path / "run.jsonl").load()
        card = score("agent", second, questions, databases, cache=reloaded)

        assert first.calls == len(questions)
        assert second.calls == 0, "the model was asked again for answers on disk"
        assert len(card.results) == len(questions)
        assert card.complete

    def test_a_run_stopped_by_the_cap_resumes_where_it_stopped(
        self, toy_set, tmp_path
    ):
        questions, databases = toy_set
        assert len(questions) > 2, "fixture too small to stop halfway through"
        stop_after = len(questions) - 2
        path = tmp_path / "run.jsonl"

        capped = CountingSolver(quota=stop_after)
        first = score("agent", capped, questions, databases, cache=Checkpoint(path).load())

        # The card is still abandoned: this run did not answer everything, and
        # scoring it would count questions it never asked as questions it missed.
        assert not first.complete
        assert f"stopped after {stop_after}" in first.abandoned

        # The next day, with allowance again.
        tomorrow = CountingSolver()
        card = score("agent", tomorrow, questions, databases, cache=Checkpoint(path).load())

        assert card.complete
        assert len(card.results) == len(questions)
        assert tomorrow.calls == len(questions) - stop_after, (
            "resumed run re-asked questions that were already answered"
        )

    def test_two_solvers_in_one_file_do_not_read_each_others_answers(
        self, toy_set, tmp_path
    ):
        questions, databases = toy_set
        path = tmp_path / "run.jsonl"
        cache = Checkpoint(path).load()

        score("model-a", CountingSolver(), questions, databases, cache=cache)
        second = CountingSolver()
        score("model-b", second, questions, databases, cache=Checkpoint(path).load())

        assert second.calls == len(questions)


class TestItDoesNotCacheTheVerdict:
    def test_the_scorer_runs_again_on_resume(self, toy_set, tmp_path):
        # Only what the agent did is stored. If the verdict were stored too, a
        # fix to the scorer would be invisible on every question already run --
        # which is exactly the kind of silent staleness this project exists to
        # refuse.
        questions, databases = toy_set
        path = tmp_path / "run.jsonl"

        score("agent", CountingSolver(), questions, databases, cache=Checkpoint(path).load())

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert "transcript" in record
        assert "correct" not in json.dumps(record["transcript"])
        assert "judgement" not in record

    def test_answers_from_disk_are_still_scored(self, toy_set, tmp_path):
        questions, databases = toy_set
        path = tmp_path / "run.jsonl"
        score("agent", CountingSolver(), questions, databases, cache=Checkpoint(path).load())

        card = score(
            "agent", CountingSolver(), questions, databases, cache=Checkpoint(path).load()
        )

        # CountingSolver answers with the gold query, so a resumed run that is
        # genuinely re-judged must come out perfect.
        assert card.summary()["execution_accuracy"] == 1.0


class TestItSurvivesABadFile:
    def test_a_half_written_last_line_costs_one_question_not_the_file(
        self, toy_set, tmp_path
    ):
        questions, databases = toy_set
        path = tmp_path / "run.jsonl"
        cache = Checkpoint(path).load()
        for question in questions:
            cache.put("agent", question, transcript_for())

        # A process killed mid-write.
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"version": 1, "key": "agent\tx\t9", "transcr')

        reloaded = Checkpoint(path).load()

        assert len(reloaded) == len(questions)

    def test_a_file_from_an_older_shape_is_ignored_not_misread(
        self, toy_set, tmp_path
    ):
        questions, databases = toy_set
        path = tmp_path / "run.jsonl"
        path.write_text(
            json.dumps(
                {
                    "version": VERSION + 1,
                    "key": key("agent", questions[0]),
                    "transcript": {"sql": "SELECT 1"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        assert len(Checkpoint(path).load()) == 0

    def test_a_missing_file_is_simply_an_empty_run(self, tmp_path):
        assert len(Checkpoint(tmp_path / "nothing.jsonl").load()) == 0


class TestTheTranscriptSurvivesTheRoundTrip:
    def test_cost_and_usage_come_back(self, toy_set, tmp_path):
        questions, _ = toy_set
        path = tmp_path / "run.jsonl"
        cache = Checkpoint(path).load()
        original = transcript_for()
        cache.put("agent", questions[0], original)

        restored = Checkpoint(path).load().get("agent", questions[0])

        assert restored is not None
        assert restored.sql == original.sql
        assert restored.cost_usd == original.cost_usd
        assert restored.total_tokens == original.total_tokens
        assert restored.turns == original.turns
        assert restored.repairs == original.repairs
        assert restored.last_query == original.last_query

    def test_tool_calls_come_back_as_pairs(self, toy_set, tmp_path):
        # JSON has no tuple. If these came back as lists, anything unpacking
        # them as `name, args` would still work and the difference would only
        # show up somewhere far away.
        questions, _ = toy_set
        path = tmp_path / "run.jsonl"
        cache = Checkpoint(path).load()
        cache.put("agent", questions[0], transcript_for())

        restored = Checkpoint(path).load().get("agent", questions[0])

        assert restored is not None
        assert all(isinstance(call, tuple) for call in restored.tool_calls)
        assert restored.tool_calls[0] == ("list_tables", {})

    def test_a_question_with_no_query_round_trips_as_no_query(
        self, toy_set, tmp_path
    ):
        questions, _ = toy_set
        path = tmp_path / "run.jsonl"
        cache = Checkpoint(path).load()
        cache.put("agent", questions[0], transcript_for(sql=None))

        restored = Checkpoint(path).load().get("agent", questions[0])

        assert restored is not None
        assert restored.sql is None
        assert not restored.solved


class TestTheHarnessSolversAreNotCached:
    def test_the_oracle_is_never_read_from_a_checkpoint(self, toy_set, tmp_path):
        # The oracle and the mute solver cost nothing and are the harness's own
        # self-check. Serving them from disk would mean the check that proves
        # the scorer still works could itself be stale.
        questions, databases = toy_set
        path = tmp_path / "run.jsonl"
        cache = Checkpoint(path).load()

        card = score(ORACLE, oracle_solver, questions, databases, cache=cache)

        assert card.summary()["execution_accuracy"] == 1.0
        assert len(cache) == len(questions), "cache should still record, not skip"
