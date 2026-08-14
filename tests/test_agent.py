"""The loop, driven by a scripted model.

Nothing in this file may make a network call. An eval that costs money is a
choice someone makes on purpose, never something a test suite does for them.
"""

from __future__ import annotations

import json

import pytest

from schemablind.agent import ANSWERED, FAILED, GAVE_UP, OUT_OF_TURNS, Agent, sql_in
from schemablind.llm import LLMError, QuotaExhausted, Reply, ToolCall, Usage
from schemablind.sandbox import Sandbox
from schemablind.tools import ANSWER_TOOL


def usage(prompt=100, completion=20, model="scripted"):
    return Usage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=None,
        latency_ms=5.0,
    )


def calls(*pairs) -> Reply:
    """A turn in which the model calls tools."""
    tool_calls = tuple(
        ToolCall(id=f"c{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(pairs)
    )
    return Reply(
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        message={
            "role": "assistant",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in tool_calls
            ],
        },
        usage=usage(),
    )


def says(text: str) -> Reply:
    return Reply(
        text=text,
        message={"role": "assistant", "content": text},
        usage=usage(),
    )


def answers(sql: str) -> Reply:
    return calls((ANSWER_TOOL["name"], {"sql": sql}))


class ScriptedClient:
    """Plays a fixed list of turns, and remembers what it was sent."""

    model = "scripted"

    def __init__(self, *turns, then=None):
        self._turns = list(turns)
        self._then = then
        self.seen: list[list[dict]] = []
        self.forced: list[str | None] = []

    def chat(self, *, messages, tools, force=None):
        self.seen.append(list(messages))
        self.forced.append(force)
        if self._turns:
            reply = self._turns.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        if self._then is not None:
            return self._then
        return says("I have nothing further.")


@pytest.fixture
def box(school_db):
    with Sandbox(school_db) as sandbox:
        yield sandbox


GOOD_SQL = "SELECT sch_name FROM schools WHERE cnty = 'Marin' ORDER BY sch_name"


class TestItExplores:
    def test_a_straightforward_run_ends_with_sql(self, box):
        client = ScriptedClient(
            calls(("list_tables", {})),
            calls(("describe_table", {"table": "schools"})),
            answers(GOOD_SQL),
        )

        transcript = Agent(client).solve("Which schools are in Marin?", box)

        assert transcript.solved
        assert transcript.stopped == ANSWERED
        assert transcript.sql == GOOD_SQL
        assert transcript.tools_used == ["list_tables", "describe_table", "final_sql"]

    def test_it_is_told_no_schema_at_the_start(self, box):
        client = ScriptedClient(answers(GOOD_SQL))

        Agent(client).solve("Which schools are in Marin?", box)

        opening = " ".join(m.get("content") or "" for m in client.seen[0])
        for leak in ("sch_id", "cnty", "schools(", "students("):
            assert leak not in opening

    def test_tool_output_is_fed_back_to_the_model(self, box):
        client = ScriptedClient(calls(("list_tables", {})), answers(GOOD_SQL))

        Agent(client).solve("q", box)

        second_turn = client.seen[1]
        tool_messages = [m for m in second_turn if m.get("role") == "tool"]
        assert tool_messages
        assert "schools" in tool_messages[0]["content"]

    def test_several_tools_in_one_turn_all_run(self, box):
        client = ScriptedClient(
            calls(
                ("describe_table", {"table": "schools"}),
                ("describe_table", {"table": "students"}),
            ),
            answers(GOOD_SQL),
        )

        transcript = Agent(client).solve("q", box)

        assert transcript.tools_used.count("describe_table") == 2

    def test_the_evidence_hint_reaches_the_model(self, box):
        client = ScriptedClient(answers(GOOD_SQL))

        Agent(client).solve("q", box, evidence="charter means charter = 1")

        user = [m for m in client.seen[0] if m["role"] == "user"][0]
        assert "charter = 1" in user["content"]


class TestSelfRepair:
    def test_a_failing_final_query_is_handed_back_with_the_error(self, box):
        client = ScriptedClient(
            answers("SELECT county FROM schools"),  # wrong column
            answers(GOOD_SQL),
        )

        transcript = Agent(client).solve("q", box)

        assert transcript.solved
        assert transcript.sql == GOOD_SQL
        assert transcript.repairs == 1
        complaint = client.seen[1][-1]["content"]
        assert "county" in complaint  # the model is told what was wrong

    def test_a_failed_exploratory_query_counts_as_a_repair(self, box):
        client = ScriptedClient(
            calls(("run_sql_readonly", {"sql": "SELECT nope FROM schools"})),
            answers(GOOD_SQL),
        )

        transcript = Agent(client).solve("q", box)

        assert transcript.repairs == 1
        assert transcript.solved

    def test_repair_can_be_switched_off_so_its_worth_can_be_measured(self, box):
        broken = "SELECT county FROM schools"
        client = ScriptedClient(answers(broken), answers(GOOD_SQL))

        transcript = Agent(client, repair=False).solve("q", box)

        # Keeps the broken query rather than retrying: that is the point of the
        # comparison, and the scorer will mark it did-not-run.
        assert transcript.sql == broken
        assert transcript.stopped == ANSWERED

    def test_endless_repair_attempts_are_capped(self, box):
        broken = "SELECT county FROM schools"
        client = ScriptedClient(then=answers(broken))

        transcript = Agent(client, max_repairs=2, max_turns=20).solve("q", box)

        assert transcript.repairs <= 3
        assert transcript.turns < 20  # it stopped, rather than burning every turn


class TestWhenItGoesWrong:
    def test_running_out_of_turns_is_recorded_not_raised(self, box):
        client = ScriptedClient(then=calls(("list_tables", {})))

        transcript = Agent(client, max_turns=3).solve("q", box)

        assert not transcript.solved
        assert transcript.stopped == OUT_OF_TURNS
        assert transcript.turns == 3

    def test_a_model_that_stops_talking_is_recorded(self, box):
        client = ScriptedClient(says("I cannot help with that."))

        transcript = Agent(client).solve("q", box)

        assert not transcript.solved
        assert transcript.stopped == GAVE_UP

    def test_a_dead_api_ends_the_question_not_the_run(self, box):
        client = ScriptedClient(LLMError("HTTP 500"))

        transcript = Agent(client).solve("q", box)

        assert transcript.stopped == FAILED
        assert "500" in transcript.error

    def test_an_exhausted_daily_quota_stops_everything(self, box):
        # Must not be swallowed: every question after it would score zero for
        # never having been asked, which looks exactly like getting them wrong.
        client = ScriptedClient(QuotaExhausted("tokens per day"))

        with pytest.raises(QuotaExhausted):
            Agent(client).solve("q", box)

    def test_a_bad_tool_name_does_not_end_the_run(self, box):
        client = ScriptedClient(calls(("summon_schema", {})), answers(GOOD_SQL))

        transcript = Agent(client).solve("q", box)

        assert transcript.solved

    def test_the_agent_cannot_write_even_if_it_tries(self, box):
        client = ScriptedClient(
            calls(("run_sql_readonly", {"sql": "DELETE FROM students"})),
            answers(GOOD_SQL),
        )

        Agent(client).solve("q", box)

        assert box.query("SELECT COUNT(*) FROM students").rows == ((8,),)


class TestSqlWrittenAsProse:
    @pytest.mark.parametrize(
        "text",
        [
            "```sql\nSELECT sch_name FROM schools\n```",
            "```\nSELECT sch_name FROM schools\n```",
            "Here you go:\nSELECT sch_name FROM schools",
        ],
    )
    def test_a_query_written_out_is_still_accepted(self, box, text):
        # Losing a right answer to a protocol slip would be measuring
        # tool-calling rather than SQL.
        client = ScriptedClient(says(text))

        transcript = Agent(client).solve("q", box)

        assert transcript.solved
        assert "sch_name" in transcript.sql

    def test_ordinary_prose_is_not_mistaken_for_sql(self, box):
        client = ScriptedClient(says("I would need to know which county."))

        assert not Agent(client).solve("q", box).solved

    def test_extraction_ignores_a_non_select(self):
        assert sql_in("DELETE FROM t") is None
        assert sql_in("") is None


class TestMakingItCommit:
    """Asking a model to submit its answer is a request it can decline.

    On the first live run it declined on every single question: explored
    correctly, ran the right query, then answered in prose. Naming the function
    in tool_choice makes the API require the call rather than offer it.
    """

    def test_the_nudge_turn_compels_the_answer_tool(self, box):
        client = ScriptedClient(
            says("There are 6 schools."),  # answers in prose, submits nothing
            answers(GOOD_SQL),
        )

        Agent(client).solve("q", box)

        assert client.forced == [None, ANSWER_TOOL["name"]]

    def test_ordinary_turns_leave_the_choice_open(self, box):
        # Forcing every turn would stop it exploring at all.
        client = ScriptedClient(calls(("list_tables", {})), answers(GOOD_SQL))

        Agent(client).solve("q", box)

        assert client.forced == [None, None]

    def test_it_is_only_forced_once(self, box):
        client = ScriptedClient(then=says("still just talking"))

        Agent(client, max_turns=4).solve("q", box)

        assert client.forced.count(ANSWER_TOOL["name"]) == 1


class TestWhatItCost:
    def test_tokens_and_turns_are_totalled(self, box):
        client = ScriptedClient(calls(("list_tables", {})), answers(GOOD_SQL))

        transcript = Agent(client).solve("q", box)

        assert transcript.turns == 2
        assert transcript.prompt_tokens == 200
        assert transcript.completion_tokens == 40
        assert transcript.total_tokens == 240
        assert transcript.latency_ms == 10.0

    def test_an_unpriced_model_costs_none_rather_than_nothing(self, box):
        client = ScriptedClient(answers(GOOD_SQL))

        assert Agent(client).solve("q", box).cost_usd is None
