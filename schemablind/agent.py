"""The loop: a question, a database it has never seen, and four verbs.

The agent is given no schema. It has to call `list_tables`, work out which of
them matter, read their columns, and find the join path itself. That is the
whole experiment -- writing SQL against a schema you were handed is a different
and much easier task than working out which schema you are looking at, and only
the second one resembles being handed a database and a question.

The other half is self-repair. A query that fails comes back with SQLite's own
message ("no such column: county"), and the agent gets another turn to fix it.
How much that is worth is a number this project reports rather than assumes,
because it costs turns and tokens and it is not obvious it pays for itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError, QuotaExhausted, Usage
from .sandbox import Sandbox
from .tools import ANSWER_TOOL, SCHEMA, Tools

SYSTEM = """\
You answer questions by writing SQLite SQL against a database you have not been
shown. You do not know its tables, its columns, or how they join. Find out.

Work in this order:
1. list_tables -- always first. You are not told what exists.
2. describe_table on the tables that look relevant. Foreign keys are printed as
   '-> other_table.column'; that is your join path.
3. sample_rows when a column's meaning is unclear -- whether a flag is 0/1 or
   'Y'/'N', how a date is formatted, what a code actually contains. Guessing an
   encoding is the most common way to write a query that runs and is wrong.
4. run_sql_readonly to try your query and see real rows.
5. final_sql once, when the rows look right.

Rules:
- Only SELECT and WITH will run. There is no writing to this database.
- Column names are often abbreviated and are rarely what the question calls
  them. Check with describe_table rather than assuming.
- If a query errors, read the message. It names the missing column or table.
- Return exactly the columns the question asks for, and no more. An extra
  column makes the answer wrong.
- When the question asks for one value, return one column and one row.
- Do not invent a table or column you have not seen in describe_table output.
"""

#: A fenced or bare SELECT in ordinary prose, for the turns where a model
#: writes the query out instead of calling the tool it was given.
_SQL_IN_TEXT = re.compile(
    r"```(?:sql)?\s*(?P<fenced>.+?)```|(?P<bare>\b(?:SELECT|WITH)\b.+)",
    re.I | re.S,
)

ANSWERED = "answered"
OUT_OF_TURNS = "ran out of turns"
GAVE_UP = "stopped without a query"
FAILED = "the model call failed"


@dataclass
class Transcript:
    """What the agent did, and what it cost to find out."""

    sql: str | None = None
    stopped: str = OUT_OF_TURNS
    turns: int = 0
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)
    #: Times a query the agent ran came back as an error it then worked from.
    repairs: int = 0
    error: str | None = None

    @property
    def solved(self) -> bool:
        return bool(self.sql)

    @property
    def prompt_tokens(self) -> int:
        return sum(u.prompt_tokens for u in self.usage)

    @property
    def completion_tokens(self) -> int:
        return sum(u.completion_tokens for u in self.usage)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float | None:
        if not self.usage or any(not u.priced for u in self.usage):
            return None
        return sum(u.cost_usd or 0.0 for u in self.usage)

    @property
    def latency_ms(self) -> float:
        return sum(u.latency_ms for u in self.usage)

    @property
    def tools_used(self) -> list[str]:
        return [name for name, _ in self.tool_calls]


class Agent:
    def __init__(
        self,
        client: LLMClient,
        *,
        max_turns: int = 12,
        max_repairs: int = 3,
        repair: bool = True,
    ) -> None:
        self.client = client
        self.max_turns = max_turns
        self.max_repairs = max_repairs
        #: Off makes the first query final, which is how the cost of repair
        #: gets measured rather than asserted.
        self.repair = repair

    def solve(self, question: str, sandbox: Sandbox, *, evidence: str = "") -> Transcript:
        tools = Tools(sandbox)
        transcript = Transcript()
        asked = question if not evidence else f"{question}\n\nHint: {evidence}"
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": asked},
        ]
        offered = SCHEMA + [ANSWER_TOOL]

        for _ in range(self.max_turns):
            transcript.turns += 1
            try:
                reply = self.client.chat(messages=messages, tools=offered)
            except QuotaExhausted:
                raise
            except LLMError as exc:
                transcript.stopped, transcript.error = FAILED, str(exc)
                return transcript

            if reply.usage:
                transcript.usage.append(reply.usage)
            messages.append(reply.message or {"role": "assistant", "content": reply.text})

            if not reply.wants_tools:
                # Some models write the query out instead of calling the tool.
                # Losing a right answer to a protocol slip would be measuring
                # tool-calling, which is not what this is about.
                salvaged = sql_in(reply.text)
                if salvaged and self._accept(salvaged, tools, transcript, messages):
                    return transcript
                if salvaged:
                    continue
                transcript.stopped = GAVE_UP
                return transcript

            for call in reply.tool_calls:
                transcript.tool_calls.append((call.name, call.arguments))
                if call.name == ANSWER_TOOL["name"]:
                    candidate = str(call.arguments.get("sql") or "")
                    if self._accept(candidate, tools, transcript, messages, call.id):
                        return transcript
                    continue
                output = tools.call(call.name, call.arguments)
                if output.startswith("ERROR:"):
                    transcript.repairs += 1
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )

        return transcript

    def _accept(
        self,
        candidate: str,
        tools: Tools,
        transcript: Transcript,
        messages: list[dict],
        call_id: str | None = None,
    ) -> bool:
        """Take the answer, or hand back the error and let it try again.

        A final query that does not run is worth nothing, and the model is
        holding the one thing that would fix it -- SQLite's message. Handing
        that back is the difference between a failed attempt and a repaired
        one.
        """
        if not candidate.strip():
            return False
        result = tools.sandbox.query(candidate)
        if result.ok:
            transcript.sql = candidate.strip()
            transcript.stopped = ANSWERED
            return True

        transcript.repairs += 1
        if not self.repair or transcript.repairs > self.max_repairs:
            # Keep it anyway: a wrong query that ran is a different failure
            # from never producing one, and the scorer distinguishes them.
            transcript.sql = candidate.strip()
            transcript.stopped = ANSWERED
            return True

        complaint = (
            f"That query failed: {result.error}\n"
            f"Fix it and call {ANSWER_TOOL['name']} again."
        )
        if call_id:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": complaint})
        else:
            messages.append({"role": "user", "content": complaint})
        return False


def sql_in(text: str) -> str | None:
    """Pull a SELECT out of prose, for models that answer without the tool."""
    if not text:
        return None
    match = _SQL_IN_TEXT.search(text)
    if not match:
        return None
    found = (match.group("fenced") or match.group("bare") or "").strip()
    if not found.lower().lstrip().startswith(("select", "with")):
        return None
    return found.rstrip().rstrip(";").strip() or None
