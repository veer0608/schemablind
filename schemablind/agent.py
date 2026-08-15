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

Work in this order, and in as few turns as you can:
1. list_tables -- always first. You are not told what exists.
2. describe_table on EVERY table that might be relevant, in ONE call. It takes
   a list. It returns columns, foreign keys as '-> other_table.column' (your
   join path), and a couple of real rows so you can see how values are encoded.
   Asking for tables one at a time wastes turns and tells you nothing extra.
3. sample_rows only if the rows from step 2 left something genuinely unclear.
4. run_sql_readonly to try your query and see real rows.
5. final_sql once, when the rows look right.

Two or three turns is a good run. Nine is not.

Step 5 is not optional and it is not the same as telling me the answer. Running
a query that returns the right rows does not submit it -- nothing is recorded
until you call final_sql with that exact query. Do not reply with the answer in
words. Do not stop after run_sql_readonly. Call final_sql.

Rules:
- Only SELECT and WITH will run. There is no writing to this database.
- Column names are often abbreviated and are rarely what the question calls
  them. Check with describe_table rather than assuming.
- If a query errors, read the message. It names the missing column or table.
- Return exactly the columns the question asks for, and no more. An extra
  column makes the answer wrong.
- Do not select the thing you ranked or aggregated by unless it was asked for.
  "Who spent the most" wants the person, not the person and the total. "Which
  year had the highest X" wants the year alone. Put the measure in ORDER BY,
  not in SELECT -- this is the single most common way to be exactly one column
  wrong.
- When the question asks for one value, return one column and one row.
- Do not round, and do not format. ROUND(x, 2) is a different number from x and
  will be judged different. Return the raw computed value unless the question
  explicitly asks for a rounded one.
- Do not invent a table or column you have not seen in describe_table output.
"""

#: A fenced or bare SELECT in ordinary prose, for the turns where a model
#: writes the query out instead of calling the tool it was given.
_SQL_IN_TEXT = re.compile(
    r"```(?:sql)?\s*(?P<fenced>.+?)```|(?P<bare>\b(?:SELECT|WITH)\b.+)",
    re.I | re.S,
)

NUDGE = (
    "You have not submitted anything yet. Running a query does not record it. "
    "Call final_sql now with the single SELECT that answers the question."
)

SUBMITTED = "final_sql"
FROM_LAST_QUERY = "its last verified query"

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
    #: Times it had to be reminded that an answer must be submitted.
    nudges: int = 0
    #: How the answer was arrived at -- submitted, or read off the
    #: transcript because it never submitted one.
    answered_via: str = SUBMITTED
    #: The last query it ran successfully, whether or not it submitted it.
    last_query: str | None = None
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
        nudged = False
        force_next: str | None = None

        for _ in range(self.max_turns):
            transcript.turns += 1
            try:
                reply = self.client.chat(
                    messages=messages, tools=offered, force=force_next
                )
                force_next = None
            except QuotaExhausted:
                raise
            except LLMError as exc:
                # Do not discard the work. A call failing on turn seven says
                # nothing about the query verified on turn six, and throwing
                # that away turned six API errors into six "produced no query"
                # -- a failure of the agent's reasoning, which is not what
                # happened and not what the scorecard should say.
                transcript.error = str(exc)
                return self._fall_back(transcript, FAILED)

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
                if not nudged:
                    # Observed on the first live run: the agent explores
                    # correctly, runs a query that returns the right rows, and
                    # then reports the answer in prose without ever submitting
                    # it. Giving up on that is scoring the protocol rather than
                    # the SQL, so it gets told once and asked again.
                    nudged = True
                    transcript.nudges += 1
                    messages.append({"role": "user", "content": NUDGE})
                    # Words alone did not work: on the first live run every
                    # question ignored the request. Naming the function makes
                    # the API require it.
                    force_next = ANSWER_TOOL["name"]
                    continue
                return self._fall_back(transcript, GAVE_UP)

            for call in reply.tool_calls:
                transcript.tool_calls.append((call.name, call.arguments))
                if call.name == ANSWER_TOOL["name"]:
                    candidate = str(call.arguments.get("sql") or "")
                    if self._accept(candidate, tools, transcript, messages, call.id):
                        return transcript
                    continue
                output = tools.call(call.name, call.arguments)
                if call.name == "run_sql_readonly" and not output.startswith("ERROR:"):
                    # The moment the agent is most likely to think it has
                    # finished: it just saw the right rows. Six of twelve runs
                    # stopped exactly here, so the reminder goes where the
                    # mistake happens rather than only in the system prompt.
                    ran = str(call.arguments.get("sql") or "").strip()
                    if ran:
                        transcript.last_query = ran
                    output += (
                        "\n(These rows are NOT submitted. If this is the answer, "
                        "call final_sql with this exact query.)"
                    )
                if output.startswith("ERROR:"):
                    transcript.repairs += 1
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )

        return self._fall_back(transcript, OUT_OF_TURNS)

    def _fall_back(self, transcript: Transcript, why: str) -> Transcript:
        """Take the last query it ran and verified, if it never submitted one.

        Not generosity -- reading the transcript. An agent that ran
        `SELECT COUNT(*) ... WHERE cnty='Marin'`, saw the rows, and then said
        "2" in prose has unambiguously told you which query it means. Where
        several ran, the last successful one is the one it settled on.

        Recorded as a separate route rather than folded into the score, because
        how often the protocol has to be rescued is itself worth reporting.
        """
        if transcript.sql or not transcript.last_query:
            transcript.stopped = why
            return transcript
        transcript.sql = transcript.last_query
        transcript.stopped = why
        transcript.answered_via = FROM_LAST_QUERY
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
