"""Keep the questions a run already answered, so the next run starts after them.

A schema-blind run is metered by a daily token allowance, and three runs have
already ended against it. The runner is right to abandon a scorecard when the
allowance goes -- questions it never asked would otherwise be indistinguishable
from questions it got wrong -- but the answers it had already paid for went with
it. Over 215 held-out questions that is the difference between a number and no
number.

So each answered question is appended here the moment the model returns. A run
that stops at 90 of 215 starts the next day at 91, and the day after that the
card is complete and can be scored like any other.

Only the model's half is cached. Judging is local, deterministic and free, so it
is redone on every load: a change to the scorer can never be hidden behind a
verdict recorded under the old one. What is stored is what the agent did, not
whether it was right.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from schemablind.agent import Transcript
from schemablind.llm import Usage

from evals.dataset import Question

#: Bumped when the stored shape changes. A file written by an older version is
#: ignored rather than half-read -- a resumed run built from a misread record is
#: worse than one that simply starts again.
VERSION = 1


def key(solver: str, question: Question) -> str:
    """Identify a question within a solver's run.

    `question_id` is unique within BIRD but the toy set numbers from zero per
    database, so `db_id` is carried too. The solver name keeps two solvers
    sharing one file from reading each other's answers.
    """
    return f"{solver}\t{question.db_id}\t{question.question_id}"


def _usage_json(usage: Usage) -> dict:
    return {
        "model": usage.model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": usage.cost_usd,
        "latency_ms": usage.latency_ms,
    }


def _transcript_json(transcript: Transcript) -> dict:
    return {
        "sql": transcript.sql,
        "stopped": transcript.stopped,
        "turns": transcript.turns,
        # JSON has no tuple; these come back as lists and are re-paired on load.
        "tool_calls": [[name, args] for name, args in transcript.tool_calls],
        "usage": [_usage_json(u) for u in transcript.usage],
        "repairs": transcript.repairs,
        "nudges": transcript.nudges,
        "answered_via": transcript.answered_via,
        "last_query": transcript.last_query,
        "error": transcript.error,
    }


def _transcript_from(raw: dict) -> Transcript:
    return Transcript(
        sql=raw["sql"],
        stopped=raw["stopped"],
        turns=raw["turns"],
        tool_calls=[(name, args) for name, args in raw["tool_calls"]],
        usage=[Usage(**u) for u in raw["usage"]],
        repairs=raw["repairs"],
        nudges=raw["nudges"],
        answered_via=raw["answered_via"],
        last_query=raw["last_query"],
        error=raw["error"],
    )


class Checkpoint:
    """A JSONL file of answered questions, appended to as they are answered.

    Append-only and flushed per question on purpose: the failure this exists to
    survive is the process not reaching its own end.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._answers: dict[str, Transcript] = {}
        self.resumed = 0

    # --- reading ----------------------------------------------------------

    def load(self) -> "Checkpoint":
        """Read what is already answered. A missing file is simply an empty run."""
        self._answers = {}
        if not self.path.exists():
            return self
        for record in self._records():
            if record.get("version") != VERSION:
                continue
            self._answers[record["key"]] = _transcript_from(record["transcript"])
        return self

    def _records(self) -> Iterator[dict]:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a partial last line. Losing
                    # one question is the point of the format; refusing to read
                    # the other 89 because of it would not be.
                    continue

    def get(self, solver: str, question: Question) -> Transcript | None:
        found = self._answers.get(key(solver, question))
        if found is not None:
            self.resumed += 1
        return found

    def __len__(self) -> int:
        return len(self._answers)

    def __bool__(self) -> bool:
        # Without this, `__len__` makes an empty checkpoint falsy and `if cache:`
        # quietly skips the very first run -- the one that has nothing yet and
        # most needs recording.
        return True

    # --- writing ----------------------------------------------------------

    def put(self, solver: str, question: Question, transcript: Transcript) -> None:
        identifier = key(solver, question)
        self._answers[identifier] = transcript
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "version": VERSION,
            "key": identifier,
            "solver": solver,
            "db_id": question.db_id,
            "question_id": question.question_id,
            # Carried for a human reading the file; never read back.
            "question": question.question,
            "transcript": _transcript_json(transcript),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
