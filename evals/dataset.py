"""Loading questions, in BIRD's own layout.

The toy set under `evals/toy/` is written in exactly the shape BIRD ships --
same keys, same directory structure -- so pointing this at the real dev set is
a path change and nothing else. A fixture that needs its own loader is a
fixture that stops resembling the thing it stands in for.

BIRD's layout:

    dev.json                              the questions
    dev_databases/<db_id>/<db_id>.sqlite  one database per db_id
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
TOY = HERE / "toy"


@dataclass(frozen=True)
class Question:
    question_id: int
    db_id: str
    question: str
    gold_sql: str
    evidence: str = ""
    difficulty: str = "simple"


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Question(
            question_id=int(entry.get("question_id", index)),
            db_id=entry["db_id"],
            question=entry["question"],
            # BIRD spells it "SQL"; accept the lowercase form too so a
            # hand-written set does not need to shout.
            gold_sql=entry.get("SQL") or entry["sql"],
            evidence=entry.get("evidence", "") or "",
            difficulty=entry.get("difficulty", "simple"),
        )
        for index, entry in enumerate(raw)
    ]


def database_for(db_id: str, databases: Path) -> Path:
    """The SQLite file for one db_id, in BIRD's nested layout or flat beside it."""
    nested = databases / db_id / f"{db_id}.sqlite"
    if nested.is_file():
        return nested
    flat = databases / f"{db_id}.sqlite"
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"no database for {db_id!r} under {databases} "
        f"(looked for {nested.name} and {flat.name})"
    )


def toy() -> tuple[list[Question], Path]:
    """The set that ships with the repo: no download, no key, still end to end."""
    return load_questions(TOY / "questions.json"), TOY / "databases"
