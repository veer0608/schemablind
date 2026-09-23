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


def deduplicate(questions: list[Question]) -> list[Question]:
    """One row per question, because a repeat is counted twice.

    BIRD Mini-Dev ships two exact repeats -- `financial` 137 and 138, each
    appearing with the same id, the same text and the same gold SQL. Left in,
    they are asked twice, scored twice and weighted twice: the 2026-09-23 dev
    run got both right, which lifted 64.7% to 64.9% for no work at all. A
    repeat also inflates the cost and turn columns by asking again.

    A pair that shares an id while differing in text is a different animal and
    is refused rather than merged. The checkpoint keys on (solver, db_id,
    question_id), so such a pair would be served its twin's cached answer --
    a wrong answer, silently, and only on the second run.
    """
    seen: dict[tuple[str, int], Question] = {}
    kept: list[Question] = []
    for question in questions:
        key = (question.db_id, question.question_id)
        earlier = seen.get(key)
        if earlier is None:
            seen[key] = question
            kept.append(question)
            continue
        if (earlier.question, earlier.gold_sql) != (question.question, question.gold_sql):
            raise ValueError(
                f"two different questions share {question.db_id} id "
                f"{question.question_id}. A checkpoint would serve one of them "
                f"the other's answer, so this cannot be loaded as it stands."
            )
    return kept


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return deduplicate([
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
    ])


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


#: Where `minidev.zip` lands when unpacked into data/. Gitignored: 3.3GB of
#: databases is a download, not a thing to commit.
BIRD = HERE.parent / "data" / "minidev" / "MINIDEV"


def bird() -> tuple[list[Question], Path]:
    """BIRD Mini-Dev: 500 curated questions over 11 databases.

    Read by exactly the loader above, unchanged -- the toy set was written in
    BIRD's shape from the start so that arriving here would be a path change
    and nothing else. It was.
    """
    questions = BIRD / "mini_dev_sqlite.json"
    databases = BIRD / "dev_databases"
    if not questions.is_file():
        raise FileNotFoundError(
            f"no BIRD Mini-Dev at {BIRD}. Download minidev.zip from "
            f"https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip "
            f"(764MB) and unpack it into data/."
        )
    return load_questions(questions), databases
