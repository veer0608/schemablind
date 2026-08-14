"""Execution accuracy: run both queries, compare what came back.

This is the property that makes the project affordable. Nobody has to decide
whether the agent's SQL "looks right", and there is no rubric and no judge
model to be wrong in its own interesting ways -- two queries either return the
same rows against the same database or they do not. The label is free and it is
not an opinion.

The headline number is BIRD's own metric, deliberately: comparing result sets
the way the benchmark does is what makes the score mean anything to someone who
has never seen this repo. Two things follow from that choice and are worth
knowing rather than discovering later.

Set comparison ignores multiplicity, so a query returning a row twice matches a
gold that returns it once. That is BIRD's convention and it is kept, but a
stricter multiset comparison is reported alongside, because the gap between
them is exactly the population of queries with a duplicate-row bug that the
official metric forgives.

And a wrong query can still be right by accident: `WHERE cnty = 'Marin'` and
`WHERE sch_id > 2` may select the same rows in this database and different rows
in any other. Execution accuracy has always had that hole; it is a property of
the metric, not of the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sandbox import Result, Sandbox

#: Why a prediction failed, at a coarser grain than "wrong". Reported because
#: "34% wrong" is a number and "of the wrong ones, two thirds never ran" is a
#: direction to work in.
DID_NOT_RUN = "did not run"
NO_GOLD = "gold query failed"
WRONG_SHAPE = "wrong number of columns"
WRONG_COUNT = "wrong number of rows"
WRONG_VALUES = "right shape, wrong values"
CORRECT = "correct"


@dataclass(frozen=True)
class Judgement:
    """One prediction, scored."""

    correct: bool
    #: Stricter than `correct`: also requires duplicate rows to line up.
    exact: bool
    reason: str
    predicted: Result
    gold: Result

    @property
    def ran(self) -> bool:
        return self.predicted.ok


def judge(predicted_sql: str, gold_sql: str, sandbox: Sandbox) -> Judgement:
    """Score one prediction by running both queries."""
    gold = sandbox.query(gold_sql)
    if not gold.ok:
        # Never charge this to the agent. A gold query that will not run is a
        # broken row in the dataset, and scoring it as a miss would quietly
        # lower every model's number by the same amount.
        predicted = sandbox.query(predicted_sql)
        return Judgement(False, False, NO_GOLD, predicted, gold)

    predicted = sandbox.query(predicted_sql)
    if not predicted.ok:
        return Judgement(False, False, DID_NOT_RUN, predicted, gold)

    return Judgement(
        correct=_same_set(predicted, gold),
        exact=_same_multiset(predicted, gold),
        reason=_why(predicted, gold),
        predicted=predicted,
        gold=gold,
    )


def _same_set(predicted: Result, gold: Result) -> bool:
    """BIRD's comparison: the same rows, in any order, ignoring repeats."""
    if len(predicted.columns) != len(gold.columns):
        return False
    return set(predicted.rows) == set(gold.rows)


def _same_multiset(predicted: Result, gold: Result) -> bool:
    if len(predicted.columns) != len(gold.columns):
        return False
    return sorted(map(_sortable, predicted.rows)) == sorted(map(_sortable, gold.rows))


def _sortable(row: tuple) -> tuple:
    """Rows mix types and None, which do not sort against each other."""
    return tuple((value is None, type(value).__name__, str(value)) for value in row)


def _why(predicted: Result, gold: Result) -> str:
    if _same_set(predicted, gold):
        return CORRECT
    if len(predicted.columns) != len(gold.columns):
        return (
            f"{WRONG_SHAPE}: returned {len(predicted.columns)}, "
            f"expected {len(gold.columns)}"
        )
    if predicted.row_count != gold.row_count:
        return (
            f"{WRONG_COUNT}: returned {predicted.row_count}, "
            f"expected {gold.row_count}"
        )
    return WRONG_VALUES


def category(judgement: Judgement) -> str:
    """The coarse bucket, for counting failures by kind."""
    if judgement.correct:
        return CORRECT
    for known in (NO_GOLD, DID_NOT_RUN, WRONG_SHAPE, WRONG_COUNT, WRONG_VALUES):
        if judgement.reason.startswith(known):
            return known
    return judgement.reason
