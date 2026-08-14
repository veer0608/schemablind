"""If the scorer is wrong, every number this project reports is wrong."""

from __future__ import annotations

import pytest

from schemablind.sandbox import Sandbox
from schemablind.scoring import (
    CORRECT,
    DID_NOT_RUN,
    NO_GOLD,
    WRONG_COUNT,
    WRONG_SHAPE,
    WRONG_VALUES,
    category,
    judge,
)


@pytest.fixture
def box(school_db):
    with Sandbox(school_db) as sandbox:
        yield sandbox


class TestItScoresTheObviousCases:
    def test_the_same_query_is_correct(self, box):
        sql = "SELECT sch_name FROM schools WHERE cnty = 'Marin' ORDER BY sch_name"

        result = judge(sql, sql, box)

        assert result.correct and result.exact
        assert result.reason == CORRECT

    def test_a_different_query_with_the_same_rows_is_correct(self, box):
        # The whole reason for executing rather than comparing SQL text: there
        # are many right answers and they do not look alike.
        result = judge(
            "SELECT sch_name FROM schools WHERE cnty = 'Marin'",
            "SELECT sch_name FROM schools WHERE sch_id IN (3, 4)",
            box,
        )

        assert result.correct

    def test_different_rows_are_wrong(self, box):
        result = judge(
            "SELECT sch_name FROM schools WHERE cnty = 'Marin'",
            "SELECT sch_name FROM schools WHERE cnty = 'Alameda'",
            box,
        )

        assert not result.correct
        assert category(result) == WRONG_VALUES


class TestOrderAndDuplicates:
    def test_row_order_does_not_matter(self, box):
        result = judge(
            "SELECT sch_name FROM schools ORDER BY sch_name DESC",
            "SELECT sch_name FROM schools ORDER BY sch_name ASC",
            box,
        )

        assert result.correct

    def test_a_duplicated_row_passes_the_headline_metric(self, box):
        # BIRD compares sets, so this is scored correct. Kept deliberately, and
        # the stricter flag is what shows the difference.
        result = judge(
            "SELECT cnty FROM schools WHERE cnty = 'Marin'",
            "SELECT DISTINCT cnty FROM schools WHERE cnty = 'Marin'",
            box,
        )

        assert result.correct
        assert not result.exact

    def test_the_strict_flag_agrees_when_multiplicity_matches(self, box):
        sql = "SELECT cnty FROM schools ORDER BY sch_id"

        assert judge(sql, sql, box).exact

    def test_rows_mixing_nulls_and_types_still_sort(self, box):
        # None does not compare against str, so a naive sort raises here.
        sql = "SELECT opened FROM schools"

        assert judge(sql, sql, box).exact


class TestFailureIsDiagnosed:
    def test_a_query_that_does_not_run_is_named_as_such(self, box):
        result = judge("SELECT county FROM schools", "SELECT cnty FROM schools", box)

        assert not result.correct
        assert not result.ran
        assert category(result) == DID_NOT_RUN

    def test_a_write_attempt_counts_as_not_running(self, box):
        result = judge("DELETE FROM schools", "SELECT cnty FROM schools", box)

        assert category(result) == DID_NOT_RUN

    def test_the_wrong_number_of_columns_is_named(self, box):
        result = judge(
            "SELECT sch_name, cnty FROM schools", "SELECT sch_name FROM schools", box
        )

        assert category(result) == WRONG_SHAPE
        assert "returned 2, expected 1" in result.reason

    def test_the_wrong_number_of_rows_is_named(self, box):
        result = judge(
            "SELECT sch_name FROM schools",
            "SELECT sch_name FROM schools WHERE cnty = 'Marin'",
            box,
        )

        assert category(result) == WRONG_COUNT
        assert "returned 6, expected 2" in result.reason


class TestABrokenGoldQuery:
    def test_it_is_not_charged_to_the_agent(self, box):
        # A dataset row whose own SQL does not run would otherwise lower every
        # model's score by exactly the same amount, silently.
        result = judge("SELECT sch_name FROM schools", "SELECT nope FROM nothing", box)

        assert category(result) == NO_GOLD
        assert not result.correct

    def test_it_is_distinguishable_from_the_agent_failing(self, box):
        agent_fault = judge("SELECT bad FROM schools", "SELECT cnty FROM schools", box)
        dataset_fault = judge("SELECT cnty FROM schools", "SELECT bad FROM nope", box)

        assert category(agent_fault) != category(dataset_fault)


class TestNumbers:
    def test_an_integer_matches_the_same_value_as_a_float(self, box):
        # COUNT(*) against SUM(1) returns 5 and 5 in different SQLite types.
        result = judge(
            "SELECT COUNT(*) FROM schools", "SELECT SUM(1.0) FROM schools", box
        )

        assert result.correct

    def test_an_aggregate_matches_however_it_was_written(self, box):
        result = judge(
            "SELECT AVG(score) FROM scores WHERE subject='math'",
            "SELECT SUM(score)*1.0/COUNT(score) FROM scores WHERE subject='math'",
            box,
        )

        assert result.correct
