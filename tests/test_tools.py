"""The agent sees the database only through these four verbs."""

from __future__ import annotations

import pytest

from schemablind.sandbox import Sandbox
from schemablind.tools import ANSWER_TOOL, SCHEMA, Tools


@pytest.fixture
def tools(school_db):
    with Sandbox(school_db) as sandbox:
        yield Tools(sandbox)


class TestListing:
    def test_it_names_every_table(self, tools):
        listed = tools.list_tables()

        for table in ("schools", "students", "scores"):
            assert table in listed

    def test_sqlite_bookkeeping_is_not_offered_as_a_table(self, school_db):
        # sqlite_sequence and friends are never what the question is about, and
        # listing them invites the agent to go and describe them.
        with Sandbox(school_db) as sandbox:
            assert "sqlite_" not in Tools(sandbox).list_tables()


class TestDescribing:
    def test_it_gives_columns_and_types(self, tools):
        described = tools.describe_table("students")

        assert "full_name TEXT" in described
        assert "stu_id INTEGER PK" in described

    def test_foreign_keys_are_shown_as_join_paths(self, tools):
        # Without this the agent has no way to find the join path, which is the
        # part of BIRD that is actually hard.
        described = tools.describe_table("students")

        assert "-> schools.sch_id" in described

    def test_an_unknown_table_says_what_does_exist(self, tools):
        described = tools.describe_table("teachers")

        assert "no table called 'teachers'" in described
        assert "schools" in described

    def test_a_case_mismatch_is_pointed_at_the_real_name(self, tools):
        assert "Did you mean 'students'?" in tools.describe_table("Students")


class TestSampling:
    def test_it_returns_a_few_real_rows(self, tools):
        sampled = tools.sample_rows("schools")

        assert "Bayside High" in sampled
        assert sampled.count("\n") <= 5  # header, rule, and the sample

    def test_the_limit_is_capped(self, tools):
        sampled = tools.sample_rows("students", limit=500)

        assert len(sampled.strip().splitlines()) <= 2 + 20

    def test_nulls_are_visible_rather_than_blank(self, tools):
        # "opened" is NULL for one school; a blank cell would read as an empty
        # string and send the agent looking for = '' instead of IS NULL.
        assert "NULL" in tools.sample_rows("schools", limit=10)

    def test_an_unknown_table_is_reported(self, tools):
        assert "no table called" in tools.sample_rows("nope")


class TestRunningSql:
    def test_rows_come_back_as_a_readable_grid(self, tools):
        out = tools.run_sql_readonly("SELECT cnty, COUNT(*) FROM schools GROUP BY cnty")

        assert "Alameda | 2" in out

    def test_an_error_comes_back_verbatim(self, tools):
        # This string is the self-repair signal. If it is ever summarised away,
        # the agent loses the only thing that tells it what to change.
        out = tools.run_sql_readonly("SELECT county FROM schools")

        assert out.startswith("ERROR:")
        assert "county" in out

    def test_a_write_is_refused_through_the_tool_too(self, tools):
        out = tools.run_sql_readonly("DELETE FROM students")

        assert out.startswith("ERROR:")
        assert "only SELECT and WITH" in out

    def test_an_empty_result_says_zero_rows(self, tools):
        assert tools.run_sql_readonly("SELECT * FROM schools WHERE cnty='Nowhere'") == "0 rows"

    def test_a_long_result_is_trimmed_and_says_so(self, tools):
        out = tools.run_sql_readonly("SELECT * FROM scores")

        assert "showing" in out or out.count("\n") <= 18


class TestDispatch:
    def test_it_calls_by_name(self, tools):
        assert "schools" in tools.call("list_tables", {})
        assert "sch_id" in tools.call("describe_table", {"table": "schools"})

    def test_every_call_is_recorded(self, tools):
        # The turn count and the tool mix are what the scorecard reports.
        tools.call("list_tables", {})
        tools.call("describe_table", {"table": "scores"})

        assert [name for name, _ in tools.calls] == ["list_tables", "describe_table"]

    def test_an_unknown_tool_is_answered_not_raised(self, tools):
        out = tools.call("drop_everything", {})

        assert "no such tool" in out
        assert "list_tables" in out  # tell it what it can use

    def test_a_missing_argument_is_answered_not_raised(self, tools):
        assert "needs a 'table' argument" in tools.call("describe_table", {})

    def test_a_bad_argument_type_is_answered_not_raised(self, tools):
        assert "ERROR" in tools.call("run_sql_readonly", {"sql": 12345})


class TestTheAdvertisedSchema:
    def test_every_tool_has_a_name_and_parameters(self):
        for tool in SCHEMA + [ANSWER_TOOL]:
            assert tool["name"]
            assert tool["description"]
            assert tool["parameters"]["type"] == "object"

    def test_the_four_verbs_are_all_advertised(self):
        assert {tool["name"] for tool in SCHEMA} == {
            "list_tables",
            "describe_table",
            "sample_rows",
            "run_sql_readonly",
        }

    def test_no_schema_is_advertised_anywhere_in_it(self):
        # The premise of the project: the agent is told the verbs, never the
        # tables. If a column name leaks into a description, it is not blind.
        blob = " ".join(t["description"] for t in SCHEMA + [ANSWER_TOOL]).lower()

        for leak in ("schools", "students", "scores", "sch_id", "cnty"):
            assert leak not in blob
