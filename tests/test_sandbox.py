"""The sandbox runs SQL a model wrote. These are the tests that matter most.

Every other number this project reports is downstream of the claim that the
agent cannot change the database it is exploring.
"""

from __future__ import annotations

import sqlite3

import pytest

from schemablind.sandbox import Sandbox, statement_is_read_only


@pytest.fixture
def box(school_db):
    with Sandbox(school_db) as sandbox:
        yield sandbox


class TestItReads:
    def test_a_select_comes_back_with_columns_and_rows(self, box):
        result = box.query("SELECT sch_name, cnty FROM schools ORDER BY sch_id")

        assert result.ok
        assert result.columns == ("sch_name", "cnty")
        assert result.rows[0] == ("Bayside High", "Alameda")
        assert result.row_count == 6

    def test_a_with_clause_is_a_read(self, box):
        result = box.query(
            "WITH t AS (SELECT cnty, COUNT(*) n FROM schools GROUP BY cnty) "
            "SELECT * FROM t ORDER BY n DESC, cnty"
        )

        assert result.ok
        assert result.rows[0] == ("Alameda", 2)

    def test_a_join_across_three_tables_works(self, box):
        result = box.query(
            "SELECT s.sch_name, AVG(sc.score) FROM schools s "
            "JOIN students st ON st.sch_id = s.sch_id "
            "JOIN scores sc ON sc.stu_id = st.stu_id "
            "GROUP BY s.sch_name ORDER BY 2 DESC LIMIT 1"
        )

        assert result.ok
        assert result.rows[0][0] == "Oakfield Charter"  # 99 and 97

    def test_nulls_survive_as_none(self, box):
        result = box.query("SELECT opened FROM schools WHERE sch_id = 5")

        assert result.rows == ((None,),)


class TestItRefusesToWrite:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO schools VALUES (9,'X','Y',0,NULL)",
            "UPDATE schools SET cnty = 'Z'",
            "DELETE FROM students",
            "DROP TABLE scores",
            "CREATE TABLE evil (x INT)",
            "ALTER TABLE schools ADD COLUMN x INT",
            "REPLACE INTO schools VALUES (1,'X','Y',0,NULL)",
        ],
    )
    def test_a_write_is_refused_before_it_runs(self, box, sql):
        result = box.query(sql)

        assert not result.ok
        assert "only SELECT and WITH" in result.error

    def test_attach_cannot_reach_a_second_file(self, box, tmp_path):
        # Left to SQLite this would open another database entirely, and the
        # read-only promise would only ever have covered the first one.
        result = box.query(f"ATTACH DATABASE '{tmp_path / 'other.db'}' AS other")

        assert not result.ok
        assert "only SELECT and WITH" in result.error
        assert not (tmp_path / "other.db").exists()

    def test_a_write_hidden_after_a_semicolon_is_refused(self, box):
        result = box.query("SELECT 1; DROP TABLE scores")

        assert not result.ok
        assert "one statement per call" in result.error

    def test_a_write_hidden_behind_a_comment_is_refused(self, box):
        result = box.query("-- SELECT 1\nDELETE FROM students")

        assert not result.ok
        assert "only SELECT and WITH" in result.error

    def test_a_block_comment_does_not_disguise_the_first_word(self, box):
        result = box.query("/* SELECT */ UPDATE schools SET cnty='Z'")

        assert not result.ok
        assert "UPDATE" in result.error

    def test_the_database_is_unchanged_after_all_of_that(self, box):
        assert box.query("SELECT COUNT(*) FROM schools").rows == ((6,),)
        assert box.query("SELECT COUNT(*) FROM students").rows == ((8,),)

    def test_the_file_itself_is_opened_read_only(self, writable_db):
        # The layer that actually holds. Even reaching past every check above,
        # the driver will not write.
        with Sandbox(writable_db) as sandbox:
            connection = sandbox.connect()

            with pytest.raises(sqlite3.OperationalError):
                connection.execute("INSERT INTO schools VALUES (9,'X','Y',0,NULL)")

    def test_a_trailing_semicolon_is_fine(self, box):
        assert box.query("SELECT 1;").ok


class TestItSurvivesBadSql:
    def test_a_syntax_error_is_a_result_not_an_exception(self, box):
        result = box.query("SELECT FROM WHERE")

        assert not result.ok
        assert "syntax" in result.error.lower()

    def test_an_unknown_column_says_which(self, box):
        # This message is the whole self-repair signal, so it has to survive.
        result = box.query("SELECT county FROM schools")

        assert not result.ok
        assert "county" in result.error

    def test_an_unknown_table_says_which(self, box):
        result = box.query("SELECT * FROM teachers")

        assert not result.ok
        assert "teachers" in result.error

    def test_an_empty_statement_is_refused(self, box):
        assert not box.query("   ").ok

    def test_a_missing_database_is_reported_not_raised(self, tmp_path):
        result = Sandbox(tmp_path / "nope.sqlite").query("SELECT 1")

        assert not result.ok
        assert "no such database" in result.error


class TestItStaysWithinLimits:
    def test_rows_are_capped_and_the_cap_is_admitted(self, school_db):
        with Sandbox(school_db, max_rows=3) as sandbox:
            result = sandbox.query("SELECT stu_id FROM students ORDER BY stu_id")

        assert result.row_count == 3
        assert result.truncated

    def test_exactly_the_cap_is_not_called_truncated(self, school_db):
        # Off by one here would report truncation on every query that happened
        # to return exactly the limit.
        with Sandbox(school_db, max_rows=8) as sandbox:
            result = sandbox.query("SELECT stu_id FROM students")

        assert result.row_count == 8
        assert not result.truncated

    def test_a_runaway_query_is_stopped(self, school_db):
        # A cross join is a perfectly legal SELECT and will not finish.
        with Sandbox(school_db, timeout_s=0.4) as sandbox:
            result = sandbox.query(
                "WITH RECURSIVE forever(n) AS ("
                "  SELECT 1 UNION ALL SELECT n + 1 FROM forever"
                ") SELECT COUNT(*) FROM forever"
            )

        assert not result.ok
        assert "limit" in result.error and "stopped" in result.error

    def test_the_connection_still_works_after_a_timeout(self, school_db):
        with Sandbox(school_db, timeout_s=0.4) as sandbox:
            sandbox.query(
                "WITH RECURSIVE forever(n) AS ("
                "  SELECT 1 UNION ALL SELECT n + 1 FROM forever"
                ") SELECT COUNT(*) FROM forever"
            )

            assert sandbox.query("SELECT COUNT(*) FROM schools").rows == ((6,),)

    def test_elapsed_time_is_recorded(self, box):
        assert box.query("SELECT 1").elapsed_ms >= 0


class TestTheRefusalMessage:
    def test_it_names_what_the_statement_started_with(self):
        assert "INSERT" in statement_is_read_only("INSERT INTO t VALUES (1)")

    def test_a_read_is_allowed(self):
        assert statement_is_read_only("SELECT 1") is None
        assert statement_is_read_only("  with t as (select 1) select * from t") is None
