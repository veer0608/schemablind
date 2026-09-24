"""A golden set has to be sound before anything scored against it means much."""

from __future__ import annotations

import json

import pytest

from evals.dataset import database_for, load_questions, toy
from schemablind.sandbox import Sandbox


@pytest.fixture(scope="module")
def questions_and_dbs():
    return toy()


class TestTheToySet:
    def test_it_loads(self, questions_and_dbs):
        questions, _ = questions_and_dbs

        assert len(questions) >= 10
        assert all(q.question and q.gold_sql for q in questions)

    def test_question_ids_are_unique(self, questions_and_dbs):
        questions, _ = questions_and_dbs
        ids = [q.question_id for q in questions]

        assert len(ids) == len(set(ids))

    def test_every_difficulty_is_represented(self, questions_and_dbs):
        questions, _ = questions_and_dbs

        assert {q.difficulty for q in questions} >= {"simple", "moderate", "challenging"}

    def test_every_gold_query_runs(self, questions_and_dbs):
        questions, databases = questions_and_dbs
        with Sandbox(database_for("school", databases)) as box:
            for question in questions:
                result = box.query(question.gold_sql)

                assert result.ok, f"{question.question_id}: {result.error}"

    def test_no_gold_answer_is_empty(self, questions_and_dbs):
        # An empty expected result scores every query that returns nothing as
        # correct, which is a question that rewards failing. Caught once in the
        # toy set already; this is the guard so it cannot come back.
        questions, databases = questions_and_dbs
        with Sandbox(database_for("school", databases)) as box:
            for question in questions:
                result = box.query(question.gold_sql)

                assert result.rows, f"{question.question_id} has an empty gold answer"

    def test_gold_queries_are_reads(self, questions_and_dbs):
        questions, _ = questions_and_dbs

        for question in questions:
            assert question.gold_sql.strip().lower().startswith(("select", "with"))


class TestBirdsLayout:
    def test_a_nested_database_is_found(self, questions_and_dbs):
        # BIRD ships dev_databases/<db_id>/<db_id>.sqlite, and the toy set uses
        # the same shape so the real one needs no new code.
        _, databases = questions_and_dbs

        assert database_for("school", databases).is_file()

    def test_a_missing_database_says_where_it_looked(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no database for"):
            database_for("nope", tmp_path)

    def test_birds_uppercase_sql_key_is_accepted(self, tmp_path):
        path = tmp_path / "q.json"
        path.write_text(
            '[{"question_id": 1, "db_id": "d", "question": "q?", '
            '"SQL": "SELECT 1", "evidence": "e", "difficulty": "simple"}]',
            encoding="utf-8",
        )

        loaded = load_questions(path)

        assert loaded[0].gold_sql == "SELECT 1"
        assert loaded[0].evidence == "e"

    def test_a_missing_evidence_field_is_not_an_error(self, tmp_path):
        path = tmp_path / "q.json"
        path.write_text(
            '[{"db_id": "d", "question": "q?", "sql": "SELECT 1"}]', encoding="utf-8"
        )

        loaded = load_questions(path)

        assert loaded[0].evidence == ""
        assert loaded[0].question_id == 0


class TestDuplicateQuestions:
    """A question asked twice is scored twice and weighted twice."""

    def written(self, tmp_path, entries):
        path = tmp_path / "questions.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def entry(self, question_id=1, question="how many?", sql="SELECT 1"):
        return {
            "question_id": question_id,
            "db_id": "school",
            "question": question,
            "SQL": sql,
        }

    def test_an_exact_repeat_is_asked_once(self, tmp_path):
        """BIRD Mini-Dev ships two: financial 137 and 138."""
        path = self.written(tmp_path, [self.entry(), self.entry()])

        assert len(load_questions(path)) == 1

    def test_the_first_of_a_repeat_is_the_one_kept(self, tmp_path):
        path = self.written(tmp_path, [self.entry(), self.entry()])

        assert load_questions(path)[0].question_id == 1

    def test_distinct_questions_are_all_kept(self, tmp_path):
        path = self.written(
            tmp_path, [self.entry(1), self.entry(2, question="how few?")]
        )

        assert len(load_questions(path)) == 2

    def test_the_same_id_on_a_different_question_is_refused(self, tmp_path):
        """A checkpoint keys on the id, so it would serve the wrong answer."""
        path = self.written(
            tmp_path, [self.entry(1), self.entry(1, question="something else")]
        )

        with pytest.raises(ValueError, match="share school id 1"):
            load_questions(path)

    def test_the_same_id_on_a_different_gold_is_refused(self, tmp_path):
        path = self.written(tmp_path, [self.entry(1), self.entry(1, sql="SELECT 2")])

        with pytest.raises(ValueError, match="share school id 1"):
            load_questions(path)

    def test_the_same_id_in_another_database_is_a_different_question(self, tmp_path):
        first, second = self.entry(1), self.entry(1)
        second["db_id"] = "hospital"
        path = self.written(tmp_path, [first, second])

        assert len(load_questions(path)) == 2
