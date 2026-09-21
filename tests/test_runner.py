"""The harness has to be right before anything it measures is worth reading."""

from __future__ import annotations

import pytest

from evals.dataset import Question, toy
from evals.runner import (
    MARKERS,
    MUTE,
    ORACLE,
    Scorecard,
    main,
    markdown,
    mute_solver,
    oracle_solver,
    report,
    score,
    splice,
    split_of,
    warn_about_split,
)
from schemablind.agent import Transcript
from schemablind.llm import QuotaExhausted, Usage


@pytest.fixture(scope="module")
def toy_set():
    return toy()


class TestTheHarnessProvesItself:
    def test_the_gold_queries_score_themselves_perfectly(self, toy_set):
        # If this is ever below 100%, the scorer is wrong and no model number
        # produced by it means anything.
        questions, databases = toy_set

        card = score(ORACLE, oracle_solver, questions, databases)

        assert card.summary()["execution_accuracy"] == 1.0
        assert card.summary()["strict_accuracy"] == 1.0

    def test_producing_nothing_scores_nothing(self, toy_set):
        questions, databases = toy_set

        card = score(MUTE, mute_solver, questions, databases)

        assert card.summary()["execution_accuracy"] == 0.0
        assert card.failures() == {"no query produced": len(questions)}

    def test_the_check_command_passes(self, capsys):
        assert main(["--check"]) == 0
        assert "oracle 100.0%" in capsys.readouterr().out

    def test_help_renders(self, capsys):
        """`--help` died with "ValueError: incomplete format": argparse
        interpolates help strings, and one of them said 100% rather than 100%%.
        Nothing exercised --help, so the way to find it was to need it."""
        import pytest

        with pytest.raises(SystemExit) as exit:
            main(["--help"])

        assert exit.value.code == 0
        assert "--checkpoint" in capsys.readouterr().out


class TestScoringAnAgent:
    def _card(self, toy_set, sql_for):
        questions, databases = toy_set

        def solver(question, sandbox):
            sql = sql_for(question)
            return Transcript(
                sql=sql,
                stopped="answered",
                turns=3,
                usage=[Usage("m", 1000, 100, 0.001, 250.0)],
            )

        return score("stub", solver, questions, databases)

    def test_a_query_that_does_not_run_is_counted_apart_from_a_wrong_one(self, toy_set):
        card = self._card(toy_set, lambda q: "SELECT nope FROM nothing")

        assert card.summary()["execution_accuracy"] == 0.0
        assert card.summary()["produced_sql"] == 1.0  # it did produce SQL
        assert "did not run" in card.failures()

    def test_cost_and_turns_are_averaged(self, toy_set):
        card = self._card(toy_set, lambda q: q.gold_sql)
        summary = card.summary()

        assert summary["execution_accuracy"] == 1.0
        assert summary["turns"] == 3.0
        assert summary["cost_per_question"] == pytest.approx(0.001)
        assert summary["p50_latency_ms"] == 250.0

    def test_an_unpriced_model_is_unpriced_not_free(self, toy_set):
        questions, databases = toy_set

        def solver(question, sandbox):
            return Transcript(
                sql=question.gold_sql, usage=[Usage("m", 10, 1, None, 5.0)]
            )

        assert score("s", solver, questions, databases).summary()["cost_per_question"] is None


class TestARunThatDiesPartWay:
    def test_it_is_abandoned_rather_than_scored(self, toy_set):
        questions, databases = toy_set
        state = {"n": 0}

        def solver(question, sandbox):
            state["n"] += 1
            if state["n"] > 3:
                raise QuotaExhausted("tokens per day (TPD)")
            return Transcript(sql=question.gold_sql)

        card = score("dies", solver, questions, databases)

        assert not card.complete
        assert "3 of 12" in card.abandoned
        assert len(card.results) == 3

    def test_an_abandoned_card_gets_no_row(self, toy_set):
        questions, databases = toy_set
        good = score(ORACLE, oracle_solver, questions, databases)
        dead = Scorecard(name="dead", abandoned="ran out")

        table = markdown([good, dead], questions)

        assert ORACLE in table
        assert "dead" not in table

    def test_the_report_says_why_it_is_missing(self, toy_set):
        questions, _ = toy_set
        dead = Scorecard(name="dead", abandoned="ran out of quota")

        text = report([dead], questions)

        assert "NOT SCORED" in text
        assert "ran out of quota" in text


class TestTheSplit:
    def test_it_is_stable_for_the_same_question(self, toy_set):
        questions, _ = toy_set

        assert [split_of(q) for q in questions] == [split_of(q) for q in questions]

    def test_it_is_derived_from_the_question_not_its_position(self):
        first = Question(0, "db", "how many rows?", "SELECT 1")
        same_text_later = Question(0, "db", "how many rows?", "SELECT 2")

        assert split_of(first) == split_of(same_text_later)

    def test_an_empty_held_out_half_is_announced(self, toy_set):
        # The toy set really does put every question in dev. Saying so is the
        # alternative to salting the hash until it looks better, which would be
        # choosing the test set by hand.
        questions, _ = toy_set

        assert "no questions landed in the held-out half" in warn_about_split(questions)

    def test_a_thin_held_out_half_is_announced(self):
        questions = [Question(i, "db", f"q{i}?", "SELECT 1") for i in range(30)]

        warning = warn_about_split(questions)

        assert warning == "" or "Too few" in warning

    def test_a_healthy_split_says_nothing(self):
        questions = [Question(i, "db", f"question number {i}?", "SELECT 1") for i in range(400)]

        assert warn_about_split(questions) == ""


class TestThePublishedTable:
    def test_it_is_generated_from_the_run(self, toy_set):
        questions, databases = toy_set
        card = score(ORACLE, oracle_solver, questions, databases)

        table = markdown([card], questions)

        assert "| configuration |" in table
        assert "100.0%" in table

    def test_splicing_leaves_the_rest_alone(self, tmp_path):
        start, end = MARKERS
        target = tmp_path / "README.md"
        target.write_text(f"before\n{start}\nold\n{end}\nafter\n", encoding="utf-8")

        assert splice(target, "new")

        text = target.read_text(encoding="utf-8")
        assert text.startswith("before\n") and text.endswith("\nafter\n")
        assert "new" in text and "old" not in text

    def test_a_file_without_markers_is_untouched(self, tmp_path):
        target = tmp_path / "README.md"
        target.write_text("nothing here", encoding="utf-8")

        assert not splice(target, "new")
        assert target.read_text(encoding="utf-8") == "nothing here"


class TestTheCommandNeverPaysByAccident:
    def test_the_default_solvers_are_the_free_ones(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "evals.runner.build_client",
            lambda *a, **k: pytest.fail("the default run must not call a model"),
        )

        assert main([]) == 0
        assert ORACLE in capsys.readouterr().out

    def test_the_check_never_reaches_for_a_model(self, monkeypatch):
        monkeypatch.setattr(
            "evals.runner.build_client",
            lambda *a, **k: pytest.fail("--check must not call a model"),
        )

        assert main(["--check"]) == 0
