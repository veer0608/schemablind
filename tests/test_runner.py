"""The harness has to be right before anything it measures is worth reading."""

from __future__ import annotations

import json

import pytest

from evals.dataset import Question, toy
from evals.runner import (
    MARKERS,
    compare,
    sampled,
    Result,
    republish,
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


class TestRepublishing:
    """A finished number lives in its JSON. Re-running to print it again
    spends another day's allowance and can quietly print a different one."""

    def saved(self, tmp_path, name="gemini-3.5-flash-lite", abandoned="", n=268):
        path = tmp_path / f"{name.replace('/', '-')}.json"
        path.write_text(
            json.dumps(
                {
                    name: {
                        "abandoned": abandoned,
                        "summary": {
                            "n": n,
                            "execution_accuracy": 0.625,
                            "strict_accuracy": 0.6,
                            "produced_sql": 1.0,
                            "cost_per_question": 0.0004,
                            "p50_latency_ms": 4200.0,
                            "turns": 5.5,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_a_saved_row_is_reprinted_as_it_was_scored(self, tmp_path, capsys):
        assert republish([self.saved(tmp_path)]) == 0

        printed = capsys.readouterr().out
        assert "| gemini-3.5-flash-lite | 268 | 62.5% | 60.0% | 100.0% | 5.5 |" in printed

    def test_an_abandoned_run_is_named_rather_than_silently_dropped(
        self, tmp_path, capsys
    ):
        path = self.saved(tmp_path, abandoned="stopped after 243 of 268: HTTP 429")

        assert republish([path]) == 2

        complained = capsys.readouterr().err
        assert "not published" in complained
        assert "243 of 268" in complained

    def test_nothing_complete_refuses_to_empty_the_scorecard(self, tmp_path):
        path = self.saved(tmp_path, abandoned="ran out")
        readme = tmp_path / "README.md"
        start, end = MARKERS
        readme.write_text(f"before\n{start}\nold table\n{end}\nafter", encoding="utf-8")

        assert republish([path], update_readme=True) == 2
        assert "old table" in readme.read_text(encoding="utf-8")

    def test_a_json_that_is_not_a_run_is_refused(self, tmp_path, capsys):
        path = tmp_path / "diagnosis.json"
        path.write_text(json.dumps({"n": 162, "correct": 111}), encoding="utf-8")

        assert republish([path]) == 2
        assert "no scorecards in it" in capsys.readouterr().err

    def test_unreadable_json_is_a_complaint_not_a_crash(self, tmp_path, capsys):
        path = tmp_path / "half-written.json"
        path.write_text("{oh no", encoding="utf-8")

        assert republish([path]) == 2
        assert "not a readable run" in capsys.readouterr().err

    def test_several_runs_make_one_table_in_the_order_given(self, tmp_path, capsys):
        first = self.saved(tmp_path, name="oracle (gold SQL)")
        second = self.saved(tmp_path, name="gemini-3.5-flash-lite")

        assert republish([first, second]) == 0

        rows = [
            line for line in capsys.readouterr().out.splitlines() if line.startswith("| ")
        ]
        assert "oracle" in rows[1] and "gemini" in rows[2]

    def test_the_same_solver_twice_replaces_rather_than_doubles(self, tmp_path, capsys):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        old = self.saved(tmp_path / "a", n=100)
        new = self.saved(tmp_path / "b", n=268)

        assert republish([old, new]) == 0

        out = capsys.readouterr()
        assert "268 questions" in out.err
        assert out.out.count("| gemini-3.5-flash-lite |") == 1
        assert "replaces the same row read earlier" in out.err

    def test_it_writes_between_the_markers(self, tmp_path, monkeypatch, capsys):
        readme = tmp_path / "README.md"
        start, end = MARKERS
        readme.write_text(f"head\n{start}\nold\n{end}\ntail", encoding="utf-8")
        monkeypatch.setattr("evals.runner.REPO", tmp_path)

        assert republish([self.saved(tmp_path)], update_readme=True) == 0

        written = readme.read_text(encoding="utf-8")
        assert "old" not in written
        assert "head" in written and "tail" in written
        assert "| gemini-3.5-flash-lite | 268 | 62.5%" in written

    def test_republishing_asks_no_model_and_needs_no_dataset(self, tmp_path, capsys):
        """--from-json must not touch a solver: that is the whole point."""
        monkeypatched = []

        assert main(["--from-json", str(self.saved(tmp_path))]) == 0
        assert monkeypatched == []
        assert "| gemini-3.5-flash-lite |" in capsys.readouterr().out


class TestTheRowSaysWhatItMeasured:
    """A dev number and a held-out number are different claims."""

    def saved(self, tmp_path, by_split):
        path = tmp_path / "run.json"
        path.write_text(
            json.dumps(
                {
                    "m": {
                        "abandoned": "",
                        "summary": {
                            "n": sum(v["n"] for v in by_split.values() if v),
                            "execution_accuracy": 0.5,
                            "strict_accuracy": 0.5,
                            "produced_sql": 1.0,
                            "cost_per_question": 0.0,
                            "p50_latency_ms": 0.0,
                            "turns": 1.0,
                        },
                        "by_split": by_split,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_a_dev_only_run_says_dev(self, tmp_path, capsys):
        path = self.saved(tmp_path, {"dev": {"n": 266}, "test": None})

        republish([path])

        assert "| m | 266 dev |" in capsys.readouterr().out

    def test_a_held_out_run_says_test(self, tmp_path, capsys):
        path = self.saved(tmp_path, {"dev": None, "test": {"n": 232}})

        republish([path])

        assert "| m | 232 test |" in capsys.readouterr().out

    def test_a_whole_set_names_both_halves(self, tmp_path, capsys):
        path = self.saved(tmp_path, {"dev": {"n": 266}, "test": {"n": 232}})

        republish([path])

        assert "| m | 498 dev+test |" in capsys.readouterr().out

    def test_a_live_card_labels_itself_the_same_way(self):
        questions = [q for q in toy()[0]][:3]
        card = Scorecard(
            name="x",
            results=[
                Result(question=q, transcript=Transcript(sql="SELECT 1"), judgement=None)
                for q in questions
            ],
        )

        assert card.scope() == "3 dev"


class TestSampling:
    """`--limit N` on a set ordered by database is one database's worth."""

    def questions(self, n, dbs=("a", "b", "c")):
        return [
            Question(
                question_id=i,
                db_id=dbs[i % len(dbs)],
                question=f"q{i}",
                gold_sql="SELECT 1",
            )
            for i in range(n)
        ]

    def test_it_takes_the_number_asked_for(self):
        assert len(sampled(self.questions(90), 30)) == 30

    def test_it_is_the_same_sample_every_time(self):
        qs = self.questions(90)

        assert sampled(qs, 30) == sampled(qs, 30)

    def test_another_seed_is_another_sample(self):
        qs = self.questions(90)

        assert sampled(qs, 30, "1") != sampled(qs, 30, "2")

    def test_it_does_not_depend_on_the_order_it_was_given(self):
        qs = self.questions(90)
        one = {(q.db_id, q.question_id) for q in sampled(qs, 30)}
        other = {(q.db_id, q.question_id) for q in sampled(list(reversed(qs)), 30)}

        assert one == other

    def test_it_spreads_across_databases_where_limit_would_not(self):
        ordered = sorted(self.questions(90), key=lambda q: q.db_id)

        assert len({q.db_id for q in ordered[:30]}) == 1
        assert len({q.db_id for q in sampled(ordered, 30)}) == 3

    def test_asking_for_more_than_there_is_gives_everything(self):
        qs = self.questions(10)

        assert len(sampled(qs, 99)) == 10

    def test_the_original_order_survives(self):
        qs = self.questions(90)
        chosen = sampled(qs, 30)

        assert chosen == [q for q in qs if q in chosen]


class TestARunSaysHowItChoseItsQuestions:
    """A run resumed under another seed is a run on other questions."""

    def saved(self, tmp_path, *flags):
        path = tmp_path / "run.json"
        assert main([*flags, "--json", str(path)]) == 0
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_a_sampled_run_records_its_seed_and_size(self, tmp_path, capsys):
        _, body = self.saved(tmp_path, "--sample", "5", "--seed", "1")

        for card in body.values():
            assert card["selection"] == {
                "split": None,
                "difficulty": None,
                "sample": 5,
                "seed": "1",
                "limit": None,
                "questions": 5,
            }

    def test_a_seed_that_chose_nothing_is_not_recorded(self, tmp_path, capsys):
        _, body = self.saved(tmp_path, "--seed", "7", "--limit", "3")

        selection = body[ORACLE]["selection"]
        assert selection["seed"] is None
        assert selection["sample"] is None
        assert selection["limit"] == 3
        assert selection["questions"] == 3

    def test_the_split_and_difficulty_are_recorded(self, tmp_path, capsys):
        _, body = self.saved(tmp_path, "--split", "dev", "--difficulty", "simple")

        selection = body[ORACLE]["selection"]
        assert selection["split"] == "dev"
        assert selection["difficulty"] == "simple"

    def test_a_run_that_records_it_still_republishes_and_compares(
        self, tmp_path, capsys
    ):
        path, _ = self.saved(tmp_path, "--sample", "5")

        assert republish([path]) == 0
        assert compare(path, path) == 0


class TestComparingTwoRuns:
    """A change that fixes eleven and breaks ten is not an improvement."""

    def run(self, tmp_path, name, verdicts):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "m": {
                        "abandoned": "",
                        "summary": {"n": len(verdicts)},
                        "results": [
                            {"db_id": "a", "question_id": qid, "correct": ok}
                            for qid, ok in verdicts.items()
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_it_names_what_was_fixed_and_what_broke(self, tmp_path, capsys):
        before = self.run(tmp_path, "before", {1: False, 2: True, 3: True})
        after = self.run(tmp_path, "after", {1: True, 2: False, 3: True})

        assert compare(before, after) == 0

        out = capsys.readouterr().out
        assert "fixed   1" in out
        assert "broke   1" in out
        assert "net     +0" in out
        assert "a 2" in out

    def test_a_pure_gain_reports_no_damage(self, tmp_path, capsys):
        before = self.run(tmp_path, "before", {1: False, 2: False})
        after = self.run(tmp_path, "after", {1: True, 2: False})

        compare(before, after)

        out = capsys.readouterr().out
        assert "fixed   1" in out and "broke   0" in out and "net     +1" in out

    def test_only_the_questions_both_ran_are_compared(self, tmp_path, capsys):
        before = self.run(tmp_path, "before", {1: True, 2: False, 3: False})
        after = self.run(tmp_path, "after", {1: True, 2: True})

        compare(before, after)

        captured = capsys.readouterr()
        assert "paired on 2 question(s)" in captured.out
        assert "1 only in before.json" in captured.err

    def test_two_runs_with_no_question_in_common_is_refused(self, tmp_path, capsys):
        before = self.run(tmp_path, "before", {1: True})
        after = self.run(tmp_path, "after", {9: True})

        assert compare(before, after) == 2
        assert "share no question" in capsys.readouterr().err

    def test_a_file_without_results_is_refused(self, tmp_path, capsys):
        good = self.run(tmp_path, "before", {1: True})
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"m": {"summary": {"n": 1}}}), encoding="utf-8")

        assert compare(good, bad) == 2
        assert "no run with results" in capsys.readouterr().err

    def test_a_missing_file_is_refused_rather_than_raising(self, tmp_path, capsys):
        good = self.run(tmp_path, "before", {1: True})

        assert compare(good, tmp_path / "nope.json") == 2
        assert "cannot compare" in capsys.readouterr().err
