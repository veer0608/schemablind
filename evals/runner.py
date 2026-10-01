"""``python -m evals.runner`` -- what each configuration got right, and what it cost.

The headline is execution accuracy, which is BIRD's own metric and therefore
means something to someone who has never seen this repo. Everything beside it
is there because a single accuracy number hides the things worth knowing: what
a question cost, how many turns it took, and -- when it was wrong -- whether the
query failed to run at all or ran and returned the wrong rows. Those are
different problems with different fixes.

There is no free baseline here the way a regex parser was free in moneytrail;
nothing generates SQL for nothing. So the rows are configurations rather than
one parser against another, and the comparisons that matter are ablations:
schema-blind against schema-given, repair on against repair off.

Two rows are free and always available, and they exist to prove the scorer
rather than to compete: an oracle that submits the gold query, which must score
100%, and a mute agent that submits nothing, which must score 0%. If either
drifts, the harness is broken and no model number from it is worth reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from schemablind.agent import Agent, Transcript
from schemablind.llm import QuotaExhausted, Usage, build_client
from schemablind.sandbox import Sandbox
from schemablind.scoring import CORRECT, Judgement, category, judge

from .checkpoint import Checkpoint
from .dataset import Question, bird, database_for, load_questions, toy

HERE = Path(__file__).parent
REPO = HERE.parent
SPLITS = ("dev", "test")
TEST_SHARE = 40
ORACLE = "oracle (gold SQL)"
MUTE = "mute (no SQL)"


QUARANTINE = HERE / "quarantine.json"


def quarantined() -> set[tuple[str, int]]:
    """Held-out questions whose answers have since been looked at.

    A question stops being held out the moment someone reads its gold SQL while
    debugging, whatever the split says. That happened here: failures were
    printed without filtering by split first, so five test-half questions were
    inspected. Deleting them from the test set is the only honest repair --
    quietly leaving them in would report a number partly measured on questions
    whose answers were known.

    Recorded in a file rather than fixed in code so the list can only grow, and
    so anyone reading the result can see exactly what was excluded and why.
    """
    if not QUARANTINE.is_file():
        return set()
    raw = json.loads(QUARANTINE.read_text(encoding="utf-8"))
    return {(entry["db_id"], entry["question_id"]) for entry in raw.get("ids", [])}


def split_of(question: Question) -> str:
    """Which half a question belongs to, derived from the question itself.

    A prompt improved by reading the questions it failed, then scored on those
    same questions, reports a number fitted to its own answer key. Hashing the
    text means whoever tunes the prompt does not get to choose what they are
    marked on, and anyone who doubts the split can recompute it.

    There is deliberately no salt to turn. The twelve toy questions all landed
    in dev, which is a 1-in-450 coincidence and looks like something to fix --
    and adding a salt until the split came out nicer would be choosing the test
    set by hand, which is the one thing this function exists to prevent. It is
    left alone, and `warn_about_split` says so out loud instead. At the size of
    a real dataset the question does not arise.
    """
    seed = f"{question.db_id}:{question.question_id}:{question.question.strip().lower()}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return "test" if int(digest[:8], 16) % 100 < TEST_SHARE else "dev"


# --- running ---------------------------------------------------------------

#: Given a question and its database, produce a transcript.
Solver = Callable[[Question, Sandbox], Transcript]


def oracle_solver(question: Question, sandbox: Sandbox) -> Transcript:
    """Submits the gold query. Must score 100%, or the scorer is wrong."""
    return Transcript(sql=question.gold_sql, stopped="answered", turns=0)


def mute_solver(question: Question, sandbox: Sandbox) -> Transcript:
    """Submits nothing. Must score 0%, or the scorer is scoring absence."""
    return Transcript(sql=None, stopped="stopped without a query", turns=0)


def agent_solver(agent: Agent) -> Solver:
    def solve(question: Question, sandbox: Sandbox) -> Transcript:
        return agent.solve(question.question, sandbox, evidence=question.evidence)

    return solve


@dataclass
class Result:
    question: Question
    transcript: Transcript
    judgement: Judgement | None

    @property
    def correct(self) -> bool:
        return bool(self.judgement and self.judgement.correct)

    @property
    def exact(self) -> bool:
        return bool(self.judgement and self.judgement.exact)

    @property
    def produced_sql(self) -> bool:
        return bool(self.transcript.sql)

    @property
    def failure(self) -> str:
        if self.correct:
            return CORRECT
        if not self.produced_sql:
            return "no query produced"
        return category(self.judgement) if self.judgement else "unscored"


@dataclass
class Scorecard:
    name: str
    results: list[Result] = field(default_factory=list)
    abandoned: str = ""

    @property
    def complete(self) -> bool:
        return not self.abandoned

    def within(self, split: str | None = None, difficulty: str | None = None):
        rows = self.results
        if split:
            rows = [r for r in rows if split_of(r.question) == split]
        if difficulty:
            rows = [r for r in rows if r.question.difficulty == difficulty]
        return rows

    def summary(self, split: str | None = None, difficulty: str | None = None) -> dict:
        rows = self.within(split, difficulty)
        if not rows:
            return {}
        costs = [
            r.transcript.cost_usd for r in rows if r.transcript.cost_usd is not None
        ]
        unpriced = any(
            r.transcript.usage and r.transcript.cost_usd is None for r in rows
        )
        latencies = [r.transcript.latency_ms for r in rows if r.transcript.usage]
        return {
            "n": len(rows),
            "execution_accuracy": sum(r.correct for r in rows) / len(rows),
            "strict_accuracy": sum(r.exact for r in rows) / len(rows),
            "produced_sql": sum(r.produced_sql for r in rows) / len(rows),
            "cost_per_question": (
                sum(costs) / len(rows) if costs else (None if unpriced else 0.0)
            ),
            "total_cost": sum(costs) if costs else (None if unpriced else 0.0),
            "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "turns": statistics.mean([r.transcript.turns for r in rows]),
            "repairs": statistics.mean([r.transcript.repairs for r in rows]),
            "tokens": statistics.mean([r.transcript.total_tokens for r in rows]),
        }

    def scope(self) -> str:
        """What the row was measured on, for the table to say out loud.

        A number from the tuning half and a number from the held-out half are
        different claims, and a table whose only label is the model name lets
        the weaker one be read as the stronger.
        """
        present = {split_of(r.question) for r in self.results}
        return _scope(len(self.results), present)

    def failures(self, split: str | None = None) -> dict[str, int]:
        counted: dict[str, int] = {}
        for result in self.within(split):
            if result.correct:
                continue
            counted[result.failure] = counted.get(result.failure, 0) + 1
        return dict(sorted(counted.items(), key=lambda kv: -kv[1]))


def score(
    name: str,
    solver: Solver,
    questions: Sequence[Question],
    databases: Path,
    *,
    delay: float = 0.0,
    on_progress: Callable[[int, int], None] | None = None,
    cache: Checkpoint | None = None,
) -> Scorecard:
    card = Scorecard(name=name)
    for index, question in enumerate(questions, 1):
        try:
            path = database_for(question.db_id, databases)
        except FileNotFoundError as exc:
            card.abandoned = str(exc)
            return card

        with Sandbox(path) as sandbox:
            answered = cache.get(name, question) if cache is not None else None
            if answered is not None:
                # Already paid for on an earlier run. Re-judged below rather
                # than trusted, so the scorer stays the only thing deciding.
                transcript = answered
            else:
                if delay and index > 1:
                    time.sleep(delay)
                try:
                    transcript = solver(question, sandbox)
                except QuotaExhausted as exc:
                    # Every question after this scores zero for never being
                    # asked, which a scorecard cannot tell apart from getting
                    # them wrong. With a checkpoint the answers so far survive,
                    # and the next run continues from here.
                    card.abandoned = (
                        f"stopped after {index - 1} of {len(questions)}: {exc}"
                    )
                    return card
                if cache is not None:
                    cache.put(name, question, transcript)
            verdict = (
                judge(transcript.sql, question.gold_sql, sandbox)
                if transcript.sql
                else None
            )
            card.results.append(Result(question, transcript, verdict))
        if on_progress:
            on_progress(index, len(questions))
    return card


# --- reporting -------------------------------------------------------------

MARKERS = ("<!-- SCORECARD -->", "<!-- /SCORECARD -->")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _scope(n: int, splits: set[str]) -> str:
    """`266 dev`, `232 test`, `498 dev+test` -- the claim the row is making."""
    named = "+".join(s for s in SPLITS if s in splits)
    return f"{n} {named}" if named else str(n)


def _money(value: float | None) -> str:
    if value is None:
        return "unpriced"
    if value == 0:
        return "$0"
    return f"${value:.6f}".rstrip("0").rstrip(".")


def table(cards: Sequence[Scorecard], split: str | None = None) -> str:
    head = (
        f"  {'configuration':<26}{'exec acc':>10}{'strict':>9}{'ran':>8}"
        f"{'turns':>7}{'$/q':>12}{'p50':>9}"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    for card in cards:
        if not card.complete:
            continue
        s = card.summary(split)
        if not s:
            continue
        lines.append(
            f"  {card.name[:26]:<26}"
            f"{_pct(s['execution_accuracy']):>10}"
            f"{_pct(s['strict_accuracy']):>9}"
            f"{_pct(s['produced_sql']):>8}"
            f"{s['turns']:>7.1f}"
            f"{_money(s['cost_per_question']):>12}"
            f"{s['p50_latency_ms']:>6.0f} ms"
        )
    return "\n".join(lines)


#: Below this a held-out half is too thin to conclude anything from.
USABLE_SPLIT = 20


def warn_about_split(questions: Sequence[Question]) -> str:
    """Say when the held-out half cannot carry the weight of a conclusion.

    An empty or tiny test half is not a failure to hide -- it is a fact about
    how much this set can support, and a reader who is not told will read a
    percentage off a handful of questions as though it meant something.
    """
    held_out = sum(split_of(q) == "test" for q in questions)
    if held_out == 0:
        return (
            "NOTE: no questions landed in the held-out half. Nothing here can be "
            "reported as an out-of-sample result."
        )
    if held_out < USABLE_SPLIT:
        return (
            f"NOTE: only {held_out} questions in the held-out half. Too few to "
            f"separate a real difference from noise -- treat it as a smoke test."
        )
    return ""


def report(cards: Sequence[Scorecard], questions: Sequence[Question]) -> str:
    dbs = sorted({q.db_id for q in questions})
    by_difficulty = {
        d: sum(q.difficulty == d for q in questions)
        for d in ("simple", "moderate", "challenging")
    }
    out = [
        "schemablind -- schema-blind SQL agent, execution accuracy",
        f"{len(questions)} questions over {len(dbs)} database(s): "
        + ", ".join(f"{n} {d}" for d, n in by_difficulty.items() if n),
        f"split: {sum(split_of(q) == 'dev' for q in questions)} dev, "
        f"{sum(split_of(q) == 'test' for q in questions)} test "
        f"(by hash of the question, so tuning cannot pick its own marking)",
        *([warning] if (warning := warn_about_split(questions)) else []),
        "",
        "all questions",
        table(cards),
        "",
    ]
    for split in SPLITS:
        if any(c.summary(split) for c in cards if c.complete):
            out += [f"{split} half", table(cards, split), ""]

    for card in cards:
        if not card.complete:
            out += [
                f"  {card.name}: NOT SCORED -- {card.abandoned}",
                "    the questions it never reached would count as answers it "
                "got wrong, so it gets no row rather than a misleading one",
                "",
            ]
            continue
        failures = card.failures()
        if failures:
            out.append(
                f"  {card.name} missed {sum(failures.values())}: "
                + ", ".join(f"{n} {why}" for why, n in failures.items())
            )
    return "\n".join(out)


def markdown(cards: Sequence[Scorecard], questions: Sequence[Question]) -> str:
    rows = [
        "| configuration | questions | execution accuracy | strict | produced SQL | turns | $/question | p50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for card in cards:
        if not card.complete:
            continue
        s = card.summary()
        if not s:
            continue
        rows.append(
            f"| {card.name} | {card.scope()} | {_pct(s['execution_accuracy'])} "
            f"| {_pct(s['strict_accuracy'])} | {_pct(s['produced_sql'])} "
            f"| {s['turns']:.1f} | {_money(s['cost_per_question'])} "
            f"| {s['p50_latency_ms']:.0f} ms |"
        )
    return "\n".join(rows)


@dataclass(frozen=True)
class SavedCard:
    """A finished run read back from its JSON, standing in for a Scorecard.

    It carries the summary the run computed rather than recomputing one, so a
    republished row is the number that was actually paid for. It answers the
    three things the table asks of a card and nothing else.

    The one thing this cannot do is notice a scorer that has changed since.
    Verdicts here were reached at run time; the checkpoint is the artefact that
    re-judges on every load. So after touching `scoring`, reload the run from
    its checkpoint rather than republishing its JSON.
    """

    name: str
    abandoned: str
    saved: dict
    by_split: dict

    @property
    def complete(self) -> bool:
        return not self.abandoned

    def summary(self) -> dict:
        return self.saved

    def scope(self) -> str:
        present = {s for s in SPLITS if (self.by_split.get(s) or {}).get("n")}
        return _scope(int(self.saved.get("n") or 0), present)


def load_cards(paths: Sequence[Path]) -> tuple[list[SavedCard], list[str]]:
    """Cards from saved runs, in the order given, plus what was wrong."""
    cards: dict[str, SavedCard] = {}
    complaints: list[str] = []
    for path in paths:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            complaints.append(f"{path}: not a readable run ({exc})")
            continue
        if not isinstance(body, dict):
            complaints.append(f"{path}: not a run -- the top level is not an object")
            continue
        found = 0
        for name, card in body.items():
            if not isinstance(card, dict) or "summary" not in card:
                continue
            found += 1
            if name in cards:
                complaints.append(f"{path}: {name} replaces the same row read earlier")
            cards[name] = SavedCard(
                name=name,
                abandoned=str(card.get("abandoned") or ""),
                saved=card.get("summary") or {},
                by_split=card.get("by_split") or {},
            )
        if not found:
            complaints.append(f"{path}: no scorecards in it")
    return list(cards.values()), complaints


def republish(paths: Sequence[Path], *, update_readme: bool = False) -> int:
    """Write the table from runs already paid for, calling no model.

    Re-running to refresh a README spends a day's allowance to reproduce a
    number that is already on disk, and can quietly produce a *different* one.
    """
    cards, complaints = load_cards(paths)
    for complaint in complaints:
        print(f"  {complaint}", file=sys.stderr)

    for card in cards:
        if card.complete:
            n = card.summary().get("n", "?")
            print(f"  {card.name}: {n} questions", file=sys.stderr)
        else:
            # Silence here would look like a row that simply did not exist.
            print(
                f"  {card.name}: not published -- {card.abandoned}", file=sys.stderr
            )

    body = markdown(cards, ())
    if not any(card.complete for card in cards):
        print(
            "nothing complete to publish -- refusing to empty the scorecard",
            file=sys.stderr,
        )
        return 2

    print()
    print(body)
    if update_readme:
        if not splice(REPO / "README.md", body):
            print("README.md has no scorecard markers", file=sys.stderr)
            return 1
        print("\nwrote the scorecard into README.md")
    return 0


def sampled(questions: Sequence[Question], n: int, seed: str = "0") -> list[Question]:
    """N questions by hash, keeping the original order among those chosen.

    `--limit N` takes the first N, which on a set ordered by database is one or
    two databases and nothing else -- and the per-database fail rates on the dev
    half run from 12% to 61%, so that subset is not the half in miniature. This
    spreads the choice, and being a hash rather than `random.sample` it is the
    same subset on every machine and every rerun, which is what makes two arms
    of an experiment comparable.
    """
    order = sorted(
        range(len(questions)),
        key=lambda i: hashlib.sha256(
            f"{seed}:{questions[i].db_id}:{questions[i].question_id}".encode()
        ).hexdigest(),
    )
    chosen = set(order[:n])
    return [q for i, q in enumerate(questions) if i in chosen]


def selection_of(args: argparse.Namespace, asked: int) -> dict:
    """How a run chose its questions, saved beside what it answered.

    An abandoned run is resumed by repeating its command, and the results it
    saved do not say what that was: `--sample 75` under another seed is 75
    other questions, so the checkpoint is passed over and a day's allowance
    goes on a run that pairs with nothing. The seed is kept only when it chose
    something.
    """
    return {
        "split": args.split,
        "difficulty": args.difficulty,
        "sample": args.sample,
        "seed": args.seed if args.sample else None,
        "limit": args.limit,
        "questions": asked,
    }


def saved_selection(path: Path) -> dict | None:
    """The selection a saved run recorded, or None when it recorded none.

    A file from before the block existed, or one that is not a run at all, has
    nothing to compare against, and that is not a reason to refuse anything.
    """
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    for card in body.values():
        if isinstance(card, dict) and isinstance(card.get("selection"), dict):
            return card["selection"]
    return None


def resumes_another_selection(args: argparse.Namespace, asked: int) -> str:
    """Why this resume is not the run its --json holds, or '' when it is.

    Only a resume is checked: a --checkpoint and a --json that already exists.
    Recording the seed made the right command readable; this is what stops the
    wrong one before it spends anything.
    """
    if not (args.checkpoint and args.json and args.json.exists()):
        return ""
    before = saved_selection(args.json)
    if before is None:
        return ""
    now = selection_of(args, asked)
    differing = [k for k in now if before.get(k) != now[k]]
    if not differing:
        return ""
    return (
        f"{args.json} was not chosen the way this run is: "
        + ", ".join(f"{k} was {before.get(k)!r}, now {now[k]!r}" for k in differing)
        + ". Repeat its selection to resume it, or give this run its own --json."
    )


def _rows_of(path: Path) -> tuple[str, dict[tuple[str, int], bool]]:
    """One saved run's verdicts, keyed by question. Name of the card it read."""
    body = json.loads(path.read_text(encoding="utf-8"))
    for name, card in body.items():
        if isinstance(card, dict) and card.get("results"):
            return name, {
                (r["db_id"], r["question_id"]): bool(r["correct"])
                for r in card["results"]
            }
    raise ValueError(f"{path}: no run with results in it")


def compare(before: Path, after: Path) -> int:
    """What a change fixed and what it broke, on the questions both ran.

    Two averages cannot tell you this. A change that fixes eleven questions and
    breaks ten looks like +0.4 points and is actually a coin toss with extra
    steps, and the only way to see that is per question.
    """
    try:
        first_name, first = _rows_of(before)
        second_name, second = _rows_of(after)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"cannot compare: {exc}", file=sys.stderr)
        return 2

    shared = sorted(set(first) & set(second))
    if not shared:
        print("nothing to compare: the two runs share no question", file=sys.stderr)
        return 2

    only_first = len(first) - len(shared)
    only_second = len(second) - len(shared)
    if only_first or only_second:
        print(
            f"  comparing the {len(shared)} question(s) both ran; "
            f"{only_first} only in {before.name}, {only_second} only in {after.name}",
            file=sys.stderr,
        )

    fixed = [k for k in shared if not first[k] and second[k]]
    broke = [k for k in shared if first[k] and not second[k]]
    was = sum(first[k] for k in shared)
    now = sum(second[k] for k in shared)

    print()
    print(f"paired on {len(shared)} question(s)")
    print(f"  {first_name} -> {second_name}")
    print(f"  before  {was}/{len(shared)}  {_pct(was / len(shared))}")
    print(f"  after   {now}/{len(shared)}  {_pct(now / len(shared))}")
    print(f"  fixed   {len(fixed)}")
    print(f"  broke   {len(broke)}")
    print(f"  net     {now - was:+d}")
    if broke:
        # A change is not free because the average moved up. Name the damage.
        print("  broke: " + ", ".join(f"{db} {qid}" for db, qid in broke[:12]))
    return 0


def splice(readme: Path, body: str) -> bool:
    start, end = MARKERS
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n\n{body}\n\n{end}{tail}", encoding="utf-8")
    return True


# --- entry point -----------------------------------------------------------


def build_solvers(specs: Sequence[str], **agent_kwargs) -> list[tuple[str, Solver]]:
    built: list[tuple[str, Solver]] = []
    for spec in specs:
        if spec == "oracle":
            built.append((ORACLE, oracle_solver))
            continue
        if spec == "mute":
            built.append((MUTE, mute_solver))
            continue
        provider, _, model = spec.rpartition(":")
        client = build_client(provider or None, model or None)
        if client is None:
            print(f"  skipping {spec!r}: no API key configured", file=sys.stderr)
            continue
        agent = Agent(client, **agent_kwargs)
        label = client.model + ("" if agent.repair else " (no repair)")
        built.append((label, agent_solver(agent)))
    return built


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m evals.runner")
    parser.add_argument(
        "--solvers",
        default="oracle,mute",
        help=(
            "comma-separated. 'oracle' and 'mute' are free and prove the "
            "scorer; anything else is a model, optionally 'provider:model'"
        ),
    )
    parser.add_argument(
        "--bird",
        action="store_true",
        help="use BIRD Mini-Dev from data/ instead of the toy set",
    )
    parser.add_argument("--questions", type=Path, help="a BIRD-shaped questions json")
    parser.add_argument("--databases", type=Path, help="the directory of sqlite files")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--difficulty", choices=("simple", "moderate", "challenging"))
    parser.add_argument("--limit", type=int, help="first N questions -- for a smoke run")
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help=(
            "N questions chosen by hash rather than the first N, so a cheap run "
            "is not one database's worth -- reproducible from --seed"
        ),
    )
    parser.add_argument("--seed", default="0", help="which sample --sample takes")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help=(
            "pair two saved runs by question and report what a change fixed and "
            "broke, which a difference of two averages cannot tell you"
        ),
    )
    parser.add_argument("--no-repair", action="store_true", help="make the first query final")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "append each answered question here and reuse it on a later run, "
            "so a run stopped by the daily token cap resumes instead of restarting"
        ),
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        nargs="+",
        metavar="RUN",
        help=(
            "republish saved runs instead of running anything -- the only way "
            "to write a number that was already paid for without paying again"
        ),
    )
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        # argparse interpolates help strings, so a bare % raises
        # "ValueError: incomplete format" and takes --help down with it
        help="CI mode: the oracle must score 100%% and the mute agent 0%%",
    )
    args = parser.parse_args(argv)

    if args.from_json:
        return republish(args.from_json, update_readme=args.update_readme)
    if args.compare:
        return compare(*args.compare)

    if args.questions and args.databases:
        questions, databases = load_questions(args.questions), args.databases
    elif args.bird:
        questions, databases = bird()
    else:
        questions, databases = toy()

    burned = quarantined()
    if burned:
        before = len(questions)
        questions = [
            q for q in questions if (q.db_id, q.question_id) not in burned
        ]
        dropped = before - len(questions)
        if dropped:
            print(
                f"  excluding {dropped} quarantined question(s) -- their answers "
                f"were seen while debugging; see evals/quarantine.json",
                file=sys.stderr,
            )
    if args.split:
        questions = [q for q in questions if split_of(q) == args.split]
    if args.difficulty:
        questions = [q for q in questions if q.difficulty == args.difficulty]
    if args.sample:
        questions = sampled(questions, args.sample, args.seed)
        print(
            f"  sample of {len(questions)} (seed {args.seed}) over "
            f"{len(({q.db_id for q in questions}))} database(s) -- a paired "
            f"comparison, not a score",
            file=sys.stderr,
        )
    if args.limit:
        questions = questions[: args.limit]

    if args.check:
        return _check(questions, databases)

    mismatch = resumes_another_selection(args, len(questions))
    if mismatch:
        print(f"not resuming: {mismatch}", file=sys.stderr)
        return 2

    specs =[s.strip() for s in args.solvers.split(",") if s.strip()]
    solvers = build_solvers(
        specs, max_turns=args.max_turns, repair=not args.no_repair
    )
    if not solvers:
        print("nothing to run", file=sys.stderr)
        return 2

    cache = Checkpoint(args.checkpoint).load() if args.checkpoint else None
    if cache is not None and len(cache):
        # Said before the run, not after: a checkpoint this selection mostly
        # misses is the one sign of a wrong resume that an older --json, with
        # no selection in it, can still give.
        reusable = sum(
            cache.holds(name, q)
            for name, _ in solvers
            if name not in (ORACLE, MUTE)
            for q in questions
        )
        print(
            f"  checkpoint {args.checkpoint}: {len(cache)} already answered, "
            f"{reusable} of them among the {len(questions)} this run asks",
            file=sys.stderr,
        )

    cards = []
    for name, solver in solvers:
        started = time.perf_counter()

        def progress(done, total, _n=name, _s=started):
            if done % 5 and done != total:
                return
            spent = time.perf_counter() - _s
            print(
                f"    {_n}: {done}/{total}  {spent:.0f}s spent, "
                f"~{(spent / done) * (total - done):.0f}s left",
                file=sys.stderr,
                flush=True,
            )

        cards.append(
            score(
                name,
                solver,
                questions,
                databases,
                delay=args.delay,
                on_progress=None if name in (ORACLE, MUTE) else progress,
                cache=None if name in (ORACLE, MUTE) else cache,
            )
        )
        print(f"  ran {name} in {time.perf_counter() - started:.1f}s", file=sys.stderr)
        if cache is not None and cache.resumed:
            print(
                f"    {cache.resumed} of those came from the checkpoint",
                file=sys.stderr,
            )
            cache.resumed = 0

    print()
    print(report(cards, questions))

    if args.json:
        selection = selection_of(args, len(questions))
        args.json.write_text(
            json.dumps(
                {
                    c.name: {
                        "abandoned": c.abandoned,
                        "selection": selection,
                        "summary": c.summary(),
                        "by_split": {s: c.summary(s) for s in SPLITS},
                        "failures": c.failures(),
                        "results": [
                            {
                                "question_id": r.question.question_id,
                                "db_id": r.question.db_id,
                                "question": r.question.question,
                                "difficulty": r.question.difficulty,
                                "split": split_of(r.question),
                                "gold_sql": r.question.gold_sql,
                                "predicted_sql": r.transcript.sql,
                                "correct": r.correct,
                                "exact": r.exact,
                                "failure": r.failure,
                                "turns": r.transcript.turns,
                                "repairs": r.transcript.repairs,
                                "nudges": r.transcript.nudges,
                                # How the answer was reached, and what it ran
                                # last. Without these a miss cannot be told
                                # apart from a bug in the harness, which cost a
                                # live question to work out once already.
                                "answered_via": r.transcript.answered_via,
                                "last_query": r.transcript.last_query,
                                "stopped": r.transcript.stopped,
                                "error": r.transcript.error,
                                "tools": r.transcript.tools_used,
                                "tool_args": [
                                    {name: args} for name, args in r.transcript.tool_calls
                                ],
                                "prompt_tokens": r.transcript.prompt_tokens,
                                "completion_tokens": r.transcript.completion_tokens,
                                "cost_usd": r.transcript.cost_usd,
                                "latency_ms": round(r.transcript.latency_ms, 1),
                            }
                            for r in c.results
                        ],
                    }
                    for c in cards
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    if args.update_readme:
        if not splice(REPO / "README.md", markdown(cards, questions)):
            print("README.md has no scorecard markers", file=sys.stderr)
            return 2
        print("wrote the scorecard into README.md")
    return 0


def _check(questions: Sequence[Question], databases: Path) -> int:
    """The harness proving itself, with no model and no key."""
    perfect = score(ORACLE, oracle_solver, questions, databases)
    silent = score(MUTE, mute_solver, questions, databases)
    good = perfect.summary()["execution_accuracy"]
    bad = silent.summary()["execution_accuracy"]
    print(f"oracle {_pct(good)} on {len(questions)} questions; mute {_pct(bad)}")
    if good < 1.0:
        print("the gold queries do not score themselves -- the scorer is wrong")
        for result in perfect.results:
            if not result.correct:
                print(f"  [{result.failure}] {result.question.question}")
        return 1
    if bad > 0.0:
        print("an agent that produced nothing scored above zero")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
