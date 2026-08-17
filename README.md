# schemablind

[![ci](https://github.com/veer0608/schemablind/actions/workflows/ci.yml/badge.svg)](https://github.com/veer0608/schemablind/actions/workflows/ci.yml)

A SQL agent that is **given no schema**. It gets a database it has never seen,
four verbs, and a question — and has to find its own way to the answer.

```
> Which county has the highest average math score among its charter schools?

  list_tables       -> schools, scores, students
  describe_table    -> schools(sch_id PK, sch_name, cnty, charter, opened)
  sample_rows       -> charter is 0/1, not 'Y'/'N'
  run_sql_readonly  -> ERROR: no such column: county
  run_sql_readonly  -> Marin
  final_sql

  SELECT s.cnty FROM schools s
  JOIN students st ON st.sch_id = s.sch_id
  JOIN scores sc ON sc.stu_id = st.stu_id
  WHERE s.charter = 1 AND sc.subject = 'math'
  GROUP BY s.cnty ORDER BY AVG(sc.score) DESC LIMIT 1
```

Scored on **execution accuracy** — run the agent's query and the reference
query against the same database and compare the rows. That is BIRD's own
metric, which is the point: the number means something to someone who has never
seen this repo.

---

## Why schema-blind

Writing SQL against a schema you were handed is a different and much easier
task than working out which schema you are looking at. Most text-to-SQL
benchmarking measures the first. Being handed a database and a question is the
second.

So the agent starts with nothing and has four verbs:

| verb | what it is for |
|---|---|
| `list_tables` | what exists at all |
| `describe_table` | columns, types, and foreign keys — the join path |
| `sample_rows` | what values actually look like: is a flag `0/1` or `'Y'/'N'` |
| `run_sql_readonly` | try it and see, including seeing it fail |

`sample_rows` earns its place more than it looks. Guessing an encoding is the
most common way to write a query that runs cleanly and returns the wrong
answer, and that failure is invisible without looking at real values.

## The ground truth is free

Nobody decides whether the SQL "looks right". There is no rubric and no judge
model to be wrong in its own interesting ways. Two queries either return the
same rows against the same database or they do not.

Two caveats are kept in the open rather than discovered later:

- **Set comparison ignores duplicates.** That is BIRD's convention and it is
  kept, but a stricter multiset comparison is reported beside it. The gap
  between the two columns is the population of queries with a duplicate-row bug
  that the official metric forgives.
- **A wrong query can be right by accident.** `WHERE cnty = 'Marin'` and
  `WHERE sch_id > 2` may select the same rows here and different rows anywhere
  else. That is a property of execution accuracy, not of the agent.

A reference query that fails to run is never charged to the agent — it is a
broken dataset row, and scoring it as a miss would lower every model's number
by the same amount, silently.

## The agent cannot write

It executes SQL a model composed, from a prompt containing data read out of the
database. That is untrusted input, and it is treated as such — four layers,
because each has a hole on its own:

| layer | what it stops |
|---|---|
| opened `mode=ro` | the driver will not write, whatever gets past the rest |
| `PRAGMA query_only` | refuses writes again inside the connection |
| only `SELECT`/`WITH` | `ATTACH` cannot reach a second file |
| one statement per call | nothing hides behind a semicolon |

Plus a wall-clock limit and a row cap, which are survival rather than security:
a cross join is a perfectly valid `SELECT` and will sit there until the process
dies. CI asserts the database is byte-identical after trying to delete, drop,
update, attach and smuggle a write past a semicolon.

Schema discovery needs `PRAGMA table_info`, which that policy refuses — so
there are two paths, and they never mix. Fixed statements the project wrote go
through `introspect`; anything a model composed goes through `query`.

## Usage

Look around a database for free — no model, no key:

```bash
python -m schemablind schema evals/toy/databases/school/school.sqlite
```

Ask it something:

```bash
python -m schemablind ask "which county has the highest average math score among charter schools?" evals/toy/databases/school/school.sqlite --model openai/gpt-oss-20b
```

It prints the query, the rows, the turns and tools it took to get there, and
what it cost. Pass `--gold "SELECT ..."` and it scores itself.

Run the eval on the toy set that ships with the repo:

```bash
python -m evals.runner --solvers oracle,mute
```

For the real thing, fetch BIRD Mini-Dev — 500 questions over 11 databases,
764MB zipped and 3.3GB unpacked, into the gitignored `data/`:

```bash
curl -L -o data/minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
```

Unzip it there, then `--bird` selects it. The loader needs no changes: the toy
set was written in BIRD's own shape so that arriving here would be a path
change and nothing else.

```bash
python -m evals.runner --bird --solvers oracle
```

### A run that does not finish in one day

500 questions do not fit inside a free tier's daily allowance, and a run stopped
partway is still worth what it already paid for. `--checkpoint` appends each
answered question as it is answered, and reuses it next time:

```bash
python -m evals.runner --bird --solvers openai/gpt-oss-120b --checkpoint runs/held-out.jsonl
```

Run it again tomorrow with the same file and it picks up where the allowance ran
out. Only the model's half is stored — judging is local and free, so it is redone
on every load and a change to the scorer can never be masked by a verdict
recorded under the old one.

This does not weaken what a complete card means. A run that ends early is still
abandoned rather than scored; the checkpoint only means the next run starts from
question 91 instead of question 1, until one of them reaches the end and the card
is complete for real.

## The scorecard

<!-- SCORECARD -->

| configuration | execution accuracy | strict | produced SQL | turns | $/question | p50 |
|---|---|---|---|---|---|---|
| oracle (gold SQL) | 100.0% | 100.0% | 100.0% | 0.0 | $0 | 0 ms |
| mute (no SQL) | 0.0% | 0.0% | 0.0% | 0.0 | $0 | 0 ms |

<!-- /SCORECARD -->

Those two rows are not competitors — they are the harness proving itself. The
oracle submits the reference query and must score 100%; the mute agent submits
nothing and must score 0%. If either ever drifts, the scorer is broken and no
model number produced by it is worth reading. CI runs exactly that check, and
it needs no key. The oracle also scores 100% on all 500 real Mini-Dev
questions, so the scorer holds against real data and not only a fixture built
to suit it.

### Where the agent actually stands

**There is no complete model run yet, so there is no model row.** A run that
hits the daily token cap is abandoned rather than scored — the questions it
never reached would count as answers it got wrong. Free-tier budget has ended
three runs that way so far.

Those three runs also threw away every answer they had already paid for, which
is why the cap was fatal rather than merely slow. It is not any more:
`--checkpoint` keeps each answered question, so a capped run now resumes instead
of restarting, and a complete card can be assembled across several days of
allowance. The scoring rule is unchanged — an unfinished run is still not a
number.

What there is, on `openai/gpt-oss-120b`, is indicative and small:

| | n | execution accuracy |
|---|---|---|
| dev half | 14 | 43% |
| **held-out half** | **12** | **50%** |

At n=12 the interval around 50% runs roughly from a quarter to three quarters,
so this is a direction rather than a measurement. What it does show is that the
prompt rules **generalised**: they were written from dev failures and scored no
worse on questions they had never seen. The same discipline on a sister project
caught the opposite result, which is the point of running it.

Getting from here to a real number is the rest of the held-out half: roughly 180
more questions — 40% of 500, less the five quarantined, less the twelve already
answered. Roughly, because that denominator has never actually been printed.
`--bird --split test --solvers oracle` settles it exactly, for free, before a
single token is spent.

Those twelve do not carry over. The run JSON keeps `predicted_sql` but not the
transcript, and the checkpoint needs the transcript — so they are a measurement
that already happened, not a down payment on the next one.

#### What changed it

Starting point was 16%. Three things moved it, and only one was the model's
fault:

- **An API failure discarded queries the agent had already verified.** Six of
  nineteen questions were scored "produced no query" when a good query was
  sitting in the transcript and the client had fallen over on a later turn.
  That was written up as a finding about the agent before it was recognised as
  a bug — a failure of the client reported as a failure of reasoning.
- **Selecting the column it ranked by.** "Who spent the most" was answered with
  the person *and* the total, which is exactly one column wrong. A prompt rule
  putting the measure in `ORDER BY` rather than `SELECT` fixed several.
- **Rounding.** `ROUND(x, 2)` is a different number from `x` and is judged
  different.

Turns fell from 9.0 to 5.6 and tokens from 13,024 to 8,576 over the same
questions, from batching `describe_table` into one call that also carries
sample values. That is a third less budget per question, and on a
request-metered provider it is the difference between three questions a day and
five.

## What this will cost, before it is spent

Worth stating up front, because it is the thing that decides whether the
project is finishable rather than a detail discovered halfway through.

A schema-blind agent is not a single prompt: on real BIRD databases it runs
**9.2 turns and 13,600 tokens per question**, measured over 19 live questions.
Two earlier estimates were both wrong and both too low — reasoning from schema
sizes gave 8,750, and the toy set suggested 9,500. Real questions need more
turns, and turns are what cost.

Batching `describe_table` later brought that down to 5.5 turns and 8,360 tokens
— but on the **dev** half, and the run this project still owes is the held-out
one. There the same agent costs **11,816 tokens over 6.6 turns**, 41% more. So
a third estimate was wrong for a third reason, and this one is the least
excusable: not optimism about the agent, just a number measured on the half that
was cheap and quietly generalised to the half that is not.

Planning therefore uses the held-out figure:

| | free tier | paid |
|---|---|---|
| per question, held-out half | ~11,800 tokens | ~$0.0021 |
| daily cap | 200k tokens per model | none |
| **questions/day/model** | **~16** | unlimited |
| **held-out half (~195 q) × 3 models** | **~13 days** | **~$0.75** |
| **all 500 × 3 models** | **~32 days** | **~$1.90** |

Two things make that cheaper than the earlier estimate rather than dearer,
despite the per-question cost going up. The cap is **per model** — 200,000 is
what the 429 names for `openai/gpt-oss-120b` — and `run()` abandons one card
without touching the next, so three solvers in one invocation draw on three
independent budgets and take one model's wall clock, not three. And the card
this project owes is the held-out one: paying for all 500 buys 305 answers on
the half the prompt was tuned against, whose number should not be quoted anyway.

The paid column assumes the cheap end of the Groq basket and the token split
measured on `gpt-oss-120b`; a weaker model that needs more turns will cost more
than its price-per-token suggests.

The free tier's daily token cap is what blocks this, not the money. And that
daily limit appears in **no response header**: the per-minute bucket reads a
healthy 12,000 while the daily budget is already gone, and the real limit is
only named in the body of the 429 that eventually refuses you. So a daily
refusal is its own error type here, and a run that hits one is **abandoned
rather than scored** — the questions it never reached would otherwise count as
answers it got wrong, which a scorecard cannot tell apart from a model that got
them wrong.

## Holding a test half back

The only lever on agent quality is the prompt, and a prompt tuned by reading
the questions it failed, then scored on those same questions, reports a number
fitted to its own answer key.

So 40% of the question set is held back, and which 40% is decided by a hash of
the question text rather than by any editable field. Whoever tunes the prompt
does not get to choose what they are marked on.

There is deliberately no salt to turn. The twelve toy questions all landed in
dev — a 1-in-450 coincidence that looks like something to fix, and adding a
salt until the split came out nicer would be choosing the test set by hand.
It is left alone and the report says so out loud instead.

## Status

**Harness complete, unmeasured.** The agent, the four verbs, the read-only
sandbox, the scorer, the eval and a 12-question toy set that runs on a clone
with no download and no key. 119 tests.

Next, in order:

| | |
|---|---|
| 1 | Settle the budget — the daily token cap, not the money, is the blocker |
| 2 | BIRD Mini-Dev (500 curated pairs, 11 databases) — the loader already reads BIRD's layout, so this is a path change |
| 3 | Measure three models, with the ablations that are the actual finding: schema-blind against schema-given, repair on against repair off |

The headline will be an **accuracy × cost curve**, not a single score. The
useful question is not which model is best; it is where the curve flattens.

## Licence

MIT.
