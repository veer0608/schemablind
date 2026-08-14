# schemablind

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

Run the eval:

```bash
python -m evals.runner --solvers oracle,mute
```

## The scorecard

<!-- SCORECARD -->

| configuration | execution accuracy | strict | produced SQL | turns | $/question | p50 |
|---|---|---|---|---|---|---|
| oracle (gold SQL) | 100.0% | 100.0% | 100.0% | 0.0 | $0 | 0 ms |
| mute (no SQL) | 0.0% | 0.0% | 0.0% | 0.0 | $0 | 0 ms |

<!-- /SCORECARD -->

**No model numbers yet, deliberately.** Those two rows are not competitors —
they are the harness proving itself. The oracle submits the reference query and
must score 100%; the mute agent submits nothing and must score 0%. If either
ever drifts, the scorer is broken and no model number produced by it is worth
reading. CI runs exactly that check, and it needs no key.

The model rows arrive when the budget question below is settled.

## What this will cost, before it is spent

Worth stating up front, because it is the thing that decides whether the
project is finishable rather than a detail discovered halfway through.

A schema-blind agent is not a single prompt. Five turns with a growing context
runs about **20k input and 1k output tokens per question**.

| | free tier | paid |
|---|---|---|
| per question | ~20k tokens | ~$0.0018 |
| daily cap | 100k tokens per model | none |
| **questions/day/model** | **~5** | unlimited |
| 150 questions × 3 models | **~90 days** | **~$1** |

The free tier's daily token cap is what blocks this, not the money — the whole
eval costs about a dollar. And that daily limit appears in **no response
header**: the per-minute bucket reads a healthy 12,000 while the daily budget
is already gone, and the real limit is only named in the body of the 429 that
eventually refuses you. So a daily refusal is its own error type here, and a
run that hits one is **abandoned rather than scored** — the questions it never
reached would otherwise count as answers it got wrong, which a scorecard cannot
tell apart from a model that got them wrong.

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
