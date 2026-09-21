# What the first 162 dev answers show

Read on 2026-09-21, while the dev-split run was paused on its daily quota. Every
row here is from the **dev half** — the split filter runs in the same expression
that loads the checkpoint, so nothing held out has been read, and
`evals/quarantine.json` is unchanged.

**These are not a score.** 162 of 268 questions had been answered when the
allowance ran out. The subset is 68.5% correct (111 of 162), and that number is
a diagnosis aid, not the split's execution accuracy: the 106 unanswered
questions are not a random sample of what is left, they are whatever came next
in order. The scorecard stays empty until the split completes.

## The harness is not the problem

Checked first, in the order `diagnose-failures` gives, because reading a client
bug as a finding about the model has already happened here once:

- `stopped`: 157 answered, 5 ran out of turns. No `the model call failed`.
- `error`: none, on any row.
- final SQL present on all 162, so nothing was produced and then lost.
- the 5 that ran out of turns still answered `via its last verified query`.

## Where the 51 failures are

| category | count |
|---|---|
| right shape, wrong values | 33 |
| wrong number of columns | 13 |
| wrong number of rows | 5 |

By difficulty: 27 moderate, 15 challenging, 9 simple, against a mix of 77
moderate, 53 simple and 32 challenging. Simple questions are mostly fine; the
moderate band is where the losses are.

## The column failures are mostly a shape convention, not a mistake

Reading all 13 rather than the summary, because the summary pointed the wrong
way. A count of select-list entries said "gold had more columns" 9 times and
read as the agent under-selecting, which is true in letter and misleading in
substance:

- **The agent answers the question; the gold returns the parts.** Gold asks for
  `forename, surname, url`, the agent returns `url`. Gold asks for
  `id, finishing, curve`, the agent returns `finishing, curve` — the `id` is in
  the gold and in no part of the question.
- **Twice the agent concatenated**: `forename || ' ' || surname AS driver_name`
  answers the question and loses the two-column shape the scorer compares.
- **Only twice did it dump the whole row**: `SELECT p.ID, p.SEX, p.Birthday, …`
  where the gold is `SELECT T1.ID`, on "list all patients who…".

So of 13, perhaps 4 are the agent being wrong about what to return, and the rest
are BIRD's convention that a result carries the identifying columns whether or
not the question names them. That is a prompt-shaped gap, not a reasoning one.

## The failures are not spread evenly across databases

| database | asked | failed | fail rate |
|---|---|---|---|
| thrombosis_prediction | 23 | 14 | 61% |
| debit_card_specializing | 22 | 9 | 41% |
| formula_1 | 37 | 13 | 35% |
| european_football_2 | 35 | 8 | 23% |
| student_club | 18 | 3 | 17% |
| superhero | 24 | 3 | 12% |

Two databases carry 23 of the 51 failures. That is worth more than the overall
rate, because it says the gap is not uniform incompetence at SQL.

Reading the `debit_card_specializing` ones, one pattern is a **denominator
dispute on an underspecified question**. "What is the percentage of the
customers who used EUR on 2012/8/25" - the gold counts rows of transactions,
the agent counts `DISTINCT CustomerID`. Both read the English correctly; only
one matches the gold.

**Correction to an earlier draft of this file.** It said these are not failures
the agent could have avoided, and that a schema-blind agent has no way to learn
the convention. Both claims were too generous, and checking them took running
every failing pair rather than reading three of them:

- **The agent is given BIRD's `evidence` field as a hint** (`agent.solve(...,
  evidence=question.evidence)`), and all 14 thrombosis failures carry one. It is
  not working blind on the convention; it is not following the hint.
- On Q1150 the agent's arithmetic is *identical* to the gold and it simply
  omits the `* 100` that the word percentage implies. Q881 is the same. So two
  of the 33 are a scaling slip, which is the agent being wrong, not the
  question being ambiguous.

Running the gold and the agent query for all 33 value failures and comparing
what came back: 18 are genuinely different numbers, 7 are multi-row or
multi-column, 4 return something not numeric, 2 are the missing `* 100`, and 2
come back equal within rounding.

`thrombosis_prediction` is the one to look at properly once the split is done:
61% is far off the rest, its columns are medical abbreviations, and its
questions lean on conventions ("normal platelet level") that live in the
evidence field rather than the schema.

## One failure is the scorer, and it is staying

Of the two that come back equal within rounding, one is real: Q1473 returns
`459.95626428710585` where the gold is `459.9562642871061`. Thirteen significant
figures agree; they differ in the last bits of a float. `_same_set` compares
`set(predicted.rows) == set(gold.rows)`, so that scores as wrong. (The other,
Q1037, is genuinely different: 24.561 against 24.567.)

**That exact-equality comparison is BIRD's own, and it should stay.** The whole
argument for this metric is that the number means something to someone who has
never seen this repo, and a tolerant comparison would quietly stop being BIRD's
number. The honest handling is to know the cost: one question in 162, about 0.6
points, charged against the agent for a float representation.

## What is deliberately not being done yet

**The agent is not being changed until the split finishes.** A prompt rule about
result shape would make the remaining 106 answers incomparable with these 162,
and the run would have to start again from zero rather than from a checkpoint.
The fix is worth trying; it is worth trying as its own measured change, against
a completed baseline.
