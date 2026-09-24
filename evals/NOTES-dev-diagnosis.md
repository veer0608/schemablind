# What the dev half shows

The dev split finished on 2026-09-23, five daily windows after it started:
**64.7% execution accuracy over 266 questions**, 172 right and 94 wrong.

This replaces an earlier read of the first 162 answers, which put the same
agent at 68.5%. That subset was not a random sample, it was whatever came
first, and it was **3.8 points optimistic**. Where a claim here contradicts the
partial, this file is the one that ran every failing pair.

Everything below is the dev half only, filtered in the expression that loads
the results. `evals/quarantine.json` is unchanged.

## The harness is not the problem

Checked first, in the order `diagnose-failures` gives, because reading a client
bug as a finding about the model has already happened here once:

- `error`: none, on any of the 266.
- `stopped`: 254 answered, 12 ran out of turns.
- `produced SQL`: 100%. Nothing was generated and then lost.
- `nudges`: **zero**. The protocol failure that dominated the first live run -
  an agent that explores correctly and then answers in prose - is gone.
- the 12 that ran out of turns still answered `via its last verified query`.

So all 94 are the agent's.

## Where the failures are

| category | count |
|---|---|
| right shape, wrong values | 54 |
| wrong number of columns | 20 |
| wrong number of rows | 20 |

Difficulty behaves exactly as it should, which is itself a check on the
dataset's own labels:

| difficulty | failed / asked | |
|---|---|---|
| simple | 12 / 71 | 17% |
| moderate | 56 / 145 | 39% |
| challenging | 26 / 50 | 52% |

By database, the spread is wide enough that "the agent is bad at SQL" does not
describe it:

| database | failed / asked | |
|---|---|---|
| thrombosis_prediction | 14 / 23 | 61% |
| financial | 9 / 16 | 56% |
| california_schools | 10 / 20 | 50% |
| debit_card_specializing | 9 / 22 | 41% |
| codebase_community | 9 / 24 | 38% |
| card_games | 11 / 30 | 37% |
| formula_1 | 13 / 37 | 35% |
| toxicology | 5 / 17 | 29% |
| european_football_2 | 8 / 35 | 23% |
| student_club | 3 / 18 | 17% |
| superhero | 3 / 24 | 12% |

The partial read said two databases carried nearly half the failures. Over the
full half that is no longer true - `financial`, `california_schools`,
`card_games` and `codebase_community` were mostly in the unanswered tail, and
they fail at 37-56%. `thrombosis_prediction` is still the worst at 61%, and
still the one whose questions lean on conventions ("normal platelet level")
that live in the evidence field rather than the schema.

## Running all 94 pairs

Both queries executed against the real database, and the results compared:

| what came back | count |
|---|---|
| a different number | 30 |
| a different number of rows | 27 |
| same shape, different content | 21 |
| a different number of columns | 13 |
| off by exactly a factor of 100 | 2 |
| equal within rounding | 1 |

The two scaling slips are the ones the partial found, and they survive the full
run: `thrombosis_prediction` 1150 (94.037 against 0.94037) and `formula_1` 881
(17.241 against 0.17241). The agent's arithmetic is identical to the gold in
both; it omits the `* 100` that the word percentage implies. That is the agent
being wrong, not the question being ambiguous.

## The column failures, read one at a time

All 20, because the summary statistic pointed the wrong way last time. Gold is
wider in 14 and the agent is wider in 6.

**The dominant genuine error is dropping a quantity the question named.** Not a
convention, not a shape dispute - the question asks for it in words:

- "State the driver with the most points scored. Find his full name **with that
  points**" - the agent returns the name and not the points.
- "The oldest SJS patient's work was completed on what date, **and what age**" -
  date only.
- "Rank heroes by their height" - the name, without the height it ranked by.
- "What are the valid e-mail **addresses**" - one of the two.
- "the top nine districts ... **the number of** female clients" - the district
  without the number.

Against that, a smaller group really is BIRD's convention rather than a mistake:
gold carrying an identifying column the question never names (`id`, alongside
`finishing, curve`), and twice the agent concatenating `forename || ' ' ||
surname` into one column, which answers the question and loses the shape the
scorer compares. And the agent over-answers in 6, twice dumping the whole
patient row where the gold selects `ID` on "list all patients who...".

So the partial's reading - that most of these are convention - does not hold at
full size. Most of them are the agent under-answering a question that said what
it wanted.

## The row failures have their own convention trap

Four of the 20 are yes/no questions where the gold returns every matching row
rather than an answer: "Did Maya Mclean attend the Women's Soccer event?" has a
14-row gold; "Was the patient's uric acid within a normal range?" has 67. The
agent answers the question asked, in one row, and is marked wrong.

The rest are ordinary: a missing `DISTINCT` returning 9,103 rows against 94, a
filter applied to the wrong table, a `LIMIT` the question did not ask for.

## One failure is the scorer, and it is staying

`debit_card_specializing` 1473 returns `459.95626428710585` where the gold is
`459.9562642871061`. Thirteen significant figures agree; they differ in the
last bits of a float, and `_same_set` compares tuples exactly, so it scores as
wrong.

**That comparison is BIRD's own and it should stay.** The argument for this
metric is that the number means something to someone who has never seen this
repo, and a tolerant comparison would quietly stop being BIRD's number. The
honest handling is to write the cost down: one question in 266, about 0.4
points, charged against the agent for a float representation.

## What to try next, and how

Each of these is a separate measured change against this 64.7% baseline, on the
dev half, never folded in together:

1. **A rule that "percentage" means `* 100`.** Worth 2 questions outright, and
   the cheapest thing here.
2. **A rule to return every quantity the question names**, not only the entity
   it identifies. This is the largest addressable group: on the order of 10 of
   the 20 column failures.
3. **Follow the formula the evidence hint states.** All 14
   `thrombosis_prediction` failures carry a hint; the agent is given it
   (`agent.solve(..., evidence=question.evidence)` at `evals/runner.py:109`)
   and is not using it as a specification.

The held-out half stays untouched until the agent stops changing. 232 questions
at about 6 requests each is three daily windows on one key.
