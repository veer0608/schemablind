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

## What is deliberately not being done yet

**The agent is not being changed until the split finishes.** A prompt rule about
result shape would make the remaining 106 answers incomparable with these 162,
and the run would have to start again from zero rather than from a checkpoint.
The fix is worth trying; it is worth trying as its own measured change, against
a completed baseline.
