---
name: diagnose-failures
description: Read schemablind eval failures without destroying the held-out set — filter to the dev half before looking at anything, distinguish harness bugs from agent mistakes, and quarantine any test question whose answer gets seen. Use when investigating why an eval scored badly.
---

# Diagnosing a bad score

## Filter to dev BEFORE looking at anything

This is the whole discipline and it has already been broken once. A question
stops being held out the moment its gold SQL is read, whatever the split says.

```python
dev = [r for r in results if r["split"] == "dev"]
```

Do that first, in the same expression that loads the results. Do not print a
failure list and filter afterwards — by then it has been read.

**If a test question is seen anyway**, add it to `evals/quarantine.json` with
the date and reason. The runner drops quarantined questions before scoring. The
list only grows. Silently leaving a seen question in the test half reports a
number partly measured on known answers, and that the leak was accidental
changes nothing about what it does to the number.

## Rule out the harness before blaming the agent

The most expensive mistake available here is reading a bug in the client as a
finding about the model. It has happened: six questions were reported as "the
agent produced no query", written up as a finding that committing to an answer
is a schema-blind agent's weakness — and all six were API errors where the
agent had run a good query that the code then discarded.

Check, in this order:

1. `stopped` — `the model call failed` is the client, not the agent.
2. `error` — the actual API message.
3. `last_query` — if a query is present and the score says none was produced,
   the harness lost it.
4. `answered_via` — `its last verified query` means the agent never submitted
   and the fallback rescued it. A high rate here is a protocol problem, not a
   SQL one.
5. `nudges` — how often it had to be told to submit at all.

## Then read the failure taxonomy

`failure` separates problems with different fixes:

| failure | what it usually means |
|---|---|
| `did not run` | invented a column or table; check whether `describe_table` was called |
| `wrong number of columns` | almost always selecting the column it ranked by |
| `wrong number of rows` | missing or wrong `WHERE`, or a missing `DISTINCT` |
| `right shape, wrong values` | an encoding or rounding trap; check whether `sample_rows` was called |
| `no query produced` | rule out the harness first, per above |
| `gold query failed` | a broken dataset row, never the agent's fault |

## Fixing

- Prompt rules must be **general**, not patches for a question. "Do not select
  the column you ranked by" is a rule; "for question 1471, use CustomerID" is
  memorising the answer key.
- Re-scoring offline is free — the predicted queries and the databases are both
  on disk. Use it to estimate a fix's effect before spending budget on it.
- Report dev and test separately. If dev improves and test does not, the change
  fitted the questions rather than the problem.
