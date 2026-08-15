---
name: run-eval
description: Spend API budget on a schemablind eval without wasting it — check the daily token budget first, pick a model that can finish, pace the run, and never publish a partial one. Use before any run that calls a model, including smoke runs.
---

# Running an eval

Every trap below cost a wasted run at least once. None of them is theoretical.

## Before spending anything

1. **Check the daily budget, with a full-size prompt.** A one-token probe
   succeeds while the daily budget is gone, so it proves nothing:

   ```bash
   python -m evals.runner --bird --limit 1 --solvers groq:openai/gpt-oss-20b
   ```

   A refusal names the real limit: `(TPD): Limit 200000, Used 199088`.

2. **Do the arithmetic before the run, not after.** A real BIRD question costs
   **~13,600 tokens over ~9 turns**. Daily caps are 100–200k per model, so a
   free tier affords **~15 questions per model per day**. If the run needs more
   than the budget allows, it will die partway and produce nothing — shrink it
   or pick another model.

3. **Free tiers do not reset on a calendar day.** A model at 199k/200k
   yesterday can still be at 199k today. Never plan around "tomorrow"; check.

4. **`--solvers oracle,mute` is free** and needs no key. Run it first if the
   harness has changed at all.

## Choosing what to run

- Models each have their own daily budget. When one is spent, another is
  usually not — check all of them rather than giving up.
- `--limit N` for a smoke run, always, before a full pass.
- `--split dev` while iterating. Never `--split test` unless reporting a
  finished result.

## During

- The per-minute limit binds harder than the per-day one on big models:
  `gpt-oss-120b` allows 8,000 tokens/minute against a ~13,600-token question,
  so throughput is ~0.9 questions/minute regardless of model speed. A 20
  question run takes ~20 minutes. This is normal, not a hang.
- Progress prints every 5 questions with a projected finish. Read that rather
  than guessing; CPU time near zero means it is pacing, not stuck.

## After

- **A run that hit the daily cap is abandoned and gets no row.** That is
  correct — the questions it never reached would count as answers it got wrong.
  Do not hand-compute a percentage from the partial results and present it as a
  score. Describing "N of the M that completed" is fine if labelled.
- **Republish from saved JSON, never by re-running.** One pass is most of a
  day's budget:

  ```bash
  python -m evals.runner --from-json <runs...> --update-readme
  ```

- Record every run's JSON. Per-question tokens are in there, so a price
  correction never costs another run.
