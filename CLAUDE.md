# schemablind

A SQL agent given no schema, scored on BIRD execution accuracy. See README.md for
what it is and why. This file is the operational half: how to run it without
wasting a day.

## Layout

- `schemablind/` — the package: `sandbox.py`, `tools.py`, `agent.py`, `scoring.py`,
  `llm.py`, `cli.py`
- `evals/` — `runner.py` (the eval), `dataset.py`, `checkpoint.py`, `quarantine.json`,
  and `toy/` (a 12-question set that runs on a fresh clone, no download, no key)
- `data/` — gitignored. BIRD Mini-Dev is already unpacked at `data/minidev`.
- `runs/`, `evals/runs/` — saved run JSON. Keep every one; per-question tokens live
  there, so a price correction never costs another run.

## Commands

Run from the repo root. Python 3.11.

```bash
python -m pytest -q                                  # 140 tests, no key needed
python -m evals.runner --solvers oracle,mute         # free harness self-check
python -m evals.runner --check                       # what CI runs
python -m schemablind schema evals/toy/databases/school/school.sqlite
```

## Before any run that calls a model

**Read `.claude/skills/run-eval` first.** It exists because each of its rules cost a
wasted run. The short version:

- Probe the budget with a **full-size** prompt (`--bird --limit 1`). A one-token probe
  succeeds while the daily allowance is gone and proves nothing.
- Budgets are **per model per day**, ~200k tokens. A real question costs ~12,400
  tokens over ~7 turns, so a free tier affords **~16 questions/day/model**.
- Free tiers do not reset on a calendar day. Check, never assume.
- Always pass `--checkpoint PATH`. A capped run then resumes instead of restarting.
- `--split dev` while iterating. `--split test` only when reporting a finished result.

## Rules that are not style preferences

- **A run that hits the cap is abandoned and gets no row.** Questions it never asked
  would otherwise be indistinguishable from questions it got wrong. Never
  hand-compute a percentage from partial results and present it as a score;
  "N of the M that completed" is fine if labelled that way.
- **Never publish a partial run.** Republish from saved JSON with
  `--from-json <runs...> --update-readme`, never by re-running.
- **The oracle must score 100% and the mute solver 0%.** If either drifts the scorer
  is broken and no model number from it is worth reading. CI asserts this.
- **The sandbox has two paths that never mix.** `introspect()` runs fixed statements
  this project wrote; `query()` runs anything a model composed and permits only
  SELECT/WITH. Never pass model output to `introspect`.
- **Do not read held-out failures.** Filter to the dev half before looking at
  anything; a test question whose answer you see must be added to
  `evals/quarantine.json`. See `.claude/skills/diagnose-failures`.

## Environment

- The key lives in a gitignored `.env` (`GROQ_API_KEY`, `GEMINI_API_KEY`), parsed by
  `llm.py`'s own `load_dotenv` — it handles UTF-16 written by PowerShell.

Machine-wide constraints (PowerShell, no Docker, Groq's invisible daily cap) live in
`~/.claude/CLAUDE.md` and are not repeated here.
