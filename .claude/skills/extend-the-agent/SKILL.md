---
name: extend-the-agent
description: Change schemablind's sandbox, tools, agent loop or scorer without breaking the guarantees everything else rests on. Use when adding a tool, editing the system prompt, changing the loop, or touching read-only enforcement.
---

# Changing the agent

## The invariants

Break any of these and every number the project reports becomes worthless.

1. **The agent cannot write.** Four layers, each with a hole on its own: the
   file opened `mode=ro`, `PRAGMA query_only`, only `SELECT`/`WITH` accepted,
   one statement per call. Adding a tool that reaches the database must go
   through `Sandbox.query`, never around it.

2. **Two paths that never mix.** `Sandbox.introspect` runs fixed statements
   this project wrote and permits `PRAGMA`; `Sandbox.query` runs anything a
   model composed and does not. **Never pass model output to `introspect`** —
   that is the one call that would turn a read-only sandbox into an arbitrary
   one.

3. **The agent is told no schema.** If a table or column name appears in the
   system prompt or a tool description, the experiment is no longer
   schema-blind. `test_no_schema_is_advertised_anywhere_in_it` guards this.

4. **Nothing in the test suite may call a model.** Use `ScriptedClient`. An
   eval that costs money is a choice someone makes on purpose.

## After any change

```bash
python -m pytest -q
python -m evals.runner --check      # free: oracle must be 100%, mute 0%
```

The `--check` is the one that matters. If the gold queries stop scoring
themselves 100%, the scorer is broken and no model number from it is readable.
CI runs it, plus an assertion that a database is byte-identical after trying to
delete, drop, update, attach and smuggle a write past a semicolon.

## Editing the system prompt

- Prompt changes are scored on the **dev half**, and reported against **test**.
  A rule that helps dev and not test fitted the questions, not the problem.
- Prefer fixing where the mistake happens over adding to the prompt. The agent
  ignored "call final_sql" in the system prompt on every question; a reminder
  appended to successful `run_sql_readonly` output, plus `tool_choice` forcing
  the call, worked where words alone did not.
- `tool_choice` naming a function makes the API **require** that call. Force it
  only on the turn that needs it — forcing every turn stops the exploration
  that is the thing being measured.

## Adding a tool

1. Implement it on `Tools`, returning **text**, not JSON. Tokens are the
   binding constraint; prose costs about a third of the same content as JSON.
2. Add it to `SCHEMA` with a description that names no table or column.
3. Handle a missing or malformed argument by returning a message, never by
   raising — a model that sends a bad argument should get another turn.
4. Test it against the toy database in `tests/test_tools.py`.
