"""``python -m schemablind ask "question" path/to.sqlite``

Two commands. `ask` puts a question to the agent, and prints not just the SQL
it settled on but the path it took to get there -- which tables it looked at,
what it tried, what failed. An answer you cannot audit is the thing this
project is arguing against.

`schema` needs no model and no key. It prints exactly what the agent would
discover on its first two turns, which is the fastest way to see whether a
database is legible before spending anything asking questions of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent
from .sandbox import Sandbox
from .scoring import judge
from .tools import Tools


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="schemablind")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a question against a database")
    ask.add_argument("question")
    ask.add_argument("database", type=Path)
    ask.add_argument("--model", help="model to use; without it there is nothing to ask")
    ask.add_argument("--provider", help="which API (default: whichever has a key)")
    ask.add_argument("--evidence", default="", help="a hint, as BIRD supplies one")
    ask.add_argument("--gold", help="expected SQL; prints whether the answer matched")
    ask.add_argument("--max-turns", type=int, default=12)
    ask.add_argument(
        "--no-repair", action="store_true", help="make the first query final"
    )

    schema = sub.add_parser(
        "schema", help="print what the agent would discover -- no model needed"
    )
    schema.add_argument("database", type=Path)
    schema.add_argument("--samples", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "schema":
        return _schema(args.database, args.samples)
    return _ask(args)


def _schema(database: Path, samples: int) -> int:
    if not database.is_file():
        print(f"no database at {database}")
        return 2
    with Sandbox(database) as sandbox:
        tools = Tools(sandbox, sample_limit=samples)
        print(tools.list_tables())
        print()
        for name in tools.list_tables().removeprefix("tables: ").split(", "):
            print(tools.describe_table(name))
            print(_indent(tools.sample_rows(name)))
            print()
    return 0


def _ask(args) -> int:
    if not args.database.is_file():
        print(f"no database at {args.database}")
        return 2

    from .llm import build_client

    client = build_client(args.provider, args.model)
    if client is None:
        wanted = args.provider or args.model or "any provider"
        print(
            f"no API key configured for {wanted!r}.\n"
            f"  Put one in .env (it is gitignored), e.g. GROQ_API_KEY=...\n"
            f"  Or run `schemablind schema {args.database}` to look around for free."
        )
        return 2

    agent = Agent(client, max_turns=args.max_turns, repair=not args.no_repair)
    with Sandbox(args.database) as sandbox:
        transcript = agent.solve(args.question, sandbox, evidence=args.evidence)

        print(f"> {args.question}")
        print()
        if not transcript.sql:
            print(f"  no query produced -- {transcript.stopped}")
            if transcript.error:
                print(f"  {transcript.error}")
            return 1

        print("  " + transcript.sql.replace("\n", "\n  "))
        print()
        result = sandbox.query(transcript.sql)
        print(_indent(_render(result)))
        print()
        print(
            f"  {transcript.turns} turns, "
            f"{len(transcript.tool_calls)} tool calls, "
            f"{transcript.repairs} repair(s), "
            f"{transcript.total_tokens} tokens"
            + (
                f", ${transcript.cost_usd:.6f}"
                if transcript.cost_usd is not None
                else ", unpriced"
            )
        )
        print(f"  path: {' -> '.join(transcript.tools_used) or 'none'}")

        if args.gold:
            verdict = judge(transcript.sql, args.gold, sandbox)
            print()
            print(f"  {'CORRECT' if verdict.correct else 'WRONG'} -- {verdict.reason}")
            return 0 if verdict.correct else 1
    return 0


def _render(result) -> str:
    from .tools import render_result

    return render_result(result)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
