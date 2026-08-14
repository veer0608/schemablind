"""What the agent is allowed to do, and how little it is told to start with.

The agent gets no schema. It gets four verbs and a database it has never seen,
and has to find its own way to the answer. That is the point of the project:
generating SQL against a schema you were handed is a different, easier task
than working out which schema you are looking at, and only one of them
resembles being given a database and a question.

Everything here renders to text rather than JSON. Tokens are the binding
constraint on this whole project -- a schema rendered as prose costs a third of
the same schema as JSON, and the model reads it just as well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sandbox import Result, Sandbox

#: SQLite's own bookkeeping, which is never what the question is about.
_INTERNAL = re.compile(r"^sqlite_", re.I)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool
    not_null: bool
    references: str | None = None

    def rendered(self) -> str:
        parts = [self.name, self.type or "?"]
        if self.primary_key:
            parts.append("PK")
        if self.references:
            parts.append(f"-> {self.references}")
        elif self.not_null:
            parts.append("NOT NULL")
        return " ".join(parts)


class Tools:
    """The four verbs, bound to one database."""

    def __init__(self, sandbox: Sandbox, *, sample_limit: int = 3) -> None:
        self.sandbox = sandbox
        self.sample_limit = sample_limit
        self.calls: list[tuple[str, dict]] = []

    # -- the verbs ----------------------------------------------------------

    def list_tables(self) -> str:
        names = self._table_names()
        if not names:
            return "this database has no tables"
        return "tables: " + ", ".join(names)

    def describe_table(self, table: str) -> str:
        """Columns, types, keys and foreign keys, in one line per table."""
        known = self._table_names()
        if table not in known:
            return self._no_such_table(table, known)
        columns = self._columns(table)
        rendered = ", ".join(column.rendered() for column in columns)
        return f"{table}({rendered})"

    def sample_rows(self, table: str, limit: int | None = None) -> str:
        known = self._table_names()
        if table not in known:
            return self._no_such_table(table, known)
        limit = min(limit or self.sample_limit, 20)
        columns = [column.name for column in self._columns(table)]
        # The table name came from `known`, so it is one of ours, not the
        # model's -- but quote it anyway rather than rely on that holding.
        quoted = '"' + table.replace('"', '""') + '"'
        rows = self.sandbox.introspect(f"SELECT * FROM {quoted} LIMIT {limit}")
        if not rows:
            return f"{table} is empty"
        return _grid(columns, [tuple(row) for row in rows])

    def run_sql_readonly(self, sql: str) -> str:
        """The only verb that runs SQL the model composed."""
        result = self.sandbox.query(sql)
        return render_result(result)

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, arguments: dict) -> str:
        """Invoke by name, the way a tool-use block arrives. Never raises.

        A model that sends a misspelled tool or a missing argument should be
        told so and get another turn, not end the run.
        """
        self.calls.append((name, dict(arguments)))
        try:
            if name == "list_tables":
                return self.list_tables()
            if name == "describe_table":
                return self.describe_table(str(arguments["table"]))
            if name == "sample_rows":
                limit = arguments.get("limit")
                return self.sample_rows(
                    str(arguments["table"]), int(limit) if limit else None
                )
            if name == "run_sql_readonly":
                return self.run_sql_readonly(str(arguments["sql"]))
        except KeyError as exc:
            return f"{name} needs a {exc.args[0]!r} argument"
        except (TypeError, ValueError) as exc:
            return f"{name} could not read its arguments: {exc}"
        return f"no such tool: {name!r}. Available: {', '.join(t['name'] for t in SCHEMA)}"

    # -- internals ----------------------------------------------------------

    def _table_names(self) -> list[str]:
        rows = self.sandbox.introspect(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
        return [name for (name,) in rows if not _INTERNAL.match(name)]

    def _columns(self, table: str) -> list[Column]:
        # PRAGMA will not take a bound parameter, so the name is checked against
        # sqlite_master by the caller and quoted here.
        quoted = '"' + table.replace('"', '""') + '"'
        info = self.sandbox.introspect(f"PRAGMA table_info({quoted})")
        keys = {
            row[3]: f"{row[2]}.{row[4]}"
            for row in self.sandbox.introspect(f"PRAGMA foreign_key_list({quoted})")
        }
        return [
            Column(
                name=row[1],
                type=(row[2] or "").upper(),
                primary_key=bool(row[5]),
                not_null=bool(row[3]),
                references=keys.get(row[1]),
            )
            for row in info
        ]

    def _no_such_table(self, table: str, known: list[str]) -> str:
        near = [name for name in known if name.lower() == table.lower()]
        hint = f" Did you mean {near[0]!r}?" if near else ""
        return (
            f"no table called {table!r}.{hint} "
            f"This database has: {', '.join(known) or 'nothing'}"
        )


def render_result(result: Result, *, max_rows: int = 30) -> str:
    """A query result as the model will read it.

    An error comes back verbatim and unadorned, because it is the single most
    useful thing the agent ever receives: "no such column: county" is what
    turns a wrong guess into a right one on the next turn.
    """
    if not result.ok:
        return f"ERROR: {result.error}"
    if not result.rows:
        return "0 rows"
    shown = result.rows[:max_rows]
    text = _grid(list(result.columns), list(shown))
    notes = []
    if len(result.rows) > max_rows:
        notes.append(f"showing {max_rows} of {result.row_count}")
    if result.truncated:
        notes.append("more rows exist beyond the row cap")
    return text + (f"\n({'; '.join(notes)})" if notes else "")


def _grid(columns: list[str], rows: list[tuple]) -> str:
    """A pipe-separated grid. Cheaper in tokens than JSON and just as readable."""
    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
    return "\n".join(lines)


#: Tool definitions in the shape every provider takes.
SCHEMA: list[dict] = [
    {
        "name": "list_tables",
        "description": "List every table in the database. Start here; you have not been given the schema.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "describe_table",
        "description": (
            "Columns, types, primary keys and foreign keys for one table. "
            "Foreign keys are shown as '-> other_table.column' and are how you find join paths."
        ),
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Exact table name"}},
            "required": ["table"],
        },
    },
    {
        "name": "sample_rows",
        "description": (
            "A few real rows from one table. Use it when a column's meaning or "
            "encoding is unclear -- whether a flag is 0/1 or 'Y'/'N', how a date is written."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "limit": {"type": "integer", "description": "Rows to show, at most 20"},
            },
            "required": ["table"],
        },
    },
    {
        "name": "run_sql_readonly",
        "description": (
            "Run one read-only SELECT (or WITH) and see the rows. Use this to check "
            "your query before answering; if it errors, read the message and fix it."
        ),
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement"}},
            "required": ["sql"],
        },
    },
]

#: What the agent calls when it is ready to commit to an answer.
ANSWER_TOOL = {
    "name": "final_sql",
    "description": (
        "Submit the single SELECT that answers the question. Only call this once "
        "you have run it with run_sql_readonly and seen sensible rows."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The final SELECT statement"},
        },
        "required": ["sql"],
    },
}
