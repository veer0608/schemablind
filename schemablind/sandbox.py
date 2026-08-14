"""Running SQL a model wrote, without trusting it.

Everything downstream depends on this being genuinely read-only. The agent is
handed a database it has never seen and asked to explore it, which means the
statements executed here are written by a model, from a prompt that itself
contains data from the database. Treating that as untrusted input is not
paranoia; it is the only assumption that holds.

Four layers, because any one of them alone has a hole:

  the file is opened `mode=ro`      -- the driver refuses writes outright
  `PRAGMA query_only`               -- refuses them again inside the connection
  only SELECT and WITH are accepted -- ATTACH cannot reach a second file
  one statement per call            -- nothing hides behind a semicolon

On top of that a wall-clock limit and a row cap, which are not security so much
as survival: a cross join over two large tables is a perfectly valid SELECT and
will sit there until the process dies.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

#: A statement may only begin this way. Anything else -- INSERT, UPDATE, DROP,
#: ATTACH, PRAGMA, VACUUM -- is refused before SQLite ever sees it.
_READS = re.compile(r"^\s*(?:select|with)\b", re.I | re.S)

#: Stripped before that check, or a comment would hide the real first word.
_COMMENTS = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.S)

DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_ROWS = 1000
#: How often the progress handler runs, in VM instructions. Small enough that a
#: runaway query notices the clock, large enough not to dominate a fast one.
_TICK = 2000


@dataclass(frozen=True)
class Result:
    """What a query did. `error` is the whole point: the agent repairs from it."""

    columns: tuple[str, ...] = ()
    rows: tuple[tuple, ...] = ()
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return len(self.rows)


class Refused(Exception):
    """The statement was rejected before execution."""


def statement_is_read_only(sql: str) -> str | None:
    """Why this statement may not run, or None if it may.

    Returns the reason rather than a bool so the agent is told what it did
    wrong -- a model that gets "not allowed" learns nothing, and a model that
    gets "only SELECT and WITH are allowed, this began with INSERT" fixes it on
    the next turn.
    """
    bare = _COMMENTS.sub(" ", sql).strip()
    if not bare:
        return "empty statement"

    # sqlite3.complete_statement wants the trailing semicolon to decide.
    trimmed = bare.rstrip().rstrip(";").rstrip()
    if ";" in trimmed:
        return (
            "one statement per call -- found a ';' with more after it, and a "
            "read-only check that stopped at the first statement would be no "
            "check at all"
        )
    if not _READS.match(trimmed):
        first = (trimmed.split() or ["nothing"])[0].upper()
        return f"only SELECT and WITH may run here; this began with {first}"
    return None


class Sandbox:
    """A read-only connection to one SQLite file, with limits."""

    def __init__(
        self,
        path: str | Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self.path = Path(path)
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        # A URI with mode=ro is the layer that actually holds: the driver will
        # not open a write journal, so a write fails even if every check above
        # it were bypassed.
        uri = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=self.timeout_s)
        connection.execute("PRAGMA query_only = ON")
        connection.text_factory = lambda b: b.decode("utf-8", errors="replace")
        self._connection = connection
        return connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "Sandbox":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def introspect(self, sql: str, params: tuple = ()) -> list[tuple]:
        """The trusted path, for SQL this project wrote itself.

        Schema discovery needs `PRAGMA table_info` and `sqlite_master`, which
        `query` refuses on purpose -- it cannot tell a PRAGMA the tools issued
        from one a model talked its way into. So the two paths stay separate:
        fixed statements with bound parameters here, anything a model composed
        over there. Never pass model output to this.
        """
        cursor = self.connect().execute(sql, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()

    def query(self, sql: str) -> Result:
        """Run one read-only statement. Never raises for bad SQL -- that is a Result.

        A syntax error, a missing column, a timeout: all of these are things the
        agent is expected to recover from, so they come back as text it can read
        rather than as an exception that ends the run.
        """
        started = time.perf_counter()
        refusal = statement_is_read_only(sql)
        if refusal is not None:
            return Result(error=refusal, elapsed_ms=_since(started))

        try:
            connection = self.connect()
        except FileNotFoundError as exc:
            return Result(error=f"no such database: {exc}", elapsed_ms=_since(started))

        deadline = started + self.timeout_s
        connection.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, _TICK)
        try:
            cursor = connection.execute(sql)
            columns = tuple(d[0] for d in cursor.description or ())
            # One more than the cap, so truncation is detected rather than
            # guessed at from a row count that happens to equal the limit.
            fetched = cursor.fetchmany(self.max_rows + 1)
            truncated = len(fetched) > self.max_rows
            rows = tuple(tuple(row) for row in fetched[: self.max_rows])
            cursor.close()
            return Result(
                columns=columns,
                rows=rows,
                truncated=truncated,
                elapsed_ms=_since(started),
            )
        except sqlite3.OperationalError as exc:
            # The progress handler aborts by returning non-zero, which surfaces
            # here as "interrupted" and is indistinguishable from a real one.
            message = str(exc)
            if "interrupted" in message.lower() and time.perf_counter() > deadline:
                message = f"query exceeded the {self.timeout_s:g}s limit and was stopped"
            return Result(error=message, elapsed_ms=_since(started))
        except sqlite3.Error as exc:
            return Result(error=f"{type(exc).__name__}: {exc}", elapsed_ms=_since(started))
        finally:
            connection.set_progress_handler(None, _TICK)


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000
