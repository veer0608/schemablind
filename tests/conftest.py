"""A small database with the shape of a real one.

BIRD's difficulty is not SQL syntax, it is that column names are abbreviated,
values are coded, and the join path is not obvious from the question. So the
fixture keeps those properties at a size that fits in a test: `cnty` rather
than `county`, a charter flag stored as 0/1, a grade stored as text, and one
table that only matters because it sits between two others.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

SCHEMA = """
CREATE TABLE schools (
    sch_id      INTEGER PRIMARY KEY,
    sch_name    TEXT NOT NULL,
    cnty        TEXT NOT NULL,
    charter     INTEGER NOT NULL DEFAULT 0,
    opened      TEXT
);

CREATE TABLE students (
    stu_id      INTEGER PRIMARY KEY,
    sch_id      INTEGER NOT NULL REFERENCES schools(sch_id),
    full_name   TEXT NOT NULL,
    grade       TEXT NOT NULL,
    enrolled_on TEXT
);

CREATE TABLE scores (
    score_id    INTEGER PRIMARY KEY,
    stu_id      INTEGER NOT NULL REFERENCES students(stu_id),
    subject     TEXT NOT NULL,
    score       INTEGER
);
"""

SCHOOLS = [
    (1, "Bayside High", "Alameda", 1, "2001-08-15"),
    (2, "Riverdale Academy", "Alameda", 0, "1994-09-01"),
    (3, "Hillcrest School", "Marin", 1, "2010-08-20"),
    (4, "Oakfield Charter", "Marin", 1, "2015-08-17"),
    (5, "Lakeview High", "Sonoma", 0, None),
    # Enrols nobody. Without a school like this, "which schools have no
    # students" has an empty answer, and every query returning nothing scores
    # correct against it -- a question that rewards failure.
    (6, "Pinecrest Prep", "Sonoma", 0, "2005-08-15"),
]

STUDENTS = [
    (1, 1, "Ada Bell", "11", "2023-08-20"),
    (2, 1, "Ravi Menon", "12", "2022-08-22"),
    (3, 2, "Sofia Cruz", "11", "2023-08-21"),
    (4, 2, "Tom Riley", "10", "2024-08-19"),
    (5, 3, "Yuki Tanaka", "12", "2022-08-23"),
    (6, 3, "Omar Haddad", "11", "2023-08-20"),
    (7, 4, "Lena Petrov", "12", "2022-08-22"),
    (8, 5, "Kwame Osei", "10", "2024-08-19"),
]

SCORES = [
    (1, 1, "math", 91), (2, 1, "reading", 88),
    (3, 2, "math", 74), (4, 2, "reading", 80),
    (5, 3, "math", 95), (6, 3, "reading", 92),
    (7, 4, "math", 61), (8, 4, "reading", 70),
    (9, 5, "math", 88), (10, 5, "reading", 84),
    (11, 6, "math", 79), (12, 6, "reading", 85),
    (13, 7, "math", 99), (14, 7, "reading", 97),
    (15, 8, "math", 55), (16, 8, "reading", None),
]


def build(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany("INSERT INTO schools VALUES (?,?,?,?,?)", SCHOOLS)
        connection.executemany("INSERT INTO students VALUES (?,?,?,?,?)", STUDENTS)
        connection.executemany("INSERT INTO scores VALUES (?,?,?,?)", SCORES)
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture(scope="session")
def school_db(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("db") / "school.sqlite")


@pytest.fixture
def writable_db(tmp_path) -> Path:
    """Its own copy, for the tests that try to write to it."""
    return build(tmp_path / "writable.sqlite")
