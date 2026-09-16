"""Safe structured query utilities for relational application data.

This module complements ``search.py`` rather than replacing it. ``search.py``
handles lexical/BM25/fuzzy retrieval over an inverted index. ``queries.py``
handles deterministic structured SELECT queries over SQLite using validated
identifiers and bound parameters.

Intentionally excluded:
* raw SQL execution
* INSERT/UPDATE/DELETE/DDL
* agent-specific semantic/inference queries

Those boundaries keep the reusable application layer predictable and prevent
this module from becoming a second database ORM or an agent reasoning system.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time

from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.functions_error import *
from logs.logger import get_logger # type: ignore

logger = get_logger("Queries")

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_LIMIT = 100
_DEFAULT_MAX_LIMIT = 1000


class QueryOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


@dataclass(frozen=True)
class QueryFilter:
    field: str
    operator: QueryOperator = QueryOperator.EQ
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _validate_identifier(self.field, "filter field"))

        if not isinstance(self.operator, QueryOperator):
            try:
                object.__setattr__(
                    self,
                    "operator",
                    QueryOperator(str(self.operator)),
                )
            except ValueError as exc:
                raise QueryValidationError(
                    f"unsupported query operator: {self.operator!r}"
                ) from exc

        if self.operator in {QueryOperator.IS_NULL, QueryOperator.NOT_NULL}:
            if self.value is not None:
                raise QueryValidationError(
                    f"operator '{self.operator.value}' does not accept a value"
                )

        if self.operator in {QueryOperator.IN, QueryOperator.NOT_IN}:
            if isinstance(self.value, (str, bytes, bytearray)):
                raise QueryValidationError(
                    f"operator '{self.operator.value}' requires a non-string sequence"
                )
            if not isinstance(self.value, Sequence):
                raise QueryValidationError(
                    f"operator '{self.operator.value}' requires a sequence"
                )


@dataclass(frozen=True)
class QuerySort:
    field: str
    direction: SortDirection = SortDirection.ASC

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _validate_identifier(self.field, "sort field"))
        if not isinstance(self.direction, SortDirection):
            try:
                object.__setattr__(
                    self,
                    "direction",
                    SortDirection(str(self.direction).lower()),
                )
            except ValueError as exc:
                raise QueryValidationError(
                    f"unsupported sort direction: {self.direction!r}"
                ) from exc


@dataclass(frozen=True)
class QuerySpec:
    """Declarative SELECT query.

    ``columns=()`` means all columns. Table and column names are validated and
    checked against the SQLite schema or an explicit allow-list before SQL is
    generated.
    """

    table: str
    columns: Tuple[str, ...] = ()
    filters: Tuple[QueryFilter, ...] = ()
    sort: Tuple[QuerySort, ...] = ()
    limit: int = _DEFAULT_LIMIT
    offset: int = 0
    distinct: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "table", _validate_identifier(self.table, "table"))

        columns = tuple(
            _validate_identifier(column, "column") for column in self.columns
        )
        object.__setattr__(self, "columns", columns)

        filters = tuple(
            item if isinstance(item, QueryFilter) else QueryFilter(**item)
            for item in self.filters
        )
        sort = tuple(
            item if isinstance(item, QuerySort) else QuerySort(**item)
            for item in self.sort
        )
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "sort", sort)

        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise QueryValidationError("limit must be an integer")
        if self.limit <= 0:
            raise QueryValidationError("limit must be greater than zero")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise QueryValidationError("offset must be an integer")
        if self.offset < 0:
            raise QueryValidationError("offset must be zero or greater")


@dataclass(frozen=True)
class QueryResult:
    columns: Tuple[str, ...]
    rows: Tuple[Dict[str, Any], ...]
    row_count: int
    elapsed_seconds: float
    fingerprint: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "row_count": self.row_count,
            "elapsed_seconds": self.elapsed_seconds,
            "fingerprint": self.fingerprint,
        }


class SQLiteQueryService:
    """Read-only structured query service for SQLite application data."""

    def __init__(
        self,
        database: Union[str, Path],
        *,
        read_only: bool = True,
        timeout_seconds: float = 5.0,
        max_limit: int = _DEFAULT_MAX_LIMIT,
        allowed_schema: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> None:
        if not database:
            raise QueryValidationError("database path must not be empty")
        if timeout_seconds <= 0:
            raise QueryValidationError("timeout_seconds must be greater than zero")
        if isinstance(max_limit, bool) or int(max_limit) <= 0:
            raise QueryValidationError("max_limit must be a positive integer")

        self.database = Path(database).expanduser()
        self.read_only = bool(read_only)
        self.timeout_seconds = float(timeout_seconds)
        self.max_limit = int(max_limit)
        self.allowed_schema = _normalize_allowed_schema(allowed_schema)

        if self.read_only and not self.database.exists():
            raise QueryValidationError(
                f"read-only database does not exist: {self.database}"
            )

    def select(self, spec: QuerySpec) -> QueryResult:
        """Execute a validated SELECT query and return mapping rows."""

        if not isinstance(spec, QuerySpec):
            raise TypeError("spec must be a QuerySpec")
        if spec.limit > self.max_limit:
            raise QueryValidationError(
                f"limit {spec.limit} exceeds configured maximum {self.max_limit}"
            )

        started = time.perf_counter()
        with closing(self._connect()) as conn:
            available_columns = self._resolve_columns(conn, spec.table)
            sql, params = self._build_select(spec, available_columns)
            fingerprint = _query_fingerprint(sql, params)

            try:
                cursor = conn.execute(sql, params)
                fetched = cursor.fetchall()
            except sqlite3.Error as exc:
                logger.error(
                    "Structured query failed fingerprint=%s error=%s",
                    fingerprint,
                    exc,
                )
                raise QueryExecutionError(
                    str(exc),
                    fingerprint=fingerprint,
                ) from exc

        rows = tuple(dict(row) for row in fetched)
        columns = (
            tuple(rows[0].keys())
            if rows
            else self._selected_columns(spec, available_columns)
        )
        elapsed = time.perf_counter() - started

        logger.info(
            "Structured query completed table=%s rows=%s fingerprint=%s",
            spec.table,
            len(rows),
            fingerprint,
        )
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            elapsed_seconds=elapsed,
            fingerprint=fingerprint,
        )

    def count(
        self,
        table: str,
        *,
        filters: Sequence[QueryFilter] = (),
    ) -> int:
        """Count rows matching validated filters."""

        table = _validate_identifier(table, "table")
        normalized_filters = tuple(
            item if isinstance(item, QueryFilter) else QueryFilter(**item)
            for item in filters
        )

        with closing(self._connect()) as conn:
            available_columns = self._resolve_columns(conn, table)
            self._validate_referenced_columns(
                available_columns,
                filters=normalized_filters,
                sort=(),
                selected=(),
            )
            where_sql, params = self._build_where(normalized_filters)
            sql = f'SELECT COUNT(*) AS "count" FROM {_quote_identifier(table)}'
            if where_sql:
                sql += f" WHERE {where_sql}"

            fingerprint = _query_fingerprint(sql, params)
            try:
                row = conn.execute(sql, params).fetchone()
            except sqlite3.Error as exc:
                raise QueryExecutionError(
                    str(exc),
                    fingerprint=fingerprint,
                ) from exc

        return int(row["count"]) if row is not None else 0

    def exists(
        self,
        table: str,
        *,
        filters: Sequence[QueryFilter] = (),
    ) -> bool:
        """Return whether at least one row matches the filters."""

        table = _validate_identifier(table, "table")
        normalized_filters = tuple(
            item if isinstance(item, QueryFilter) else QueryFilter(**item)
            for item in filters
        )

        with closing(self._connect()) as conn:
            available_columns = self._resolve_columns(conn, table)
            self._validate_referenced_columns(
                available_columns,
                filters=normalized_filters,
                sort=(),
                selected=(),
            )
            where_sql, params = self._build_where(normalized_filters)
            sql = f'SELECT 1 FROM {_quote_identifier(table)}'
            if where_sql:
                sql += f" WHERE {where_sql}"
            sql += " LIMIT 1"

            fingerprint = _query_fingerprint(sql, params)
            try:
                row = conn.execute(sql, params).fetchone()
            except sqlite3.Error as exc:
                raise QueryExecutionError(
                    str(exc),
                    fingerprint=fingerprint,
                ) from exc

        return row is not None

    def schema(self, table: str) -> Tuple[str, ...]:
        """Return queryable columns for a table."""

        table = _validate_identifier(table, "table")
        with closing(self._connect()) as conn:
            return tuple(sorted(self._resolve_columns(conn, table)))

    def _connect(self) -> sqlite3.Connection:
        try:
            if self.read_only:
                resolved = self.database.resolve()
                uri = f"file:{resolved.as_posix()}?mode=ro"
                conn = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=self.timeout_seconds,
                )
            else:
                conn = sqlite3.connect(
                    str(self.database),
                    timeout=self.timeout_seconds,
                )
        except sqlite3.Error as exc:
            raise QueryExecutionError(
                f"unable to open database: {exc}"
            ) from exc

        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        if self.read_only:
            conn.execute("PRAGMA query_only = ON")
        return conn

    def _resolve_columns(
        self,
        conn: sqlite3.Connection,
        table: str,
    ) -> frozenset[str]:
        if self.allowed_schema is not None:
            columns = self.allowed_schema.get(table)
            if columns is None:
                raise QueryValidationError(
                    f"table '{table}' is not in allowed_schema"
                )
            return columns

        quoted = _quote_identifier(table)
        try:
            rows = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
        except sqlite3.Error as exc:
            raise QueryExecutionError(
                f"unable to inspect table '{table}': {exc}"
            ) from exc

        columns = frozenset(str(row["name"]) for row in rows)
        if not columns:
            raise QueryValidationError(
                f"table '{table}' does not exist or has no columns"
            )
        return columns

    def _build_select(
        self,
        spec: QuerySpec,
        available_columns: frozenset[str],
    ) -> tuple[str, Tuple[Any, ...]]:
        self._validate_referenced_columns(
            available_columns,
            filters=spec.filters,
            sort=spec.sort,
            selected=spec.columns,
        )

        selected = self._selected_columns(spec, available_columns)
        projection = ", ".join(_quote_identifier(c) for c in selected)
        distinct = "DISTINCT " if spec.distinct else ""

        sql = (
            f"SELECT {distinct}{projection} "
            f"FROM {_quote_identifier(spec.table)}"
        )
        params: List[Any] = []

        where_sql, where_params = self._build_where(spec.filters)
        if where_sql:
            sql += f" WHERE {where_sql}"
            params.extend(where_params)

        if spec.sort:
            order_sql = ", ".join(
                f"{_quote_identifier(item.field)} {item.direction.value.upper()}"
                for item in spec.sort
            )
            sql += f" ORDER BY {order_sql}"

        sql += " LIMIT ? OFFSET ?"
        params.extend([spec.limit, spec.offset])
        return sql, tuple(params)

    @staticmethod
    def _build_where(
        filters: Sequence[QueryFilter],
    ) -> tuple[str, Tuple[Any, ...]]:
        clauses: List[str] = []
        params: List[Any] = []

        binary_ops = {
            QueryOperator.EQ: "=",
            QueryOperator.NE: "!=",
            QueryOperator.LT: "<",
            QueryOperator.LTE: "<=",
            QueryOperator.GT: ">",
            QueryOperator.GTE: ">=",
            QueryOperator.LIKE: "LIKE",
        }

        for item in filters:
            column = _quote_identifier(item.field)

            if item.operator in binary_ops:
                clauses.append(f"{column} {binary_ops[item.operator]} ?")
                params.append(item.value)
                continue

            if item.operator is QueryOperator.IS_NULL:
                clauses.append(f"{column} IS NULL")
                continue

            if item.operator is QueryOperator.NOT_NULL:
                clauses.append(f"{column} IS NOT NULL")
                continue

            if item.operator in {QueryOperator.IN, QueryOperator.NOT_IN}:
                values = tuple(item.value)
                if not values:
                    clauses.append(
                        "0 = 1"
                        if item.operator is QueryOperator.IN
                        else "1 = 1"
                    )
                    continue

                placeholders = ", ".join("?" for _ in values)
                keyword = "IN" if item.operator is QueryOperator.IN else "NOT IN"
                clauses.append(f"{column} {keyword} ({placeholders})")
                params.extend(values)
                continue

            raise QueryValidationError(
                f"unsupported query operator: {item.operator.value}"
            )

        return " AND ".join(clauses), tuple(params)

    @staticmethod
    def _validate_referenced_columns(
        available: frozenset[str],
        *,
        filters: Sequence[QueryFilter],
        sort: Sequence[QuerySort],
        selected: Sequence[str],
    ) -> None:
        referenced = set(selected)
        referenced.update(item.field for item in filters)
        referenced.update(item.field for item in sort)

        unknown = sorted(referenced - set(available))
        if unknown:
            raise QueryValidationError(
                f"unknown or disallowed column(s): {', '.join(unknown)}"
            )

    @staticmethod
    def _selected_columns(
        spec: QuerySpec,
        available_columns: frozenset[str],
    ) -> Tuple[str, ...]:
        if spec.columns:
            return spec.columns
        # SQLite PRAGMA order is not preserved once a set is used; sorting
        # gives a deterministic projection and result contract.
        return tuple(sorted(available_columns))


def _validate_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise QueryValidationError(f"{field_name} must be a string")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise QueryValidationError(
            f"{field_name} contains an invalid SQL identifier: {value!r}"
        )
    return normalized


def _quote_identifier(identifier: str) -> str:
    # Identifier has already passed the strict regex above.
    return f'"{identifier}"'


def _normalize_allowed_schema(
    schema: Optional[Mapping[str, Iterable[str]]],
) -> Optional[Dict[str, frozenset[str]]]:
    if schema is None:
        return None
    if not isinstance(schema, Mapping):
        raise QueryValidationError("allowed_schema must be a mapping")

    normalized: Dict[str, frozenset[str]] = {}
    for table, columns in schema.items():
        table_name = _validate_identifier(table, "allowed table")
        if isinstance(columns, (str, bytes)):
            raise QueryValidationError(
                f"columns for table '{table_name}' must be an iterable of names"
            )
        column_set = frozenset(
            _validate_identifier(column, "allowed column")
            for column in columns
        )
        if not column_set:
            raise QueryValidationError(
                f"allowed_schema for table '{table_name}' must not be empty"
            )
        normalized[table_name] = column_set

    return normalized


def _query_fingerprint(sql: str, params: Sequence[Any]) -> str:
    # Values are included in the local fingerprint but neither SQL nor values
    # are logged. This supports correlation without leaking query content.
    payload = json.dumps(
        {
            "sql": sql,
            "params": [repr(value) for value in params],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    "QueryError",
    "QueryExecutionError",
    "QueryFilter",
    "QueryOperator",
    "QueryResult",
    "QuerySort",
    "QuerySpec",
    "QueryValidationError",
    "SQLiteQueryService",
    "SortDirection",
]


# -------------------------------------------------------------------------
# Temporary self-test block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("\n=== Running Queries self-test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "query_test.sqlite3"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """
                CREATE TABLE items (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    score REAL NOT NULL,
                    active INTEGER NOT NULL
                )
                """
            )
            conn.executemany(
                "INSERT INTO items(name, category, score, active) VALUES (?, ?, ?, ?)",
                [
                    ("Alpha", "a", 0.91, 1),
                    ("Beta", "b", 0.73, 1),
                    ("Gamma", "a", 0.88, 0),
                    ("Delta", "a", 0.95, 1),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        service = SQLiteQueryService(
            db_path,
            read_only=True,
            allowed_schema={
                "items": {"id", "name", "category", "score", "active"}
            },
        )

        result = service.select(
            QuerySpec(
                table="items",
                columns=("id", "name", "score"),
                filters=(
                    QueryFilter("category", QueryOperator.EQ, "a"),
                    QueryFilter("active", QueryOperator.EQ, 1),
                ),
                sort=(QuerySort("score", SortDirection.DESC),),
                limit=10,
            )
        )
        assert result.row_count == 2
        assert result.rows[0]["name"] == "Delta"
        assert service.count(
            "items",
            filters=(QueryFilter("category", QueryOperator.EQ, "a"),),
        ) == 3
        assert service.exists(
            "items",
            filters=(QueryFilter("name", QueryOperator.EQ, "Beta"),),
        )

        try:
            service.select(
                QuerySpec(
                    table="items",
                    columns=("name; DROP TABLE items",),
                )
            )
        except QueryValidationError:
            pass
        else:
            raise AssertionError("Unsafe identifier should fail")

    print("✔ Queries self-test passed")
