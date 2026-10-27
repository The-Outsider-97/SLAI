"""Transactional local records and SLDC-backed notes.

Events, tasks, reminders and board coordinates are *not* stored in SLDC. Note
content is stored by src.utils.sldc, not a secondary document implementation.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.utils import sldc


class PlannerError(ValueError):
    """Invalid user input, missing object, or conflicting revision."""


class Missing(PlannerError):
    pass


class Conflict(PlannerError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id() -> str:
    return uuid.uuid4().hex


def _text(value: Any, label: str, limit: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise PlannerError(f"{label} must be text of 1–{limit} characters")
    return value.strip() if not empty else value


def _instant(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise PlannerError(f"{label} requires an ISO-8601 timestamp with UTC offset")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlannerError(f"Invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PlannerError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _zone(value: Any) -> str:
    name = _text(value, "timeZone", 100)
    try:
        ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise PlannerError("Unknown IANA time zone") from exc
    return name


def _choice(value: Any, choices: tuple[str, ...], label: str) -> str:
    if value not in choices:
        raise PlannerError(f"Invalid {label}; expected {', '.join(choices)}")
    return value


class PlannerStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.note_dir = self.data_dir / "notes"
        self.note_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.data_dir / "planner.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self._schema()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def _schema(self) -> None:
        with self._lock, self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS notes (
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, version INTEGER NOT NULL,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, start_at TEXT NOT NULL,
                  end_at TEXT NOT NULL, time_zone TEXT NOT NULL, all_day INTEGER NOT NULL,
                  description TEXT NOT NULL, color TEXT NOT NULL, note_id TEXT REFERENCES notes(id),
                  version INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                  CHECK(end_at > start_at));
                CREATE INDEX IF NOT EXISTS events_by_time ON events(start_at,end_at);
                CREATE TABLE IF NOT EXISTS tasks (
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL,
                  priority TEXT NOT NULL, due_at TEXT, note_id TEXT REFERENCES notes(id),
                  version INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS tasks_by_due ON tasks(due_at);
                CREATE TABLE IF NOT EXISTS reminders (
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, due_at TEXT NOT NULL,
                  status TEXT NOT NULL, entity_type TEXT, entity_id TEXT,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS reminders_due ON reminders(status,due_at);
                CREATE TABLE IF NOT EXISTS stickies (
                  id TEXT PRIMARY KEY, note_id TEXT NOT NULL UNIQUE REFERENCES notes(id),
                  x REAL NOT NULL, y REAL NOT NULL, color TEXT NOT NULL,
                  version INTEGER NOT NULL);
            """)

    def _all(self, table: str) -> list[dict[str, Any]]:
        with self._lock:
            query = (f"SELECT * FROM {table} ORDER BY created_at DESC" if table != "stickies"
                     else "SELECT * FROM stickies")
            return [dict(r) for r in self.db.execute(query)]

    def snapshot(self) -> dict[str, Any]:
        return {"events": self._all("events"), "tasks": self._all("tasks"),
                "notes": self._all("notes"), "reminders": self._all("reminders"),
                "stickies": self._all("stickies")}

    def _row(self, table: str, item_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute(f"SELECT * FROM {table} WHERE id=?", (item_id,)).fetchone()
        if row is None:
            raise Missing(f"{table} record was not found")
        return dict(row)

    def _mutate(self, table: str, item_id: str, allowed: dict[str, Any],
                version: int | None = None) -> dict[str, Any]:
        with self._lock, self.db:
            old = self._row(table, item_id)
            if version is not None and old.get("version") != version:
                raise Conflict("This item was modified elsewhere; refresh before saving")
            fields = list(allowed)
            if not fields:
                return old
            if "version" in old:
                allowed["version"] = old["version"] + 1
            if "updated_at" in old:
                allowed["updated_at"] = _now()
            pairs = ", ".join(f"{name}=?" for name in allowed)
            self.db.execute(f"UPDATE {table} SET {pairs} WHERE id=?",
                            (*allowed.values(), item_id))
        return self._row(table, item_id)

    def create_event(self, data: dict[str, Any]) -> dict[str, Any]:
        start = _instant(data.get("start_at"), "start_at")
        end = _instant(data.get("end_at"), "end_at")
        if end <= start:
            raise PlannerError("An event must end after it starts")
        color = _text(data.get("color", "blue"), "color", 24)
        color = _choice(color, ("blue", "violet", "green", "pink", "amber"), "color")
        all_day = data.get("all_day", False)
        if type(all_day) is not bool:
            raise PlannerError("all_day must be true or false")
        timestamp, item_id = _now(), _id()
        row = (item_id, _text(data.get("title"), "title", 180), start, end,
               _zone(data.get("time_zone", "UTC")), int(all_day),
               _text(data.get("description", ""), "description", 5000, empty=True),
               color, 1, timestamp, timestamp)
        with self._lock, self.db:
            self.db.execute("""INSERT INTO events
              (id,title,start_at,end_at,time_zone,all_day,description,color,version,created_at,updated_at)
              VALUES (?,?,?,?,?,?,?,?,?,?,?)""", row)
        return self._row("events", item_id)

    def update_event(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        old = self._row("events", item_id)
        updates: dict[str, Any] = {}
        for key in ("title", "description", "start_at", "end_at", "time_zone", "color", "all_day"):
            if key not in data:
                continue
            value = data[key]
            if key in ("title", "description"):
                value = _text(value, key, 180 if key == "title" else 5000, empty=key == "description")
            elif key in ("start_at", "end_at"):
                value = _instant(value, key)
            elif key == "time_zone":
                value = _zone(value)
            elif key == "color":
                value = _choice(value, ("blue", "violet", "green", "pink", "amber"), key)
            elif key == "all_day":
                if type(value) is not bool:
                    raise PlannerError("all_day must be boolean")
                value = int(value)
            updates[key] = value
        if updates.get("end_at", old["end_at"]) <= updates.get("start_at", old["start_at"]):
            raise PlannerError("An event must end after it starts")
        return self._mutate("events", item_id, updates, data.get("version"))

    def create_task(self, data: dict[str, Any]) -> dict[str, Any]:
        timestamp, item_id = _now(), _id()
        row = (item_id, _text(data.get("title"), "title", 180), "open",
               _choice(data.get("priority", "normal"), ("low", "normal", "high"), "priority"),
               _instant(data.get("due_at"), "due_at", nullable=True), 1, timestamp, timestamp)
        with self._lock, self.db:
            self.db.execute("""INSERT INTO tasks
              (id,title,status,priority,due_at,version,created_at,updated_at)
              VALUES (?,?,?,?,?,?,?,?)""", row)
        return self._row("tasks", item_id)

    def update_task(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        updates = {}
        for key in ("title", "status", "priority", "due_at"):
            if key not in data:
                continue
            value = data[key]
            if key == "title":
                value = _text(value, key, 180)
            elif key == "status":
                value = _choice(value, ("open", "done"), key)
            elif key == "priority":
                value = _choice(value, ("low", "normal", "high"), key)
            else:
                value = _instant(value, key, nullable=True)
            updates[key] = value
        return self._mutate("tasks", item_id, updates, data.get("version"))

    def create_note(self, data: dict[str, Any]) -> dict[str, Any]:
        title = _text(data.get("title"), "title", 180)
        content = _text(data.get("content", ""), "content", 50000, empty=True)
        item_id, timestamp = _id(), _now()
        package = sldc.new_package(title, [sldc.make_block("paragraph", content)], doc_id=f"splanner-{item_id}")
        path = self.note_dir / f"{item_id}.sldc"
        sldc.write_package(path, package)
        try:
            with self._lock, self.db:
                self.db.execute("INSERT INTO notes VALUES (?,?,?,?,?)", (item_id, title, 1, timestamp, timestamp))
        except sqlite3.Error:
            path.unlink(missing_ok=True)
            raise
        return self.get_note(item_id)

    def get_note(self, item_id: str) -> dict[str, Any]:
        record = self._row("notes", item_id)
        package = sldc.read_package(self.note_dir / f"{item_id}.sldc")
        return {**record, "content": sldc.extract_text(package)}

    def update_note(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            existing = self._row("notes", item_id)
            if data.get("version") != existing["version"]:
                raise Conflict("Note version changed; reload to prevent overwriting edits")
            if "title" not in data and "content" not in data:
                return self.get_note(item_id)
            path = self.note_dir / f"{item_id}.sldc"
            package = sldc.read_package(path)
            if package.signatures:
                raise Conflict("Signed documents cannot be edited through the basic note editor")
            if (len(package.document["blocks"]) != 1
                    or package.document["blocks"][0]["type"] != "paragraph"):
                raise Conflict("This note requires SReader or Writer for advanced-content editing")
            title = _text(data.get("title", existing["title"]), "title", 180)
            content = _text(
                data.get("content", package.document["blocks"][0]["text"]),
                "content", 50000, empty=True,
            )
            package.document["title"] = title
            package.document["blocks"][0]["text"] = content
            sldc.write_package(path, package, overwrite=True)
            return self._mutate("notes", item_id, {"title": title}, existing["version"])

    def create_reminder(self, data: dict[str, Any]) -> dict[str, Any]:
        timestamp, item_id = _now(), _id()
        row = (item_id, _text(data.get("title"), "title", 180),
               _instant(data.get("due_at"), "due_at"), "pending", timestamp, timestamp)
        with self._lock, self.db:
            self.db.execute("""INSERT INTO reminders
              (id,title,due_at,status,created_at,updated_at) VALUES (?,?,?,?,?,?)""", row)
        return self._row("reminders", item_id)

    def due_reminders(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.db.execute(
                "SELECT * FROM reminders WHERE status='pending' AND due_at<=? ORDER BY due_at ASC",
                (_now(),))]

    def change_reminder(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        action = _choice(data.get("action"), ("dismiss", "snooze"), "action")
        updates = {"status": "dismissed"} if action == "dismiss" else {
            "due_at": _instant(data.get("due_at"), "due_at"), "status": "pending"}
        if action == "snooze" and updates["due_at"] <= _now():
            raise PlannerError("Snooze time must be in the future")
        return self._mutate("reminders", item_id, updates)

    def create_sticky(self, data: dict[str, Any]) -> dict[str, Any]:
        x, y = self._coords(data)
        color = _choice(data.get("color", "amber"),
                        ("blue", "violet", "green", "pink", "amber"), "color")
        note = self.create_note({"title": data.get("title", "Sticky note"),
                                 "content": data.get("content", "")})
        item_id = _id()
        with self._lock, self.db:
            self.db.execute("INSERT INTO stickies VALUES (?,?,?,?,?,?)",
                            (item_id, note["id"], x, y, color, 1))
        return self._row("stickies", item_id)

    @staticmethod
    def _coords(data: dict[str, Any]) -> tuple[float, float]:
        vals = (data.get("x", 48), data.get("y", 48))
        for value in vals:
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not -10000 <= value <= 10000:
                raise PlannerError("Board coordinates are out of bounds")
        return float(vals[0]), float(vals[1])

    def update_sticky(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        updates = {}
        if "x" in data or "y" in data:
            old = self._row("stickies", item_id)
            x, y = self._coords({"x": data.get("x", old["x"]), "y": data.get("y", old["y"])})
            updates.update(x=x, y=y)
        if "color" in data:
            updates["color"] = _choice(data["color"], ("blue", "violet", "green", "pink", "amber"), "color")
        return self._mutate("stickies", item_id, updates, data.get("version"))

    def delete_item(self, kind: str, item_id: str) -> None:
        if kind not in ("events", "tasks"):
            raise PlannerError("This record type cannot be deleted here")
        with self._lock, self.db:
            self._row(kind, item_id)
            self.db.execute(f"DELETE FROM {kind} WHERE id=?", (item_id,))
