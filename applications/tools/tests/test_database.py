"""Tests for the shared SQLite wrapper: transactions, migrations, backups."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from applications.tools.database import Database, Migration
from applications.tools.errors import Conflict

V1 = Migration(1, "create items", ("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",))
V2 = Migration(2, "add colour and backfill", (
    "ALTER TABLE items ADD COLUMN colour TEXT NOT NULL DEFAULT 'blue'",
    lambda conn: conn.execute("INSERT INTO items (name, colour) VALUES ('seed', 'green')"),
))


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "data.sqlite3"
        self.backups = self.dir / "backups"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fresh_database_runs_every_migration_without_a_backup(self) -> None:
        db = Database(self.path, [V1, V2], backups_dir=self.backups)
        self.assertEqual((db.schema_version, db.migrated, db.migration_backup), (2, [1, 2], None))
        self.assertFalse(db.upgraded)                                  # a brand-new file is not "an upgrade": nothing to tell the person
        self.assertEqual(db.rows("SELECT name, colour FROM items"), [{"name": "seed", "colour": "green"}])
        db.close()

    def test_upgrade_backs_up_first_and_only_runs_new_steps(self) -> None:
        db = Database(self.path, [V1], backups_dir=self.backups)
        db.execute("INSERT INTO items (name) VALUES ('old')")
        db.close()
        db = Database(self.path, [V1, V2], backups_dir=self.backups)
        self.assertEqual(db.migrated, [2])
        self.assertTrue(db.upgraded)
        self.assertIsNotNone(db.migration_backup)
        assert db.migration_backup is not None
        copy = sqlite3.connect(str(db.migration_backup))
        self.assertEqual(copy.execute("SELECT name FROM items").fetchall(), [("old",)])          # the copy predates the migration
        self.assertEqual([row[1] for row in copy.execute("PRAGMA table_info(items)")], ["id", "name"])
        copy.close()
        self.assertEqual(sorted(r["name"] for r in db.rows("SELECT name FROM items")), ["old", "seed"])
        db.close()
        again = Database(self.path, [V1, V2], backups_dir=self.backups)
        self.assertEqual(again.migrated, [])
        again.close()

    def test_a_failing_migration_rolls_back_completely(self) -> None:
        bad = Migration(2, "broken", ("ALTER TABLE items ADD COLUMN half TEXT", "THIS IS NOT SQL"))
        db = Database(self.path, [V1], backups_dir=self.backups)
        db.close()
        with self.assertRaises(Exception):
            Database(self.path, [V1, bad], backups_dir=self.backups)
        db = Database(self.path, [V1], backups_dir=self.backups)
        self.assertEqual(db.schema_version, 1)
        self.assertEqual([row["name"] for row in db.rows("PRAGMA table_info(items)")], ["id", "name"])
        db.close()

    def test_a_newer_database_is_refused_untouched(self) -> None:
        db = Database(self.path, [V1, V2])
        db.close()
        before = self.path.read_bytes()
        with self.assertRaises(Conflict):
            Database(self.path, [V1])
        self.assertEqual(self.path.read_bytes(), before)

    def test_transactions_commit_roll_back_and_nest(self) -> None:
        db = Database(self.path, [V1])
        with db.transaction():
            db.execute("INSERT INTO items (name) VALUES ('a')")
            with db.transaction():                                   # re-entrant: joins the outer transaction
                db.execute("INSERT INTO items (name) VALUES ('b')")
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM items"), 2)
        with self.assertRaises(RuntimeError):
            with db.transaction():
                db.execute("INSERT INTO items (name) VALUES ('c')")
                with db.transaction():
                    db.execute("INSERT INTO items (name) VALUES ('d')")
                raise RuntimeError("boom")
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM items"), 2)       # nothing from the failed transaction survived
        with self.assertRaises(sqlite3.IntegrityError):                     # body errors reach the caller unclassified
            with db.transaction():
                db.execute("INSERT INTO items (name) VALUES (NULL)")
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM items"), 2)
        db.close()

    def test_backup_is_coherent_and_named(self) -> None:
        db = Database(self.path, [V1], backups_dir=self.backups)
        db.execute("INSERT INTO items (name) VALUES ('x')")
        first = db.backup("manual")
        second = db.backup("manual")
        self.assertNotEqual(first, second)
        self.assertTrue(first.name.startswith("manual-") and first.suffix == ".sqlite3")
        copy = sqlite3.connect(str(first))
        self.assertEqual(copy.execute("SELECT name FROM items").fetchall(), [("x",)])
        copy.close()
        self.assertEqual(db.integrity(), "ok")
        db.close()

    def test_a_damaged_file_is_reported_not_overwritten(self) -> None:
        self.path.write_bytes(b"this is not a database" * 200)
        before = self.path.read_bytes()
        from applications.tools.errors import StorageError
        with self.assertRaises(StorageError) as caught:
            Database(self.path, [V1])
        self.assertEqual(caught.exception.reason, "corrupt")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
