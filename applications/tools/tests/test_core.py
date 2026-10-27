"""Tests for the shared Python tools: validation, errors, storage, preferences, configuration and the SLAI link."""
from __future__ import annotations

import errno
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from applications.tools import config, errors, slai, storage, validation
from applications.tools.errors import BadRequest


class ValidationTests(unittest.TestCase):
    def test_text(self) -> None:
        self.assertEqual(validation.text("  Hello ", "title", 20), "Hello")
        self.assertEqual(validation.text("  keep  ", "t", 20, empty=True), "  keep  ")
        self.assertEqual(validation.text("", "t", 20, empty=True), "")
        for bad in (None, 5, "", "   ", "x" * 21, "a\x00b", "line\nbreak"):
            with self.assertRaises(BadRequest, msg=repr(bad)):
                validation.text(bad, "title", 20)
        self.assertEqual(validation.text("a\nb", "body", 20, multiline=True), "a\nb")
        with self.assertRaises(BadRequest):
            validation.text("a\x07b", "body", 20, multiline=True)

    def test_instant_normalises_to_utc(self) -> None:
        self.assertEqual(validation.instant("2026-10-27T12:00:00+02:00", "t"), "2026-10-27T10:00:00+00:00")
        self.assertEqual(validation.instant("2026-10-27T10:00:00.000Z", "t"), "2026-10-27T10:00:00+00:00")
        self.assertIsNone(validation.instant(None, "t", nullable=True))
        for bad in (None, 3, "yesterday", "2026-10-27T10:00:00", "2026-13-01T00:00:00+00:00", "x" * 80):
            with self.assertRaises(BadRequest, msg=repr(bad)):
                validation.instant(bad, "t")

    def test_stored_instants_sort_as_strings(self) -> None:
        a = validation.instant("2026-01-01T00:30:00+01:00", "t")       # 2025-12-31T23:30 UTC
        b = validation.instant("2026-01-01T00:00:00+00:00", "t")
        assert a is not None and b is not None
        self.assertLess(a, b)

    def test_zone(self) -> None:
        self.assertEqual(validation.zone("Europe/Amsterdam"), "Europe/Amsterdam")
        self.assertEqual(validation.zone("UTC"), "UTC")
        for bad in ("Mars/Olympus", "../etc/passwd", "/etc/localtime", "", 5):
            with self.assertRaises(BadRequest, msg=repr(bad)):
                validation.zone(bad)

    def test_choice_does_not_let_booleans_pass_for_numbers(self) -> None:
        self.assertEqual(validation.choice(1, (0, 1), "n"), 1)
        with self.assertRaises(BadRequest):
            validation.choice(True, (0, 1), "n")
        with self.assertRaises(BadRequest):
            validation.choice("x", ("a", "b"), "n")

    def test_numbers(self) -> None:
        self.assertEqual(validation.integer(5, "n", 1, 9), 5)
        self.assertIsNone(validation.integer(None, "n", 1, 9, nullable=True))
        for bad in (True, 5.0, "5", 0, 10, None):
            with self.assertRaises(BadRequest, msg=repr(bad)):
                validation.integer(bad, "n", 1, 9)
        self.assertEqual(validation.number(1, "x", 0, 2), 1.0)
        for bad in (True, float("nan"), "1", 3):
            with self.assertRaises(BadRequest, msg=repr(bad)):
                validation.number(bad, "x", 0, 2)
        self.assertTrue(validation.boolean(True, "b"))
        with self.assertRaises(BadRequest):
            validation.boolean(1, "b")

    def test_record_id_and_like(self) -> None:
        good = validation.new_id()
        self.assertEqual(validation.record_id(good), good)
        for bad in ("../x", "abc", good.upper(), 5, good + "0"):
            with self.assertRaises(BadRequest, msg=repr(bad)):
                validation.record_id(bad)
        self.assertEqual(validation.like_pattern("50%_\\"), "%50\\%\\_\\\\%")


class ErrorTests(unittest.TestCase):
    def test_envelope(self) -> None:
        err = errors.NotFound("gone", details={"id": 1}, hint="try again")
        self.assertEqual((err.status, err.to_dict()), (404, {"code": "not_found", "message": "gone", "hint": "try again", "details": {"id": 1}}))

    def test_classify_oserror(self) -> None:
        cases = {errno.ENOSPC: "disk_full", errno.EACCES: "permission_denied", errno.EXDEV: "cross_volume", errno.ENOENT: "invalid_path",
                 errno.EIO: "unexpected"}
        for number, reason in cases.items():
            self.assertEqual(errors.classify_oserror(OSError(number, "x")).reason, reason)
        win = OSError(0, "x")
        win.winerror = 32  # type: ignore[attr-defined]
        self.assertEqual(errors.classify_oserror(win).reason, "locked")
        self.assertIsInstance(errors.classify_oserror(OSError(errno.ENOSPC, "x")), OSError)

    def test_classify_sqlite(self) -> None:
        self.assertEqual(errors.classify_sqlite(sqlite3.OperationalError("database is locked")).reason, "locked")
        self.assertEqual(errors.classify_sqlite(sqlite3.OperationalError("database or disk is full")).reason, "disk_full")
        self.assertEqual(errors.classify_sqlite(sqlite3.DatabaseError("database disk image is malformed")).reason, "corrupt")


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_folders(self) -> None:
        old = {k: os.environ.get(k) for k in ("XDG_DATA_HOME", "LOCALAPPDATA")}
        try:
            os.environ["XDG_DATA_HOME"] = "/x/share"
            os.environ["LOCALAPPDATA"] = "C:\\L"
            path = storage.default_data_dir("splanner", "SPlanner")
            if os.name == "nt":
                self.assertEqual(path, Path("C:\\L") / "SLAI" / "SPlanner")
            else:
                self.assertEqual(path, Path("/x/share/slai/splanner"))
            self.assertEqual(storage.handoff_dir(path), path.parent / "handoff")
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_atomic_write_replaces_and_leaves_no_temporaries(self) -> None:
        target = self.dir / "sub" / "a.txt"
        storage.atomic_write_text(target, "one")
        storage.atomic_write_text(target, "two")
        self.assertEqual(target.read_text(), "two")
        self.assertEqual([p.name for p in target.parent.iterdir()], ["a.txt"])

    def test_atomic_write_exclusive_never_overwrites(self) -> None:
        target = self.dir / "once.txt"
        storage.atomic_write(target, b"first", exclusive=True)
        with self.assertRaises(FileExistsError):
            storage.atomic_write(target, b"second", exclusive=True)
        self.assertEqual(target.read_bytes(), b"first")
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name != "sub"], ["once.txt"])

    def test_atomic_write_falls_back_to_writing_in_place_when_renaming_is_refused(self) -> None:
        import errno
        from unittest import mock

        def refuse(*_a, **_k):
            raise OSError(errno.EXDEV, "cannot move the file to a different disk drive")
        for exclusive in (False, True):
            target = self.dir / f"fallback-{exclusive}.txt"
            with mock.patch("os.replace", refuse), mock.patch("time.sleep"):
                storage.atomic_write(target, b"saved", exclusive=exclusive)
            self.assertEqual(target.read_bytes(), b"saved")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir() if p.is_file()), ["fallback-False.txt", "fallback-True.txt"])

    def test_atomic_write_json_is_stable(self) -> None:
        target = self.dir / "a.json"
        storage.atomic_write_json(target, {"b": 1, "a": "é"})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"a": "é", "b": 1})
        with self.assertRaises(ValueError):
            storage.atomic_write_json(target, {"x": float("nan")})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"a": "é", "b": 1})      # the old file survived

    def test_prefs_validate_persist_and_recover(self) -> None:
        schema = {"theme": storage.Pref(str, allowed=("light", "dark")), "scale": storage.Pref(float, low=0.8, high=1.6),
                  "n": storage.Pref(int, low=0, high=23), "on": storage.Pref(bool), "name": storage.Pref(str, max_len=5, pattern=r"[A-Za-z ]*"),
                  "wk": storage.Pref(int, allowed=(0, 1))}
        path = self.dir / "prefs.json"
        prefs = storage.Prefs(path, schema)
        self.assertEqual(prefs.update({"theme": "dark", "scale": 1, "n": 8, "on": True}), {"theme": "dark", "scale": 1.0, "n": 8, "on": True})
        for bad in ({"theme": "neon"}, {"scale": 2}, {"n": 8.5}, {"n": True}, {"on": 1}, {"zzz": 1}, {"name": "toolong"}, {"name": "a1"}, {"wk": 2},
                    {"wk": True}):
            with self.assertRaises(BadRequest, msg=repr(bad)):
                prefs.update(bad)
        self.assertEqual(storage.Prefs(path, schema).data["theme"], "dark")
        # a hand-edited file: valid keys survive, invalid values and unknown keys are dropped, nothing raises
        path.write_text(json.dumps({"theme": "neon", "n": 3, "unknown": 1}), encoding="utf-8")
        self.assertEqual(storage.Prefs(path, schema).data, {"n": 3})
        path.write_text("{not json", encoding="utf-8")
        broken = storage.Prefs(path, schema)
        self.assertEqual(broken.data, {})
        self.assertIn("reset", broken.recovered or "")


class ConfigTests(unittest.TestCase):
    FIELDS = {"port": config.Field(8769, 1, 65535), "poll": config.Field(20, 5, 3600), "flag": config.Field(False, kind=bool)}

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.yaml"
        slai.activate(slai.find_slai_root())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_defaults_when_missing(self) -> None:
        values, note = config.load_section(self.path, "app", self.FIELDS)
        self.assertEqual(values, {"port": 8769, "poll": 20, "flag": False})
        self.assertIn("built-in defaults", note or "")

    def test_reads_yaml_through_the_canonical_repository(self) -> None:
        self.path.write_text("app:\n  port: 9000\n  poll: 30\n  flag: true\n", encoding="utf-8")
        values, note = config.load_section(self.path, "app", self.FIELDS)
        self.assertEqual(values, {"port": 9000, "poll": 30, "flag": True})
        self.assertIsNone(note)

    def test_overrides_win_and_invalid_values_name_the_setting(self) -> None:
        self.path.write_text("app:\n  port: 9000\n", encoding="utf-8")
        self.assertEqual(config.load_section(self.path, "app", self.FIELDS, overrides={"port": 9100, "poll": None})[0]["port"], 9100)
        # A fresh file name each time: ConfigRepository may cache by size and modification time, and these files have the same size
        # and are written within the same clock tick on coarse-grained file systems.
        for index, (text, name) in enumerate((("app:\n  port: 'x'\n", "port"), ("app:\n  poll: 1\n", "poll"), ("app:\n  flag: 1\n", "flag"),
                                              ("app:\n  port: true\n", "port"))):
            path = self.path.with_name(f"invalid-{index}.yaml")
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                config.load_section(path, "app", self.FIELDS, label="Test configuration")
            self.assertIn(name, str(caught.exception))
        broken = self.path.with_name("not-a-mapping.yaml")
        broken.write_text("app: [1, 2]\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            config.load_section(broken, "app", self.FIELDS)


class SlaiLinkTests(unittest.TestCase):
    def test_finds_this_checkout_and_the_canonical_sldc(self) -> None:
        root = slai.find_slai_root()
        self.assertIsNotNone(root)
        assert root is not None
        self.assertTrue(slai.looks_like_slai_root(root))
        slai.activate(root)
        module = slai.sldc()
        self.assertEqual(module.__name__, "src.utils.sldc")
        self.assertTrue(slai.sldc_info()["available"])
        self.assertEqual(slai.applications_dir(), root / "applications")
        self.assertEqual(slai.shared_assets_dir(), root / "applications" / "assets")

    def test_explicit_wrong_path_is_ignored_not_trusted(self) -> None:
        self.assertNotEqual(slai.find_slai_root("/definitely/not/here"), Path("/definitely/not/here"))


if __name__ == "__main__":
    unittest.main()
