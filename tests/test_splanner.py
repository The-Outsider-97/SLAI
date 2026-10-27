"""SPlanner local foundation integration tests with the real SLDC module."""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

from applications.splanner.config import Settings
from applications.splanner.server import make_handler
from applications.splanner.store import Conflict, PlannerError, PlannerStore
from src.utils.sldc import inspect_package, read_package, extract_text


class PlannerStoreTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.TemporaryDirectory()
        self.store = PlannerStore(Path(self.sandbox.name))

    def tearDown(self):
        self.store.close()
        self.sandbox.cleanup()

    def test_event_validates_times_and_utc(self):
        event = self.store.create_event({"title": "BIM review", "start_at": "2026-10-28T09:00:00+01:00",
                                         "end_at": "2026-10-28T10:00:00+01:00",
                                         "time_zone": "Europe/Amsterdam"})
        self.assertEqual(event["start_at"], "2026-10-28T08:00:00+00:00")
        self.assertEqual(event["time_zone"], "Europe/Amsterdam")
        with self.assertRaises(PlannerError):
            self.store.create_event({"title": "Invalid", "start_at": "2026-10-28T09:00",
                                     "end_at": "2026-10-28T10:00"})
        with self.assertRaises(PlannerError):
            self.store.update_event(event["id"], {"end_at": "2026-10-28T07:00:00Z"})

    def test_optimistic_concurrency(self):
        task = self.store.create_task({"title": "Read paper", "priority": "high"})
        edited = self.store.update_task(task["id"], {"status": "done", "version": task["version"]})
        self.assertEqual(edited["version"], 2)
        with self.assertRaises(Conflict):
            self.store.update_task(task["id"], {"title": "Stale write", "version": 1})
        self.assertEqual(self.store._row("tasks", task["id"])["status"], "done")

    def test_sldc_real_roundtrip_and_recovery(self):
        note = self.store.create_note({"title": "Research plan", "content": "Healthy ageing on Bonaire"})
        path = Path(self.sandbox.name) / "notes" / (note["id"] + ".sldc")
        self.assertEqual(inspect_package(path)["version"], "1.0.0")
        self.assertEqual(extract_text(read_package(path)), "Healthy ageing on Bonaire")
        edited = self.store.update_note(note["id"], {"title": "Revised plan", "content": "SLAI work", "version": 1})
        self.assertEqual(edited["version"], 2)
        with self.assertRaises(Conflict):
            self.store.update_note(note["id"], {"content": "Overwrite", "version": 1})
        self.store.close()
        self.store = PlannerStore(Path(self.sandbox.name))
        self.assertEqual(self.store.get_note(note["id"])["content"], "SLAI work")

    def test_stickies_use_sldc_notes(self):
        sticky = self.store.create_sticky({"title": "Idea", "content": "Test", "x": 25, "y": 80})
        self.assertEqual(self.store.get_note(sticky["note_id"])["content"], "Test")
        moved = self.store.update_sticky(sticky["id"], {"x": 91, "y": 20, "version": 1})
        self.assertEqual(moved["version"], 2)
        self.assertEqual(moved["x"], 91)

    def test_reminders_and_persistence(self):
        earlier = datetime.now(timezone.utc) - timedelta(minutes=2)
        item = self.store.create_reminder({"title": "Go to class", "due_at": earlier.isoformat()})
        self.assertEqual(len(self.store.due_reminders()), 1)
        self.store.change_reminder(item["id"], {"action": "snooze", "due_at":
                                    (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()})
        self.assertEqual(self.store.due_reminders(), [])
        self.store.change_reminder(item["id"], {"action": "dismiss"})
        self.assertEqual(self.store._row("reminders", item["id"])["status"], "dismissed")


class PlannerHttpTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.TemporaryDirectory()
        self.store = PlannerStore(Path(self.sandbox.name))
        settings = Settings(data_dir=Path(self.sandbox.name), port=8769)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store, "secret-token", settings))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.store.close()
        self.sandbox.cleanup()

    def request(self, method, path, body=None, *, auth=True, origin=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        headers = {}
        if auth:
            headers["X-SPlanner-Token"] = "secret-token"
        if origin:
            headers["Origin"] = origin
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body)
        else:
            payload = None
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        text = response.read().decode()
        result = response.status, json.loads(text)
        conn.close()
        return result

    def test_http_auth_and_roundtrip(self):
        self.assertEqual(self.request("GET", "/api/bootstrap", auth=False)[0], 401)
        status, event = self.request("POST", "/api/events", {
            "title": "Plan", "start_at": "2026-10-30T09:00:00Z", "end_at": "2026-10-30T10:00:00Z"})
        self.assertEqual(status, 201)
        status, snapshot = self.request("GET", "/api/bootstrap")
        self.assertEqual(status, 200)
        self.assertEqual(snapshot["events"][0]["id"], event["id"])
        self.assertFalse(snapshot["capabilities"]["sharing"])

    def test_mutation_rejects_cross_origin(self):
        status, _ = self.request("POST", "/api/tasks", {"title": "Unsafe"}, origin="https://hostile.example")
        self.assertEqual(status, 403)
        self.assertEqual(self.store.snapshot()["tasks"], [])


if __name__ == "__main__":
    unittest.main()
