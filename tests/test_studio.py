"""Tests for forge studio server, telemetry, spec architect, and API endpoints."""

import io
import json
import tempfile
import unittest
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from forge.studio import (
    StudioServer,
    StudioTelemetry,
    find_todo_path,
    lint_todo_content,
    make_studio_handler,
    requeue_stuck_items,
)


class TestStudioTelemetry(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.telemetry = StudioTelemetry(project_dir=Path(self.tmp_dir.name))

    def tearDown(self):
        self.telemetry._stop_poller.set()
        self.tmp_dir.cleanup()

    def test_initial_state(self):
        state = self.telemetry.get_state()
        self.assertEqual(state["status"], "idle")
        self.assertFalse(state["is_running"])
        self.assertIn("tokens", state)
        self.assertIn("gpu", state)
        self.assertIn("ollama", state)
        self.assertEqual(state["current_task"]["stage"], "idle")

    def test_log_listener(self):
        self.telemetry.on_log("Test message")
        state = self.telemetry.get_state()
        self.assertTrue(any("Test message" in entry["msg"] for entry in state["recent_logs"]))

    def test_stage_change(self):
        self.telemetry.set_stage(1, "In `foo.py`, add bar", "coding", ["foo.py"])
        state = self.telemetry.get_state()
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["current_task"]["stage"], "coding")
        self.assertEqual(state["current_task"]["files"], ["foo.py"])

    def test_token_update(self):
        self.telemetry.update_tokens(prompt_tokens=100, completion_tokens=50, seconds=1.0)
        state = self.telemetry.get_state()
        self.assertEqual(state["tokens"]["prompt"], 100)
        self.assertEqual(state["tokens"]["completion"], 50)
        self.assertEqual(state["tokens"]["total"], 150)
        self.assertEqual(state["tokens"]["tps"], 50.0)

    def test_runner_stop_when_not_active(self):
        ok, msg = self.telemetry.stop_runner()
        self.assertTrue(ok)
        self.assertIn("No active runner", msg)


class TestSpecArchitectHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_find_todo_path_defaults(self):
        p = find_todo_path(self.project_dir)
        self.assertEqual(p.name, "TODO.md")

    def test_find_todo_path_existing_lowercase(self):
        (self.project_dir / "todo.md").write_text("- [ ] item\n", encoding="utf-8")
        p = find_todo_path(self.project_dir)
        self.assertEqual(p.name, "todo.md")

    def test_lint_todo_content(self):
        # Valid item with backticked file
        valid_content = "- [ ] In `src/app.py`, add helper\n"
        res = lint_todo_content(valid_content, self.project_dir)
        self.assertTrue(res["valid"])
        self.assertEqual(res["total_items"], 1)

        # Invalid item without backticked file
        invalid_content = "- [ ] Just do something with no file\n"
        res = lint_todo_content(invalid_content, self.project_dir)
        self.assertFalse(res["valid"])
        self.assertTrue(any(iss["fatal"] for iss in res["results"][0]["issues"]))

    def test_requeue_stuck_items(self):
        content = (
            "- [x] In `a.py`, done already\n"
            "- [!] In `b.py`, blocked item\n"
            "- [?] In `c.py`, needs review\n"
            "- [ ] In `d.py`, still open\n"
        )
        (self.project_dir / "TODO.md").write_text(content, encoding="utf-8")

        count = requeue_stuck_items(self.project_dir)
        self.assertEqual(count, 2)

        new_content = (self.project_dir / "TODO.md").read_text(encoding="utf-8")
        self.assertIn("- [x] In `a.py`, done already", new_content)
        self.assertIn("- [ ] In `b.py`, blocked item", new_content)
        self.assertIn("- [ ] In `c.py`, needs review", new_content)
        self.assertIn("- [ ] In `d.py`, still open", new_content)

        # Idempotent: nothing left to requeue on a second pass.
        self.assertEqual(requeue_stuck_items(self.project_dir), 0)

    def test_requeue_stuck_items_no_todo_file(self):
        empty_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(empty_dir, ignore_errors=True))
        self.assertEqual(requeue_stuck_items(empty_dir), 0)


class TestStudioServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.project_dir = Path(cls.tmp_dir.name)
        (cls.project_dir / "TODO.md").write_text("- [ ] In `main.py`, init app\n", encoding="utf-8")
        cls.server = StudioServer(project_dir=cls.project_dir, port=18888)
        cls.server.start(in_background=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.telemetry._stop_poller.set()
        cls.server.stop()
        cls.tmp_dir.cleanup()

    def test_get_root_html(self):
        req = urllib.request.Request("http://127.0.0.1:18888/")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("<title>Forge Studio</title>", content)
            self.assertIn("FORGE", content)
            self.assertIn("Spec Architect", content)
            self.assertIn("Diff & Critic", content)

    def test_api_status(self):
        req = urllib.request.Request("http://127.0.0.1:18888/api/status")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "idle")
            self.assertIn("tokens", data)
            self.assertIn("is_running", data)

    def test_api_projects(self):
        req = urllib.request.Request("http://127.0.0.1:18888/api/projects")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("projects", data)
            self.assertIsInstance(data["projects"], list)

    def test_api_todo_get_and_post(self):
        # GET
        req = urllib.request.Request("http://127.0.0.1:18888/api/todo")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["filename"], "TODO.md")
            self.assertIn("init app", data["content"])

        # POST
        payload = json.dumps({"content": "- [ ] In `test.py`, add test\n"}).encode("utf-8")
        req_post = urllib.request.Request(
            "http://127.0.0.1:18888/api/todo",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_post, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["ok"])

        # Verify updated
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("add test", data["content"])
            self.assertTrue(len(data.get("items", [])) >= 1)

    def test_api_todo_toggle_and_add(self):
        # Add new item
        payload_add = json.dumps({"task": "- [ ] In `new_file.py`, add new function"}).encode("utf-8")
        req_add = urllib.request.Request(
            "http://127.0.0.1:18888/api/todo/add",
            data=payload_add,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_add, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["ok"])

        # Fetch todo details and verify item exists
        req = urllib.request.Request("http://127.0.0.1:18888/api/todo")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items", [])
            last_item = items[-1]
            self.assertIn("new_file.py", last_item["text"])

        # Toggle item status
        payload_toggle = json.dumps({"line": last_item["line"], "status": "x"}).encode("utf-8")
        req_toggle = urllib.request.Request(
            "http://127.0.0.1:18888/api/todo/toggle",
            data=payload_toggle,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_toggle, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["ok"])

        # Verify status is now done
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            last_item = data["items"][-1]
            self.assertEqual(last_item["status"], "x")

    def test_api_todo_requeue_stuck(self):
        payload = json.dumps({
            "content": "- [!] In `x.py`, blocked\n- [?] In `y.py`, needs review\n- [ ] In `z.py`, open\n"
        }).encode("utf-8")
        req_set = urllib.request.Request(
            "http://127.0.0.1:18888/api/todo",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_set, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)

        req = urllib.request.Request(
            "http://127.0.0.1:18888/api/todo/requeue-stuck", data=b"", method="POST"
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(res["ok"])
            self.assertEqual(res["count"], 2)
            statuses = [it["status"] for it in res["details"]["items"]]
            self.assertEqual(statuses, [" ", " ", " "])

    def test_api_lint_spec(self):
        payload = json.dumps({"content": "- [ ] In `src/foo.py`, add func\n"}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:18888/api/lint-spec",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["valid"])
            self.assertEqual(data["total_items"], 1)

    def test_api_branches(self):
        req = urllib.request.Request("http://127.0.0.1:18888/api/branches")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("branches", data)

    @patch("forge.runner.main")
    def test_api_run_and_stop(self, mock_runner_main):
        mock_runner_main.return_value = 0
        payload = json.dumps({"backend": "lite", "max_items": 1}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:18888/api/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["ok"])

        # Send stop
        req_stop = urllib.request.Request(
            "http://127.0.0.1:18888/api/stop",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_stop, timeout=3.0) as resp:
            self.assertIn(resp.status, (200, 400))


if __name__ == "__main__":
    unittest.main()
