"""Tests for Forge Studio's routing, safety guards, merge bookkeeping and
HTTP error handling."""

import io
import json
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from forge import studio
from forge.studio import (
    StudioServer,
    StudioTelemetry,
    _LogTee,
    build_runner_argv,
    critique_diff,
    draft_spec_goal,
    get_todo_details,
    index_run_records,
    list_branch_summaries,
    mark_record_item_done,
    read_static_asset,
    render_studio_html,
    resolve_engine,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def make_repo(root: Path, todo: str) -> Path:
    repo = root / "proj"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "master")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("x = 1\n")
    (repo / "TODO.md").write_text(todo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def make_parked_branch(repo: Path, runs: Path, index: int, item_text: str,
                       status: str = "?") -> str:
    branch = f"aider-loop/item-{index}-20260101-000000"
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "-b", branch)
    (repo / f"feature{index}.py").write_text(f"y = {index}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"item {index}")
    git(repo, "checkout", "-q", "master")
    run_dir = runs / "20260101-000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"item-{index:03d}.json").write_text(json.dumps({
        "branch": branch, "base": base, "status": status,
        "reason": "claude review could not run", "item": item_text,
    }))
    return branch


class TestEngineRouting(unittest.TestCase):
    def test_resolve_engine(self):
        self.assertEqual(resolve_engine("claude"), ("claude", None))
        self.assertEqual(resolve_engine("claude:claude-opus-5"), ("claude", "claude-opus-5"))
        self.assertEqual(resolve_engine("gemini:gemini-2.5-pro"), ("gemini", "gemini-2.5-pro"))
        self.assertEqual(resolve_engine("qwen3-coder:30b"), ("ollama", "qwen3-coder:30b"))
        self.assertEqual(resolve_engine(""), ("ollama", None))
        # A local model whose name merely starts with "claude" is still Ollama.
        self.assertEqual(resolve_engine("claude-distill:7b"), ("ollama", "claude-distill:7b"))

    def _argv(self, **options):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "TODO.md").write_text("- [ ] x\n")
            return build_runner_argv(Path(d), options)

    def test_argv_ollama_doer_with_claude_review_model(self):
        argv = self._argv(backend="lite", model="qwen3-coder:30b",
                          review="claude:claude-sonnet-5", max_items=3)
        self.assertIn("--todo-file", argv)
        self.assertEqual(argv[argv.index("--todo-file") + 1], "TODO.md")
        self.assertEqual(argv[argv.index("--models") + 1], "qwen3-coder:30b")
        self.assertEqual(argv[argv.index("--backend") + 1], "lite")
        self.assertEqual(argv[argv.index("--review") + 1], "claude")
        self.assertEqual(argv[argv.index("--review-model") + 1], "claude-sonnet-5")
        self.assertEqual(argv[argv.index("--max-items") + 1], "3")

    def test_argv_bare_claude_passes_no_model(self):
        argv = self._argv(model="claude", review="claude", fallback_model="claude")
        self.assertEqual(argv[argv.index("--backend") + 1], "claude")
        self.assertNotIn("--models", argv)
        self.assertNotIn("--review-model", argv)
        self.assertNotIn("--fallback-model", argv)
        self.assertEqual(argv[argv.index("--fallback-backend") + 1], "claude")

    def test_argv_claude_fallback_keeps_model(self):
        argv = self._argv(model="qwen3-coder:30b", fallback_model="claude:claude-opus-5")
        self.assertEqual(argv[argv.index("--fallback-model") + 1], "claude-opus-5")

    def test_argv_ollama_fallback_uses_primary_backend(self):
        argv = self._argv(backend="lite", model="qwen2.5-coder:14b", fallback_model="qwen3-coder:30b")
        self.assertEqual(argv[argv.index("--fallback-model") + 1], "qwen3-coder:30b")
        self.assertEqual(argv[argv.index("--fallback-backend") + 1], "lite")

    def test_argv_bad_limit_is_ignored(self):
        self.assertNotIn("--max-items", self._argv(max_items="abc"))


class TestLogTee(unittest.TestCase):
    def test_forwards_only_bare_print_lines(self):
        seen = []
        tee = _LogTee(seen.append)
        with redirect_stdout(io.StringIO()), patch("sys.__stdout__", io.StringIO()):
            tee.write("[2026-09-14 18:45:09] already delivered by log()\n")
            tee.write("No TODO.md found in /x\npartial")
            tee.flush()
        self.assertEqual(seen, ["No TODO.md found in /x", "partial"])


class TestTelemetrySafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.telemetry = StudioTelemetry(Path(self.tmp.name))

    def tearDown(self):
        self.telemetry._stop_poller.set()
        self.tmp.cleanup()

    def test_broadcast_snapshots_data(self):
        q = self.telemetry.subscribe()
        data = {"by_model": {"a": 1}}
        self.telemetry.broadcast("tokens", data)
        data["by_model"]["b"] = 2  # mutation after broadcast must not leak into the event
        self.assertEqual(json.loads(q.get_nowait()["data_json"]), {"by_model": {"a": 1}})

    def test_started_at_resets_per_item(self):
        self.telemetry.set_stage(1, "item one", "worktree")
        first = self.telemetry.current_task["started_at"]
        self.telemetry.set_stage(1, "item one", "coding")
        self.assertEqual(self.telemetry.current_task["started_at"], first)
        with patch("forge.studio.datetime") as dt:
            dt.datetime.now.return_value.isoformat.return_value = "2099-01-01T00:00:00.000+00:00"
            dt.timezone.utc = None
            self.telemetry.set_stage(2, "item two", "worktree")
        self.assertEqual(self.telemetry.current_task["started_at"], "2099-01-01T00:00:00.000+00:00")

    def test_stop_marks_pending_until_worker_exits(self):
        release = threading.Event()
        with patch("forge.runner.main", side_effect=lambda argv: release.wait(5) and 0):
            (Path(self.tmp.name) / "TODO.md").write_text("- [ ] x\n")
            ok, _ = self.telemetry.start_runner({"model": "qwen3-coder:30b"})
            self.assertTrue(ok)
            self.telemetry.stop_runner()
            state = self.telemetry.get_state()
            self.assertTrue(state["is_running"])
            self.assertEqual(state["status"], "stopping")
            release.set()
            self.telemetry.runner_thread.join(5)
        state = self.telemetry.get_state()
        self.assertFalse(state["is_running"])
        self.assertFalse(state["stop_pending"])
        self.assertEqual(state["status"], "idle")


class TestStaticAssets(unittest.TestCase):
    def test_serves_known_assets_only(self):
        body, ctype = read_static_asset("app.js")
        self.assertTrue(ctype.startswith("application/javascript"))
        self.assertIn(b"function refreshBranches", body)
        for bad in ("../studio.py", "index.html/../../x", ".hidden", "studio.py", "nope.css"):
            self.assertIsNone(read_static_asset(bad), bad)

    def test_render_escapes_state_and_fills_every_token(self):
        class Fake:
            def get_state(self):
                return {"project_dir": "/p/<script>", "project_name": "<b>x</b>", "status": "idle",
                        "tokens": {"tps": 1, "peak_tps": 2, "total": 3, "prompt": 1, "completion": 2},
                        "gpu": {"used_mb": 0, "temp_c": 0, "name": "N/A"},
                        "ollama": {"loaded_model": "None", "context": 0}}
        page = render_studio_html(Fake())
        self.assertNotIn("{{", page)
        self.assertNotIn("__INITIAL_STATE__", page)
        self.assertNotIn("<b>x</b>", page.split('id="initial-state-data"')[0])
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", page)
        self.assertNotIn("</script>\"", page)


class TestMergeBookkeeping(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.item = "In `feature1.py`, add y\n  with detail"
        self.repo = make_repo(root, f"# t\n\n- [?] {self.item}\n- [ ] In `other.py`, later\n")
        self.runs = root / "runs"
        patcher = patch("forge.worktree.runs_root", return_value=self.runs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.branch = make_parked_branch(self.repo, self.runs, 1, self.item)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_and_summaries(self):
        self.assertIn(self.branch, index_run_records(self.repo))
        [summary] = list_branch_summaries(self.repo)
        self.assertEqual(summary["item_title"], "In `feature1.py`, add y")
        self.assertIn("1 file changed", summary["diffstat"])
        self.assertNotIn("diff", summary)

    def test_merge_deletes_branch_and_marks_item_done_despite_dirty_todo(self):
        # The runner leaves TODO.md modified; that must not block a merge.
        todo = self.repo / "TODO.md"
        todo.write_text(todo.read_text() + "\n")
        result = studio.merge_branch(self.repo, self.branch)
        self.assertTrue(result["item_marked_done"])
        self.assertEqual(git(self.repo, "branch", "--list", self.branch), "")
        self.assertTrue((self.repo / "feature1.py").exists())
        self.assertIn(f"- [x] {self.item.splitlines()[0]}", todo.read_text())
        self.assertIn("- [ ] In `other.py`, later", todo.read_text())

    def test_merge_refuses_dirty_tracked_file(self):
        (self.repo / "app.py").write_text("x = 2\n")
        with self.assertRaises(studio.ApiError) as ctx:
            studio.merge_branch(self.repo, self.branch)
        self.assertEqual(ctx.exception.status, 409)
        self.assertNotEqual(git(self.repo, "branch", "--list", self.branch), "")

    def test_mark_done_falls_back_to_first_line(self):
        todo = self.repo / "TODO.md"
        todo.write_text(todo.read_text().replace("  with detail", "  edited detail"))
        self.assertTrue(mark_record_item_done(self.repo, {"item": self.item}))
        self.assertFalse(mark_record_item_done(self.repo, {"item": "no such item"}))


class TestLlmFeatures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo = make_repo(root, "- [?] In `feature1.py`, add y\n")
        self.runs = root / "runs"
        patcher = patch("forge.worktree.runs_root", return_value=self.runs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.branch = make_parked_branch(self.repo, self.runs, 1, "In `feature1.py`, add y")

    def tearDown(self):
        self.tmp.cleanup()

    def test_critic_routes_bare_claude_to_cli_and_parses_verdict(self):
        with patch("forge.studio.run_llm", return_value="### 1. Verdict\n[REJECT - CRITICAL ISSUES]\n...") as llm:
            res = critique_diff(self.branch, self.repo, "claude")
        self.assertTrue(res["ok"])
        self.assertEqual(res["verdict"], "REJECT")
        self.assertEqual(llm.call_args[0][0], "claude")

    def test_run_llm_claude_passes_model_flag(self):
        done = subprocess.CompletedProcess([], 0, stdout="ok\n", stderr="")
        with patch("forge.studio.subprocess.run", return_value=done) as run:
            self.assertEqual(studio.run_llm("claude:claude-opus-5", "hi"), "ok")
        self.assertEqual(run.call_args[0][0], ["claude", "-p", "--model", "claude-opus-5", "hi"])

    def test_run_llm_surfaces_cli_error(self):
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="model not found")
        with patch("forge.studio.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "model not found"):
                studio.run_llm("claude", "hi")

    def test_draft_returns_lint_and_does_not_write(self):
        before = (self.repo / "TODO.md").read_text()
        with patch("forge.studio.run_llm", return_value="```markdown\n- [ ] In `a.py`, do a\n```"):
            res = draft_spec_goal("goal", self.repo, "claude")
        self.assertTrue(res["ok"])
        self.assertEqual(res["draft"], "- [ ] In `a.py`, do a")
        self.assertEqual(res["lint"]["total_items"], 1)
        self.assertEqual((self.repo / "TODO.md").read_text(), before)


class TestApiHardening(unittest.TestCase):
    PORT = 18911

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.repo = make_repo(root, "- [ ] In `a.py`, first\n- [?] In `b.py`, second\n")
        cls.runs = root / "runs"
        cls.patcher = patch("forge.worktree.runs_root", return_value=cls.runs)
        cls.patcher.start()
        cls.server = StudioServer(project_dir=cls.repo, port=cls.PORT)
        cls.server.start(in_background=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.patcher.stop()
        cls.tmp.cleanup()

    def request(self, path, body=None, raw=None, method=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(f"http://127.0.0.1:{self.PORT}{path}", data=data,
                                     headers={"Content-Type": "application/json"},
                                     method=method or ("POST" if data is not None else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def test_malformed_requests_get_json_errors(self):
        for raw, path in ((b"{not json", "/api/todo"), (b"[1,2]", "/api/todo"),
                          (b'{"line": "abc"}', "/api/todo/toggle"),
                          (b'{"content": 5}', "/api/todo"),
                          (b'{"line": 1, "status": "z"}', "/api/todo/toggle"),
                          (b'{"branch": "main"}', "/api/merge")):
            status, data = self.request(path, raw=raw)
            self.assertEqual(status, 400, (path, raw, data))
            self.assertFalse(data["ok"])
            self.assertTrue(data["error"])

    def test_unknown_routes_are_json_404(self):
        self.assertEqual(self.request("/api/nope")[0], 404)
        self.assertEqual(self.request("/static/../studio.py")[0], 404)

    def test_stale_save_conflicts_and_force_overwrites(self):
        details = get_todo_details(self.repo)
        stale = details["hash"]
        todo = self.repo / "TODO.md"
        original = todo.read_text()
        try:
            todo.write_text(original.replace("[ ] In `a.py`", "[x] In `a.py`"))  # runner marks item done
            status, data = self.request("/api/todo", {"content": original, "base_hash": stale})
            self.assertEqual(status, 409)
            self.assertTrue(data["conflict"])
            self.assertIn("[x] In `a.py`", todo.read_text())
            status, _ = self.request("/api/todo", {"content": original, "base_hash": stale, "force": True})
            self.assertEqual(status, 200)
            self.assertEqual(todo.read_text(), original)
        finally:
            todo.write_text(original)

    def test_run_conflicting_actions_refused_while_running(self):
        with patch.object(self.server.telemetry, "is_runner_active", return_value=True):
            for path, body in (("/api/todo", {"content": "x"}),
                               ("/api/todo/requeue-stuck", {}),
                               ("/api/merge", {"branch": "aider-loop/item-1"}),
                               ("/api/discard", {"branch": "aider-loop/item-1"}),
                               ("/api/project", {"project_dir": str(self.repo)})):
                status, data = self.request(path, body)
                self.assertEqual(status, 409, path)
                self.assertIn("Stop the runner", data["error"])

    def test_requeue_single_line(self):
        todo = self.repo / "TODO.md"
        original = todo.read_text()
        try:
            status, data = self.request("/api/todo/requeue-stuck", {"lines": [2]})
            self.assertEqual(status, 200)
            self.assertEqual(data["count"], 1)
            status, data = self.request("/api/todo/requeue-stuck", {"lines": "2"})
            self.assertEqual(status, 400)
        finally:
            todo.write_text(original)

    def test_static_and_index(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.PORT}/static/app.css")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertTrue(resp.headers["Content-Type"].startswith("text/css"))
        with urllib.request.urlopen(f"http://127.0.0.1:{self.PORT}/", timeout=5) as resp:
            page = resp.read().decode()
        self.assertIn('/static/app.js', page)
        self.assertNotIn("{{", page)


if __name__ == "__main__":
    unittest.main()
