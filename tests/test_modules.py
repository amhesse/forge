"""Direct unit tests for the deconstructed modules: config, todo, validator, tdd, runner."""

import tempfile
import unittest
from pathlib import Path

from forge import config
from forge import runner
from forge import tdd
from forge import todo
from forge import validator


class TestConfigModule(unittest.TestCase):
    def test_cfg_lookup_with_defaults(self):
        saved = dict(config.CONFIG)
        try:
            config.CONFIG = {"model": {"author": "qwen25-coder-aider", "edit_format": "diff"}}
            self.assertEqual(config.cfg("model", "author"), "qwen25-coder-aider")
            self.assertEqual(config.cfg("model", "edit_format"), "diff")
            self.assertEqual(config.cfg("model", "missing", default="fallback"), "fallback")
            self.assertIsNone(config.cfg("nonexistent", "key"))
        finally:
            config.CONFIG = saved

    def test_resolve_models(self):
        class Args:
            models = "model-a, model-b"

        models = config.resolve_models(Args())
        self.assertEqual(models, ["model-a", "model-b"])

        class ArgsEmpty:
            models = None

        saved = dict(config.CONFIG)
        try:
            config.CONFIG = {"worker": {"models": ["m1", "m2"]}}
            self.assertEqual(config.resolve_models(ArgsEmpty()), ["m1", "m2"])
        finally:
            config.CONFIG = saved

    def test_load_config_priority(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            # Empty dir returns empty dict
            self.assertEqual(config.load_config(p), {})

            # .aiderloop.toml loaded
            (p / ".aiderloop.toml").write_text('[model]\nauthor = "test-aider"\n')
            self.assertEqual(config.load_config(p)["model"]["author"], "test-aider")

            # .forge.toml takes priority over .aiderloop.toml
            (p / ".forge.toml").write_text('[model]\nauthor = "test-forge"\n')
            self.assertEqual(config.load_config(p)["model"]["author"], "test-forge")


class TestReviewServer(unittest.TestCase):
    def test_colorize_diff(self):
        from forge import review_server
        raw_diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n-old_line\n+new_line\n normal"
        html = review_server.colorize_diff(raw_diff)
        self.assertIn('<span class="diff-add">+new_line</span>', html)
        self.assertIn('<span class="diff-del">-old_line</span>', html)
        self.assertIn('<span class="diff-hunk">@@ -1,2 +1,2 @@</span>', html)


class TestTodoModule(unittest.TestCase):
    def test_parse_and_render_todo(self):
        raw = [
            "- [ ] Task 1",
            "  continuation line",
            "- [x] Task 2",
        ]
        items = todo.parse_todo_lines(raw)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].status, todo.STATUS_OPEN)
        self.assertEqual(items[1].status, todo.STATUS_DONE)
        self.assertIn("continuation line", items[0].text)

        open_item = todo.next_open_item(items)
        self.assertIsNotNone(open_item)
        self.assertEqual(open_item.text.splitlines()[0], "Task 1")

    def test_update_item_status(self):
        with tempfile.TemporaryDirectory() as d:
            todo_file = Path(d) / "TODO.md"
            todo_file.write_text("- [ ] Initial task\n- [ ] Second task\n")
            raw_lines, items = todo.parse_todo(todo_file)
            todo.update_item_status(todo_file, items[0], todo.STATUS_DONE)
            content = todo_file.read_text()
            self.assertIn("- [x] Initial task", content)
            self.assertIn("- [ ] Second task", content)


class TestValidatorModule(unittest.TestCase):
    def test_strip_template_syntax(self):
        script = "var x = {{ foo | tojson }}; {% if bar %}doSomething();{% endif %} {# note #}"
        cleaned = validator.strip_template_syntax(script)
        self.assertIn("var x = null;", cleaned)
        self.assertNotIn("{% if", cleaned)
        self.assertNotIn("{#", cleaned)
        self.assertIn("doSomething();", cleaned)

    def test_suspicious_paths_detection(self):
        clean_paths = ["src/main.py", "tests/test_all.py"]
        bad_paths = ['src/File Listing: stories.py', 'foo<bar>.txt']
        self.assertEqual(validator.suspicious_new_paths(clean_paths), [])
        self.assertEqual(len(validator.suspicious_new_paths(bad_paths)), 2)

    def test_detect_prompt_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            clean_file = p / "clean.py"
            clean_file.write_text("def hello(): pass")
            leaked_file = p / "leaked.py"
            leaked_file.write_text("def run():\n    # ONLY EVER RETURN CODE IN A SEARCH/REPLACE BLOCK\n    pass")
            corrupted = validator.detect_prompt_leakage(p, ["clean.py", "leaked.py"])
            self.assertEqual(corrupted, ["leaked.py"])


class TestTddModule(unittest.TestCase):
    def test_tdd_classification(self):
        self.assertTrue(tdd.is_tdd_item("TDD: in `src/foo.py` and `tests/test_foo.py`"))
        self.assertFalse(tdd.is_tdd_item("In `src/foo.py`, add feature"))

        classified = tdd.classify_tdd_files(["src/math.py", "tests/test_math.py"])
        self.assertEqual(classified, ("tests/test_math.py", "src/math.py"))


class TestRunnerModule(unittest.TestCase):
    def test_difficulty_routing(self):
        exact = "Replace `config.py` with exactly:\n```python\nX = 1\n```"
        self.assertEqual(runner.estimate_difficulty(exact), runner.DIFFICULTY_EASY)
        prose = "In `src/engine.py`, add collision detection"
        self.assertEqual(runner.estimate_difficulty(prose), runner.DIFFICULTY_HARD)

    def test_parse_aider_token_line(self):
        output = "Tokens: 2.5k sent, 500 received.\nTokens: 1.0k sent, 200 received."
        usage = runner.parse_aider_token_line(output)
        self.assertEqual(usage["prompt_tokens"], 3500)
        self.assertEqual(usage["completion_tokens"], 700)


if __name__ == "__main__":
    unittest.main()
