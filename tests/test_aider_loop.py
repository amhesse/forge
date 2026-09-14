"""Tests for the parsing/classification logic in forge.aider_loop.

Every case here traces to a real bug found while running this tool
against real checklists this session, not a hypothetical. That's
deliberate: this project has spent all session finding out the hard way
that a local model will confidently write a self-contradictory test, so
its own test suite should hold itself to the standard it enforces on
everyone else - assertions grounded in something that actually happened,
not a guess at what might.
"""

import unittest

from forge import aider_loop as al


class TestNegationDetection(unittest.TestCase):
    """expected_files()'s negation handling. _negates_before() was added
    after 12 of 20 items in a real overnight run parked incorrectly: the
    loop required each commit to touch the very file its own item text
    said not to touch, because the negation phrase sat BEFORE the path
    and the original _negates() only ever looked after it.
    """

    def test_do_not_touch_immediately_before_path_at_end_of_item(self):
        text = ("In `tests/test_chunk.py`, add a test. Do not change any existing "
                "test in the file, and do not touch `arch/chunk.py`.")
        self.assertEqual(al.expected_files(text), ["tests/test_chunk.py"])

    def test_do_not_touch_before_path_followed_by_more_text(self):
        text = "In `tests/test_chunk.py`, add a test. Do not touch `arch/chunk.py`. Use unittest."
        self.assertEqual(al.expected_files(text), ["tests/test_chunk.py"])

    def test_documented_mid_sentence_case_still_works(self):
        # The case _negates() was originally written for - must not regress.
        text = "In `foo.py`, add X. Do not touch `bar.py`, which is unrelated."
        self.assertEqual(al.expected_files(text), ["foo.py"])

    def test_do_not_touch_any_other_file_does_not_negate_the_real_target(self):
        # "any other file" is boilerplate scoping, not a reference to a
        # specific path - it must never cancel the item's own target.
        text = "In `arch/index.py`, do the thing. Do not touch any other file."
        self.assertEqual(al.expected_files(text), ["arch/index.py"])

    def test_two_file_item_unaffected_by_negation_logic(self):
        text = "TDD: In `tests/test_chunk.py` and `arch/chunk.py`, implement split_lines."
        self.assertEqual(sorted(al.expected_files(text)),
                        ["arch/chunk.py", "tests/test_chunk.py"])

    def test_plain_item_with_do_not_change_any_other_function(self):
        text = "In `arch/store.py`, implement stats. Do not change any other function in the file."
        self.assertEqual(al.expected_files(text), ["arch/store.py"])


class TestCodeReferencesAreNotFiles(unittest.TestCase):
    """Backticked code shaped like a filename. Found by forge calibrate:
    a correct rename parked on every trial, and a TDD item was rejected
    before any model ran for naming three files."""

    def test_method_names_in_a_rename(self):
        text = ("Rename the method `Ledger.add` to `Ledger.record` everywhere. Update "
                "`ledger/store.py`, `ledger/cli.py` and `tests/test_basic.py`.")
        self.assertEqual(al.expected_files(text),
                         ["ledger/cli.py", "ledger/store.py", "tests/test_basic.py"])

    def test_dotted_module_path_in_tdd_item(self):
        text = ("TDD: implement `ledger/budget.py` so `tests/test_budget.py` passes: takes "
                "a `ledger.store.Ledger`.")
        self.assertEqual(al.expected_files(text), ["ledger/budget.py", "tests/test_budget.py"])
        self.assertEqual(al.estimate_difficulty(text), al.DIFFICULTY_HARD)

    def test_real_files_still_count(self):
        for name in ["README.md", "notes.story", ".aider.conf.yml", "app.test.js", "Main.java"]:
            self.assertEqual(al.expected_files(f"In `{name}`, do X."), [name], name)


class TestDifficultyRouting(unittest.TestCase):
    """estimate_difficulty()'s one grounded signal: a byte-exact item is
    "easy" because it's checked deterministically regardless of which
    model writes it, not because the content is simple."""

    def test_tdd_item_is_hard(self):
        text = "TDD: In `tests/test_x.py` and `x.py`, implement f."
        self.assertEqual(al.estimate_difficulty(text), al.DIFFICULTY_HARD)

    def test_multi_file_item_is_hard(self):
        text = "In `a.py` and `b.py`, keep them in sync."
        self.assertEqual(al.estimate_difficulty(text), al.DIFFICULTY_HARD)

    def test_byte_exact_single_file_item_is_easy(self):
        text = 'Replace `config.py` with exactly:\n```python\nX = 1\n```'
        self.assertEqual(al.estimate_difficulty(text), al.DIFFICULTY_EASY)

    def test_open_ended_prose_is_hard(self):
        text = "In `src/thing.py`, add a check for negative numbers."
        self.assertEqual(al.estimate_difficulty(text), al.DIFFICULTY_HARD)


class TestTddFilePairing(unittest.TestCase):
    """classify_tdd_files() must refuse to guess rather than misassign
    which file is the test and which is the implementation."""

    def test_pairs_test_prefixed_file_with_implementation(self):
        result = al.classify_tdd_files(["arch/chunk.py", "tests/test_chunk.py"])
        self.assertEqual(result, ("tests/test_chunk.py", "arch/chunk.py"))

    def test_pytest_style_suffix_also_recognized(self):
        result = al.classify_tdd_files(["thing.py", "thing_test.py"])
        self.assertEqual(result, ("thing_test.py", "thing.py"))

    def test_neither_file_test_shaped_returns_none(self):
        self.assertIsNone(al.classify_tdd_files(["a.py", "b.py"]))

    def test_both_files_test_shaped_returns_none(self):
        self.assertIsNone(al.classify_tdd_files(["tests/test_a.py", "tests/test_b.py"]))

    def test_wrong_file_count_returns_none(self):
        self.assertIsNone(al.classify_tdd_files(["a.py"]))
        self.assertIsNone(al.classify_tdd_files(["a.py", "b.py", "test_c.py"]))


class TestParseTodo(unittest.TestCase):
    """parse_todo_lines() must capture an item's full multi-line text,
    including fenced blocks - a real bug once silently dropped
    everything after a checkbox's first line, so a task that said
    'replace with exactly:' was handed nothing after the colon."""

    def test_multiline_item_with_fence_is_captured_whole(self):
        lines = [
            "- [ ] Replace `x.py` with exactly:",
            "```python",
            "value = 1",
            "```",
            "",
            "- [ ] Next item",
        ]
        items = al.parse_todo_lines(lines)
        self.assertEqual(len(items), 2)
        self.assertIn("```python", items[0].text)
        self.assertIn("value = 1", items[0].text)

    def test_checkpoint_comment_ends_an_item(self):
        lines = [
            "- [ ] First item",
            "<!-- a note between items -->",
            "- [ ] Second item",
        ]
        items = al.parse_todo_lines(lines)
        self.assertEqual(len(items), 2)
        self.assertNotIn("note between items", items[0].text)

    def test_render_round_trips_status_and_first_line(self):
        items = al.parse_todo_lines(["- [ ] Do the thing"])
        items[0].status = "x"
        self.assertEqual(items[0].render(), "- [x] Do the thing")


class TestExactContentSpecs(unittest.TestCase):
    """extract_exact_content_specs() only fires on the literal 'exactly'
    convention - the byte-exact check's whole reliability rests on not
    treating an ordinary fenced example as a spec to enforce."""

    def test_exactly_colon_before_fence_is_a_spec(self):
        text = 'Replace `a.py` with exactly:\n```python\nx = 1\n```'
        specs = al.extract_exact_content_specs(text)
        self.assertEqual(specs, {"a.py": "x = 1"})

    def test_fence_without_exactly_is_not_a_spec(self):
        text = 'For example, `a.py` might look like:\n```python\nx = 1\n```'
        self.assertEqual(al.extract_exact_content_specs(text), {})


if __name__ == "__main__":
    unittest.main()
