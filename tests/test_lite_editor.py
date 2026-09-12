"""Tests for forge.lite_editor's output parsing.

The fence-stripping cases below are not hypothetical: the first real run
against a real model produced every file wrapped in a markdown fence
(a literal ```python as the first line), caught immediately by a
byte-exact item and one step later as a SyntaxError. These are the
regression tests for that.
"""

import unittest

from forge import lite_editor as le


class TestParseFileBlocks(unittest.TestCase):
    def test_single_file(self):
        raw = "===FILE: a.py===\nvalue = 1\n===END==="
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "value = 1"})

    def test_two_files_with_prose_between(self):
        raw = "===FILE: a.py===\nx\n===END===\nblah\n===FILE: b.py===\ny\n===END==="
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "x", "b.py": "y"})

    def test_prose_outside_blocks_is_ignored(self):
        raw = "Sure! I will do that.\n===FILE: a.py===\nz\n===END===\nHope that helps!"
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "z"})

    def test_no_block_at_all_returns_empty(self):
        self.assertEqual(le.parse_file_blocks("I think you should add a function."), {})

    def test_unterminated_block_returns_empty(self):
        self.assertEqual(le.parse_file_blocks("===FILE: a.py===\nvalue = 1"), {})

    def test_whitespace_in_markers_is_tolerated(self):
        raw = "===FILE:  a.py  ===\nq\n=== END ==="
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "q"})


class TestWrappingFenceStripping(unittest.TestCase):
    """Measured on the first real run: every file came back wrapped in a
    markdown fence even though the ===FILE:===/===END=== markers already
    delimit the content - models fence code out of habit."""

    def test_fenced_with_language_tag_is_stripped(self):
        raw = "===FILE: a.py===\n```python\nx = 1\n```\n===END==="
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "x = 1"})

    def test_fenced_without_language_tag_is_stripped(self):
        raw = "===FILE: a.py===\n```\nx = 1\n```\n===END==="
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "x = 1"})

    def test_unfenced_content_is_unaffected(self):
        raw = "===FILE: a.py===\nx = 1\n===END==="
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "x = 1"})

    def test_multiline_fenced_function_is_stripped(self):
        raw = "===FILE: a.py===\n```python\ndef f():\n    return 1\n```\n===END==="
        self.assertEqual(le.parse_file_blocks(raw), {"a.py": "def f():\n    return 1"})

    def test_markdown_file_keeps_its_own_internal_fences(self):
        # The one case this can't distinguish perfectly (see
        # strip_wrapping_fence's docstring): only an OUTERMOST fence
        # opening on line one and closing on the last line is removed,
        # so a real markdown file's own fenced examples must survive.
        raw = ("===FILE: r.md===\n# Title\n\n```python\ncode\n```\n\nmore\n===END===")
        self.assertEqual(le.parse_file_blocks(raw),
                        {"r.md": "# Title\n\n```python\ncode\n```\n\nmore"})


if __name__ == "__main__":
    unittest.main()
