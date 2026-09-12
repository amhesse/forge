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

    def test_missing_end_marker_still_captures_the_content(self):
        # Was "returns empty" until a real long-form generation proved
        # that assumption wrong - see TestWrappingFenceStripping's
        # sibling tests below for the full story. A block with no
        # closing marker is exactly as likely to be genuinely complete
        # (the model just forgot the tag) as it is to be truncated, and
        # this project has no way to tell those apart from the text
        # alone - so it's read to the end of the response, which is
        # unambiguously correct when this IS the only/last block, and
        # produces the same content the model would have written between
        # a present END and nothing after it anyway.
        self.assertEqual(le.parse_file_blocks("===FILE: a.py===\nvalue = 1"), {"a.py": "value = 1"})

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

    def test_missing_end_marker_reads_to_end_of_string(self):
        # Real, not hypothetical: a long-form generation (a story, not a
        # config file) finished its actual content and simply never
        # emitted ===END=== - no natural "closing brace" cue the way code
        # has one. num_predict was nowhere near hit (no truncation
        # warning in the real run this regresses). Without this, a
        # complete, correct file was discarded as unparseable.
        raw = "===FILE: story.md===\n# Title\n\nOnce upon a time.\nThe end."
        self.assertEqual(le.parse_file_blocks(raw),
                        {"story.md": "# Title\n\nOnce upon a time.\nThe end."})

    def test_missing_end_marker_stops_at_the_next_file_block(self):
        raw = "===FILE: a.md===\nfirst content\n===FILE: b.md===\nsecond content\n===END==="
        self.assertEqual(le.parse_file_blocks(raw),
                        {"a.md": "first content", "b.md": "second content"})

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
