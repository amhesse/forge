"""Baseline: forge.eval imports before any checklist item lands, so the
gate is green from the start - see archaeologist's scaffold for why this
matters when items run concurrently and out of order.
"""

import unittest

from forge import eval as forge_eval


class TestScaffold(unittest.TestCase):
    def test_module_imports(self):
        self.assertIsNotNone(forge_eval)
        self.assertTrue(hasattr(forge_eval, "Scenario"))
        self.assertTrue(hasattr(forge_eval, "SCENARIOS"))


if __name__ == "__main__":
    unittest.main()
