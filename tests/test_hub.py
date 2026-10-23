"""Hub plumbing tests.  No network: only the disabled/override paths."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fkjee import hub  # noqa: E402


class HubTest(unittest.TestCase):
    def test_disabled_without_repo(self):
        self.assertFalse(hub.enabled({}))
        self.assertIsNone(hub.repo_id({}))

    def test_env_override(self):
        os.environ["FKJEE_HUB_REPO"] = "someone/models"
        try:
            self.assertTrue(hub.enabled({}))
            self.assertEqual(hub.repo_id({}), "someone/models")
            self.assertEqual(hub.repo_id({"hub_repo": "explicit/repo"}), "explicit/repo")
        finally:
            del os.environ["FKJEE_HUB_REPO"]

    def test_upload_is_noop_when_disabled(self):
        # must not raise, and must not touch the network
        self.assertFalse(hub.upload_dir({}, ".", "x"))
        self.assertEqual(hub.upload_files({}, {"a": "b"}), 0)


if __name__ == "__main__":
    unittest.main()
