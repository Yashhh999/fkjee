"""Multi-source adapter tests (no network: fixture rows only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fkjee.data import dedupe, norm_key  # noqa: E402
from fkjee.sources import SOURCES, _neet_item, split_numbered_options  # noqa: E402


def neet_row(**over):
    row = {
        "question_number": 7,
        "subject": "Biology",
        "question": "Which organelle is the powerhouse of the cell?",
        "options": ["Ribosome", "Mitochondria", "Golgi body", "Lysosome"],
        "correct_option": [1],
        "answer_status": "normal",
        "solution": "Mitochondria synthesise ATP by oxidative phosphorylation, hence the name.",
    }
    row.update(over)
    return row


class NumberedOptionsTest(unittest.TestCase):
    def test_split(self):
        q = "Find x.\n(1) 1\n(2) 2\n(3) 3\n(4) 4"
        stem, opts = split_numbered_options(q)
        self.assertEqual(stem, "Find x.")
        self.assertEqual(opts, ["1", "2", "3", "4"])

    def test_rejects_incomplete(self):
        self.assertIsNone(split_numbered_options("Find x.\n(1) 1\n(2) 2"))


class NeetAdapterTest(unittest.TestCase):
    def test_basic(self):
        item = _neet_item(neet_row(), "neet2025-code45", 2025)
        self.assertIsNotNone(item)
        self.assertEqual(item.gold, "B")
        self.assertEqual(item.subject, "biology")
        self.assertEqual(item.exam, "neet")
        self.assertEqual(item.year, 2025)
        self.assertIsNotNone(item.rationale)

    def test_figure_dependency_dropped(self):
        row = neet_row(question="Refer to the diagram <image_1> and name the part.")
        self.assertIsNone(_neet_item(row, "cfg", 2025))

    def test_dropped_question_status(self):
        self.assertIsNone(_neet_item(neet_row(answer_status="bonus"), "cfg", 2025))

    def test_multi_correct_rejected(self):
        self.assertIsNone(_neet_item(neet_row(correct_option=[0, 2]), "cfg", 2025))

    def test_zero_based_letter_mapping(self):
        self.assertEqual(_neet_item(neet_row(correct_option=[3]), "c", 2025).gold, "D")

    def test_solution_images_are_stripped(self):
        row = neet_row(solution="Use <image_2> the formula and substitute the given values carefully.")
        item = _neet_item(row, "c", 2025)
        self.assertNotIn("image_", item.rationale)


class RegistryTest(unittest.TestCase):
    def test_expected_sources(self):
        self.assertEqual(set(SOURCES), {"grafite", "neet", "pwjee", "ckjee"})
        for src in SOURCES.values():
            self.assertTrue(src.description)

    def test_repeated_exam_codes_collapse(self):
        """The same question across NEET codes must survive dedupe exactly once."""
        rows = [neet_row(question_number=i) for i in range(4)]
        items = [_neet_item(r, f"neet2025-code4{i}", 2025) for i, r in enumerate(rows)]
        self.assertEqual(len(dedupe([i for i in items if i])), 1)

    def test_norm_key_is_stable_for_whitespace_and_tags(self):
        self.assertEqual(norm_key("Find  <b>x</b>"), norm_key("Find x"))


if __name__ == "__main__":
    unittest.main()
