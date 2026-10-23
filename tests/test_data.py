"""Data pipeline tests.  Run: PYTHONPATH=src python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fkjee.data import (  # noqa: E402
    Item,
    clean_text,
    dedupe,
    norm_key,
    parse_grafite_row,
    parse_jeebench,
    split_inline_options,
    split_items,
)


def grafite_row(**over):
    row = {
        "question_id": "q1",
        "subject": "physics",
        "chapter": "optics",
        "topic": "lenses",
        "question": "A lens has focal length <b>10</b> cm. Find the image distance.",
        "options": json.dumps(
            [
                {"identifier": "A", "content": "10 cm"},
                {"identifier": "B", "content": "20 cm"},
                {"identifier": "C", "content": "30 cm"},
                {"identifier": "D", "content": "40 cm"},
            ]
        ),
        "correct_option": '["B"]',
        "answer": None,
        "explanation": "Using the thin lens equation 1/f = 1/v - 1/u with u = -20 cm "
                       "gives v = 20 cm, so the image forms 20 cm from the lens.",
        "question_type": "mcq",
        "paper_id": "jee-main-2021-online-1st-shift",
    }
    row.update(over)
    return row


class CleanTextTest(unittest.TestCase):
    def test_subscript_and_tags(self):
        self.assertEqual(clean_text("H<sub>2</sub>SO<sub>4</sub>"), "H2SO4")
        self.assertEqual(clean_text("<b>bold</b> and <i>it</i>"), "bold and it")

    def test_superscript_and_br(self):
        self.assertIn("10^-34", clean_text("6.6 x 10<sup>-34</sup>"))
        self.assertIn("\n", clean_text("line one<br>line two"))

    def test_images_dropped(self):
        out = clean_text('see <img src="x.png"> here')
        self.assertNotIn("img", out)
        self.assertIn("here", out)

    def test_entities(self):
        self.assertEqual(clean_text("a &amp; b"), "a & b")


class ParseGrafiteTest(unittest.TestCase):
    def test_mcq(self):
        item = parse_grafite_row(grafite_row())
        self.assertIsNotNone(item)
        self.assertEqual(item.gold, "B")
        self.assertEqual(item.kind, "mcq")
        self.assertEqual(len(item.options), 4)
        self.assertEqual(item.year, 2021)
        self.assertEqual(item.subject, "physics")
        self.assertIsNotNone(item.rationale)

    def test_integer(self):
        item = parse_grafite_row(
            grafite_row(
                question_type="integer",
                options="[]",
                correct_option="",
                answer="42",
                explanation="The required integer value is simply 42 as computed above.",
            )
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.gold, "42")
        self.assertEqual(item.kind, "integer")

    def test_rejects_multi_correct(self):
        self.assertIsNone(parse_grafite_row(grafite_row(correct_option='["A", "B"]')))

    def test_rejects_bad_option_count(self):
        opts = json.dumps([{"identifier": "A", "content": "x"}])
        self.assertIsNone(parse_grafite_row(grafite_row(options=opts)))

    def test_rejects_empty_option_text(self):
        opts = json.dumps(
            [{"identifier": "A", "content": "x"}, {"identifier": "B", "content": " "},
             {"identifier": "C", "content": "y"}, {"identifier": "D", "content": "z"}]
        )
        self.assertIsNone(parse_grafite_row(grafite_row(options=opts)))

    def test_image_only_explanation_is_not_a_rationale(self):
        item = parse_grafite_row(grafite_row(explanation='<img src="a.png">'))
        self.assertIsNotNone(item)
        self.assertIsNone(item.rationale)

    def test_restated_answer_is_not_a_rationale(self):
        item = parse_grafite_row(grafite_row(explanation="20 cm"))
        self.assertIsNone(item.rationale)


class SplitTest(unittest.TestCase):
    def _items(self):
        out = []
        for year in (2022, 2023, 2024):
            for i in range(20):
                out.append(
                    Item(
                        id=f"{year}-{i}", subject="math", question=f"q{year}{i}",
                        options=["a", "b", "c", "d"], gold="A", kind="mcq",
                        year=year, paper=f"jee-main-{year}-shift{i % 3}",
                    )
                )
        return out

    def test_holdout_by_year(self):
        train, dev, test = split_items(self._items(), holdout=2024)
        self.assertTrue(all(i.year == 2024 for i in test))
        self.assertTrue(all(i.year < 2024 for i in train + dev))
        self.assertEqual(len(test), 20)
        self.assertEqual(len(train) + len(dev), 40)

    def test_paper_level_disjoint(self):
        train, dev, test = split_items(self._items(), holdout=2024)
        self.assertFalse({i.paper for i in train} & {i.paper for i in dev})

    def test_deterministic(self):
        a = split_items(self._items())[0]
        b = split_items(self._items())[0]
        self.assertEqual([i.id for i in a], [i.id for i in b])


class DedupeTest(unittest.TestCase):
    def test_prefers_version_with_rationale(self):
        plain = Item("a", "math", "Find x <b>2</b>", ["1", "2", "3", "4"], "B", "mcq")
        rich = Item("b", "math", "Find x 2", ["1", "2", "3", "4"], "B", "mcq",
                    rationale="Because x equals two.")
        kept = dedupe([plain, rich])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].rationale, "Because x equals two.")

    def test_distinct_questions_survive(self):
        a = Item("a", "math", "Find x", ["1", "2", "3", "4"], "B", "mcq")
        b = Item("b", "math", "Find y", ["1", "2", "3", "4"], "B", "mcq")
        self.assertEqual(len(dedupe([a, b])), 2)


class JeeBenchTest(unittest.TestCase):
    def test_inline_options(self):
        q = "What is 2+2?\n\n(A) 3\n(B) 4\n(C) 5\n(D) 6"
        stem, opts = split_inline_options(q)
        self.assertEqual(stem, "What is 2+2?")
        self.assertEqual(opts, ["3", "4", "5", "6"])

    def test_only_single_mcq_kept(self):
        rows = [
            {"type": "MCQ", "gold": "B", "subject": "phy", "description": "JEE Adv 2016",
             "question": "Q?\n(A) 1\n(B) 2\n(C) 3\n(D) 4"},
            {"type": "MCQ(multiple)", "gold": "A,B", "subject": "phy", "description": "x",
             "question": "Q?\n(A) 1\n(B) 2\n(C) 3\n(D) 4"},
            {"type": "Numeric", "gold": "5", "subject": "phy", "description": "x",
             "question": "Q?"},
        ]
        items = parse_jeebench(rows)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].gold, "B")
        self.assertEqual(items[0].subject, "physics")


if __name__ == "__main__":
    unittest.main()
