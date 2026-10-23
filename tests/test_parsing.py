"""Answer-extraction and scoring tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fkjee.data import Item  # noqa: E402
from fkjee.metrics import is_correct, summarize  # noqa: E402
from fkjee.parsing import as_number, normalise_letter, parse_answer  # noqa: E402
from fkjee.prompts import build_messages, render_user, target_text  # noqa: E402


class ParseAnswerTest(unittest.TestCase):
    def test_boxed_letter(self):
        self.assertEqual(parse_answer("so x=2\n\\boxed{C}"), ("C", "boxed"))

    def test_last_boxed_wins(self):
        text = "wrong \\boxed{A} ... actually \\boxed{D}"
        self.assertEqual(parse_answer(text)[0], "D")

    def test_boxed_with_parens(self):
        self.assertEqual(parse_answer("\\boxed{ (B) }")[0], "B")

    def test_boxed_number(self):
        self.assertEqual(parse_answer("\\boxed{42}")[0], "42")

    def test_boxed_fraction(self):
        self.assertEqual(parse_answer("\\boxed{\\frac{1}{2}}")[0], "0.5")

    def test_scientific(self):
        self.assertEqual(as_number("6.4 \\times 10^{-34}"), 6.4e-34)

    def test_phrase_fallback(self):
        self.assertEqual(parse_answer("The correct option is (C).")[0], "C")

    def test_no_answer(self):
        self.assertEqual(parse_answer("I am not sure."), (None, "none"))

    def test_hallucinated_prose_does_not_parse_as_letter(self):
        _val, method = parse_answer("This problem requires more information.")
        self.assertIn(method, {"none", "pattern"})

    def test_letter_normalisation(self):
        self.assertEqual(normalise_letter("(d)"), "D")
        self.assertEqual(normalise_letter("Option B"), "B")
        self.assertIsNone(normalise_letter("E"))


class CorrectnessTest(unittest.TestCase):
    def test_letters(self):
        self.assertTrue(is_correct("C", "C"))
        self.assertFalse(is_correct("C", "D"))

    def test_numbers_with_tolerance(self):
        self.assertTrue(is_correct("0.5", "0.50000"))
        self.assertFalse(is_correct("0.5", "0.6"))

    def test_none_is_wrong(self):
        self.assertFalse(is_correct(None, "A"))

    def test_summary_accounts_for_invalid(self):
        recs = [
            {"gold": "A", "pred": "A", "subject": "math", "kind": "mcq", "year": 2024},
            {"gold": "B", "pred": None, "subject": "math", "kind": "mcq", "year": 2024},
            {"gold": "C", "pred": "C", "subject": "physics", "kind": "mcq", "year": 2024},
            {"gold": "1", "pred": "1", "subject": "physics", "kind": "integer", "year": 2024},
        ]
        s = summarize(recs)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["accuracy"], 0.75)
        self.assertAlmostEqual(s["parsed_rate"], 0.75)
        self.assertEqual(s["by_subject"]["math"]["accuracy"], 0.5)
        self.assertEqual(s["by_kind"]["integer"]["accuracy"], 1.0)


class PromptTest(unittest.TestCase):
    def test_mcq_prompt_has_all_options(self):
        item = Item("x", "math", "What is 1+1?", ["1", "2", "3", "4"], "B", "mcq")
        text = render_user(item)
        for letter in "ABCD":
            self.assertIn(f"({letter})", text)
        self.assertIn("What is 1+1?", text)

    def test_target_uses_boxed_gold(self):
        item = Item("x", "math", "q", ["1", "2", "3", "4"], "B", "mcq",
                    rationale="Cause two.")
        self.assertTrue(target_text(item).endswith("\\boxed{B}."))
        self.assertIn("Cause two.", target_text(item))

    def test_no_rationale_target(self):
        item = Item("x", "math", "q", ["1", "2", "3", "4"], "B", "mcq")
        self.assertEqual(target_text(item), "The correct option is \\boxed{B}.")

    def test_messages_shape(self):
        item = Item("x", "math", "q", ["1", "2", "3", "4"], "B", "mcq")
        msgs = build_messages(item)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
