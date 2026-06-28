import unittest

from scripts.asr_quant_report import wer


class AsrQuantReportTests(unittest.TestCase):
    def test_wer_counts_substitution_insertion_and_deletion(self):
        self.assertEqual(wer("a b c", "a x c y"), 2 / 3)

    def test_wer_is_zero_for_equal_text_ignoring_case_and_punctuation(self):
        self.assertEqual(wer("Xin chao!", "xin chao"), 0.0)


if __name__ == "__main__":
    unittest.main()
