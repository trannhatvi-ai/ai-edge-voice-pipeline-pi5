import unittest

from scripts.pi5_kpi import parse_rss_kb, summarize_runs


class KpiTests(unittest.TestCase):
    def test_parse_rss_kb_from_linux_status(self):
        status = "Name:\tpython\nVmRSS:\t  123456 kB\nThreads:\t4\n"

        self.assertEqual(parse_rss_kb(status), 123456)

    def test_summarize_runs_flags_rtf_over_limit(self):
        summary = summarize_runs(
            [
                {"rtf": 0.20, "rss_kb": 1000},
                {"rtf": 0.31, "rss_kb": 1005},
            ],
            rtf_limit=0.30,
            rss_growth_limit_kb=64 * 1024,
        )

        self.assertFalse(summary["rtf_pass"])
        self.assertEqual(summary["max_rtf"], 0.31)

    def test_summarize_runs_flags_memory_growth(self):
        summary = summarize_runs(
            [
                {"rtf": 0.20, "rss_kb": 1000},
                {"rtf": 0.20, "rss_kb": 70000},
            ],
            rtf_limit=0.30,
            rss_growth_limit_kb=64 * 1024,
        )

        self.assertFalse(summary["memory_pass"])
        self.assertEqual(summary["rss_growth_kb"], 69000)

    def test_summarize_runs_ignores_warmup_rows(self):
        summary = summarize_runs(
            [
                {"rtf": 0.20, "rss_kb": 1000, "phase": "warmup"},
                {"rtf": 0.20, "rss_kb": 70000, "phase": "measure"},
                {"rtf": 0.20, "rss_kb": 70004, "phase": "measure"},
            ],
            rtf_limit=0.30,
            rss_growth_limit_kb=64 * 1024,
        )

        self.assertTrue(summary["memory_pass"])
        self.assertEqual(summary["rss_growth_kb"], 4)


if __name__ == "__main__":
    unittest.main()
