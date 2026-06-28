import unittest

from voice_edge.app import configure_stdio


class FakeStream:
    def __init__(self):
        self.encoding = "cp1252"
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


class StdioTests(unittest.TestCase):
    def test_configure_stdio_sets_utf8_for_reconfigurable_streams(self):
        stream = FakeStream()

        configure_stdio(stream)

        self.assertEqual(stream.calls, [{"encoding": "utf-8"}])


if __name__ == "__main__":
    unittest.main()
