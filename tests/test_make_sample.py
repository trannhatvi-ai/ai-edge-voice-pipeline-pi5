import unittest

from scripts.make_sample import fit_duration
from voice_edge.app import AudioChunk


class MakeSampleTests(unittest.TestCase):
    def test_fit_duration_pads_short_audio(self):
        audio = AudioChunk(b"\x01\x00" * 10, sample_rate=10)

        fitted = fit_duration(audio, 2.0)

        self.assertEqual(fitted.duration_seconds, 2.0)
        self.assertTrue(fitted.pcm_s16le.startswith(audio.pcm_s16le))

    def test_fit_duration_truncates_long_audio(self):
        audio = AudioChunk(b"\x01\x00" * 30, sample_rate=10)

        fitted = fit_duration(audio, 2.0)

        self.assertEqual(fitted.duration_seconds, 2.0)
        self.assertEqual(len(fitted.pcm_s16le), 40)


if __name__ == "__main__":
    unittest.main()
