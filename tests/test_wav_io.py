from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_edge.app import AudioChunk, read_wav_pcm, write_wav_pcm


class WavIoTests(unittest.TestCase):
    def test_write_then_read_pcm_wav(self):
        audio = AudioChunk(b"\x00\x00\x01\x00" * 10, sample_rate=16000)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"

            write_wav_pcm(path, audio)
            loaded = read_wav_pcm(path)

        self.assertEqual(loaded, audio)


if __name__ == "__main__":
    unittest.main()
