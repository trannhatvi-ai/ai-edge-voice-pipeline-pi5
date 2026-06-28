import sys
import time
import unittest

from voice_edge.app import AudioChunk, PcmRecorder, VoicePipeline


class FakeAsr:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(audio)
        return "xin chao"


class FakeTts:
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return AudioChunk(b"\x01\x00\x02\x00", sample_rate=22050)


class FakeSpeaker:
    def __init__(self):
        self.played = []

    def play(self, audio):
        self.played.append(audio)


class FakeRecorder:
    def __init__(self, audio):
        self.audio = audio
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1
        return self.audio


class PipelineTests(unittest.TestCase):
    def test_pipeline_passes_pcm_chunk_without_file_path(self):
        asr = FakeAsr()
        tts = FakeTts()
        speaker = FakeSpeaker()
        audio = AudioChunk(b"\x00\x00" * 1600, sample_rate=16000)
        pipeline = VoicePipeline(FakeRecorder(audio), asr, tts, speaker)

        pipeline.start_recording()
        result = pipeline.stop_and_process()

        self.assertIs(asr.calls[0], audio)
        self.assertEqual(tts.calls, ["xin chao"])
        self.assertEqual(len(speaker.played), 1)
        self.assertEqual(result.text, "xin chao")

    def test_empty_audio_skips_asr_tts_and_speaker(self):
        asr = FakeAsr()
        tts = FakeTts()
        speaker = FakeSpeaker()
        pipeline = VoicePipeline(FakeRecorder(AudioChunk(b"")), asr, tts, speaker)

        pipeline.start_recording()
        result = pipeline.stop_and_process()

        self.assertEqual(asr.calls, [])
        self.assertEqual(tts.calls, [])
        self.assertEqual(speaker.played, [])
        self.assertEqual(result.text, "")

    def test_audio_chunk_validates_pcm_shape(self):
        with self.assertRaises(ValueError):
            AudioChunk(b"\x00", sample_width_bytes=2)
        with self.assertRaises(ValueError):
            AudioChunk(b"\x00\x00", sample_rate=0)

    def test_recorder_rejects_double_start(self):
        recorder = PcmRecorder(
            [
                sys.executable,
                "-c",
                "import sys,time; sys.stdout.buffer.write(b'\\0\\0'*800); sys.stdout.flush(); time.sleep(5)",
            ]
        )
        try:
            recorder.start()
            time.sleep(0.2)
            with self.assertRaises(RuntimeError):
                recorder.start()
        finally:
            audio = recorder.stop()

        self.assertGreater(len(audio.pcm_s16le), 0)


if __name__ == "__main__":
    unittest.main()
