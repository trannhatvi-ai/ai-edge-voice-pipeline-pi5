import unittest

from scripts.prepare_models import model_paths


class PrepareModelsTests(unittest.TestCase):
    def test_model_paths_match_runtime_arguments(self):
        paths = model_paths("models")

        self.assertEqual(
            paths["asr_encoder"],
            "models/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx",
        )
        self.assertEqual(
            paths["asr_decoder"],
            "models/sherpa-onnx-whisper-tiny/tiny-decoder.onnx",
        )
        self.assertEqual(
            paths["tts_model"],
            "models/vits-piper-vi_VN-vivos-x_low/vi_VN-vivos-x_low.onnx",
        )


if __name__ == "__main__":
    unittest.main()
