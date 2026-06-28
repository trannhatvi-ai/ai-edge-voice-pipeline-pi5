# AI Edge Voice Pipeline on Raspberry Pi 5

Offline push-to-talk voice-to-voice pipeline for Raspberry Pi 5 CPU ARM64.

Flow:

```text
button down -> record raw PCM in RAM -> button up -> ASR -> text -> TTS -> speaker
```

The implementation keeps ASR and TTS objects alive after startup. Each request passes raw PCM bytes in memory; it does not write `temp.wav` for ASR.

## Stack

- ASR: Whisper Tiny exported to ONNX and run with `sherpa-onnx`
- TTS: Piper/VITS model through `sherpa-onnx` offline TTS
- Audio I/O: Linux `arecord` and `aplay`
- Tests: Python stdlib `unittest`

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[runtime]"
```

On Raspberry Pi OS, install ALSA tools if missing:

```bash
sudo apt-get update
sudo apt-get install -y alsa-utils
```

## Prepare Real Models

Download the official sherpa-onnx Whisper Tiny and Vietnamese Piper/VITS artifacts:

```bash
python -m scripts.prepare_models
```

This creates:

- `models/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx`
- `models/sherpa-onnx-whisper-tiny/tiny-decoder.onnx`
- `models/sherpa-onnx-whisper-tiny/tiny-tokens.txt`
- `models/vits-piper-vi_VN-vivos-x_low/vi_VN-vivos-x_low.onnx`
- `models/vits-piper-vi_VN-vivos-x_low/tokens.txt`
- `models/vits-piper-vi_VN-vivos-x_low/espeak-ng-data`

The ASR default is intentionally mixed quantization: INT8 encoder + FP32 decoder. Full INT8 was measured and rejected because its WER delta exceeded the 2% requirement on the bundled test set.

Generate the fixed 5-second Vietnamese benchmark sample:

```bash
python -m scripts.make_sample \
  --tts-model models/vits-piper-vi_VN-vivos-x_low/vi_VN-vivos-x_low.onnx \
  --tts-tokens models/vits-piper-vi_VN-vivos-x_low/tokens.txt \
  --tts-data-dir models/vits-piper-vi_VN-vivos-x_low/espeak-ng-data \
  --seconds 5 \
  --out samples/vi_5s.wav
```

## Run

```bash
python -m voice_edge.app \
  --asr-encoder models/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx \
  --asr-decoder models/sherpa-onnx-whisper-tiny/tiny-decoder.onnx \
  --asr-tokens models/sherpa-onnx-whisper-tiny/tiny-tokens.txt \
  --tts-model models/piper-vi/model.onnx \
  --tts-tokens models/piper-vi/tokens.txt \
  --tts-data-dir models/piper-vi/espeak-ng-data \
  --num-threads 2
```

Benchmark a fixed mono 16-bit PCM WAV without playing audio:

```bash
python -m voice_edge.app \
  --asr-encoder models/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx \
  --asr-decoder models/sherpa-onnx-whisper-tiny/tiny-decoder.onnx \
  --asr-tokens models/sherpa-onnx-whisper-tiny/tiny-tokens.txt \
  --tts-model models/piper-vi/model.onnx \
  --tts-tokens models/piper-vi/tokens.txt \
  --tts-data-dir models/piper-vi/espeak-ng-data \
  --num-threads 2 \
  --wav samples/vi_5s.wav \
  --no-play
```

The output prints ASR time, TTS time, input duration, and RTF.

## Test

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
```

## KPI Verification on Raspberry Pi 5

Unit tests do not prove the assignment KPIs. Run the KPI script on the real Pi 5 with real quantized models:

```bash
python -m scripts.pi5_kpi \
  --wav samples/vi_5s.wav \
  --warmup-iterations 3 \
  --iterations 30 \
  --asr-encoder models/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx \
  --asr-decoder models/sherpa-onnx-whisper-tiny/tiny-decoder.onnx \
  --asr-tokens models/sherpa-onnx-whisper-tiny/tiny-tokens.txt \
  --tts-model models/piper-vi/model.onnx \
  --tts-tokens models/piper-vi/tokens.txt \
  --tts-data-dir models/piper-vi/espeak-ng-data \
  --num-threads 2
```

The run is accepted only if:

- `max_rtf < 0.30`
- `memory_pass == true`
- `benchmarks/pi5_kpi_report.json` was generated on Raspberry Pi 5 ARM64, not a desktop/laptop.

Quantization check:

```bash
python -m scripts.asr_quant_report
```

The current mixed quantization report passed locally on the official bundled ASR test WAVs with `avg_wer_delta = -0.00694`.

## Submission Notes

The design answers are in `docs/DESIGN.md`. Current repository status is implementation-ready, not Pi-verified, until `benchmarks/pi5_kpi_report.json` is produced on the target Raspberry Pi 5.

Model binaries are intentionally ignored by git; generate them with `scripts.prepare_models` on the Pi. If the grader requires committed model artifacts, use Git LFS or a release asset because ONNX/GGUF model files are too large for a normal git repo.
