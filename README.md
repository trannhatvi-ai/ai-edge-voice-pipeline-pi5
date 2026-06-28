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

## Run

```bash
python -m voice_edge.app \
  --asr-encoder models/whisper-tiny/tiny-encoder.int8.onnx \
  --asr-decoder models/whisper-tiny/tiny-decoder.int8.onnx \
  --asr-tokens models/whisper-tiny/tiny-tokens.txt \
  --tts-model models/piper-vi/model.onnx \
  --tts-tokens models/piper-vi/tokens.txt \
  --tts-data-dir models/piper-vi/espeak-ng-data \
  --num-threads 2
```

Benchmark a fixed mono 16-bit PCM WAV without playing audio:

```bash
python -m voice_edge.app \
  --asr-encoder models/whisper-tiny/tiny-encoder.int8.onnx \
  --asr-decoder models/whisper-tiny/tiny-decoder.int8.onnx \
  --asr-tokens models/whisper-tiny/tiny-tokens.txt \
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
  --iterations 30 \
  --asr-encoder models/whisper-tiny/tiny-encoder.int8.onnx \
  --asr-decoder models/whisper-tiny/tiny-decoder.int8.onnx \
  --asr-tokens models/whisper-tiny/tiny-tokens.txt \
  --tts-model models/piper-vi/model.onnx \
  --tts-tokens models/piper-vi/tokens.txt \
  --tts-data-dir models/piper-vi/espeak-ng-data \
  --num-threads 2
```

The run is accepted only if:

- `max_rtf < 0.30`
- `memory_pass == true`
- `benchmarks/pi5_kpi_report.json` was generated on Raspberry Pi 5 ARM64, not a desktop/laptop.

## Submission Notes

The design answers are in `docs/DESIGN.md`. Current repository status is implementation-ready, not Pi-verified, until `benchmarks/pi5_kpi_report.json` is produced on the target Raspberry Pi 5.

Model binaries are intentionally ignored by git; put them under `models/` on the Pi. If the grader requires committed model artifacts, use Git LFS or a release asset because ONNX/GGUF model files are too large for a normal git repo.
