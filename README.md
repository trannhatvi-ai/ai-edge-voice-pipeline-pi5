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

## Submission Notes

The design answers are in `docs/DESIGN.md`. Model binaries are intentionally ignored by git; put them under `models/` on the Pi.
