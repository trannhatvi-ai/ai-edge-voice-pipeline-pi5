from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def configure_stdio(*streams) -> None:
    for stream in streams or (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class AudioChunk:
    pcm_s16le: bytes
    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 1:
            raise ValueError("only mono PCM is supported")
        if self.sample_width_bytes != 2:
            raise ValueError("only signed 16-bit PCM is supported")
        if len(self.pcm_s16le) % self.sample_width_bytes:
            raise ValueError("PCM byte length must align to sample width")

    @property
    def duration_seconds(self) -> float:
        if not self.pcm_s16le:
            return 0.0
        frame_bytes = self.channels * self.sample_width_bytes
        return len(self.pcm_s16le) / frame_bytes / self.sample_rate

    def as_float32(self):
        import numpy as np

        return np.frombuffer(self.pcm_s16le, dtype="<i2").astype("float32") / 32768.0


@dataclass(frozen=True)
class PipelineResult:
    text: str
    reply_audio: AudioChunk
    asr_seconds: float
    tts_seconds: float
    input_seconds: float

    @property
    def rtf(self) -> float:
        if self.input_seconds <= 0:
            return 0.0
        return (self.asr_seconds + self.tts_seconds) / self.input_seconds


class VoicePipeline:
    def __init__(self, recorder, asr, tts, speaker) -> None:
        self._recorder = recorder
        self._asr = asr
        self._tts = tts
        self._speaker = speaker
        self._recording = False

    def start_recording(self) -> None:
        if self._recording:
            raise RuntimeError("recording already started")
        self._recorder.start()
        self._recording = True

    def stop_and_process(self) -> PipelineResult:
        if not self._recording:
            raise RuntimeError("recording has not started")
        audio = self._recorder.stop()
        self._recording = False

        if not audio.pcm_s16le:
            empty = AudioChunk(b"", sample_rate=audio.sample_rate)
            return PipelineResult("", empty, 0.0, 0.0, 0.0)

        asr_started = time.perf_counter()
        text = self._asr.transcribe(audio).strip()
        asr_seconds = time.perf_counter() - asr_started

        if not text:
            empty = AudioChunk(b"", sample_rate=audio.sample_rate)
            return PipelineResult("", empty, asr_seconds, 0.0, audio.duration_seconds)

        tts_started = time.perf_counter()
        reply_audio = self._tts.synthesize(text)
        tts_seconds = time.perf_counter() - tts_started

        if reply_audio.pcm_s16le:
            self._speaker.play(reply_audio)

        return PipelineResult(
            text=text,
            reply_audio=reply_audio,
            asr_seconds=asr_seconds,
            tts_seconds=tts_seconds,
            input_seconds=audio.duration_seconds,
        )


class PcmRecorder:
    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.command = list(
            command
            or [
                "arecord",
                "-q",
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                str(channels),
                "-t",
                "raw",
            ]
        )
        self._proc: subprocess.Popen[bytes] | None = None
        self._chunks: list[bytes] = []
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("recorder already started")

        self._chunks = []
        self._proc = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def stop(self) -> AudioChunk:
        proc = self._proc
        if proc is None:
            raise RuntimeError("recorder has not started")

        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)

        if self._reader is not None:
            self._reader.join(timeout=2.0)

        if proc.stdout is not None:
            proc.stdout.close()

        self._proc = None
        self._reader = None
        return AudioChunk(
            b"".join(self._chunks),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def _read_stdout(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        while True:
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            self._chunks.append(chunk)


class AlsaSpeaker:
    def __init__(self, command: Sequence[str] | None = None) -> None:
        self.command = list(command) if command else None

    def play(self, audio: AudioChunk) -> None:
        command = self.command or [
            "aplay",
            "-q",
            "-f",
            "S16_LE",
            "-r",
            str(audio.sample_rate),
            "-c",
            str(audio.channels),
            "-t",
            "raw",
        ]
        result = subprocess.run(command, input=audio.pcm_s16le, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"speaker command failed with code {result.returncode}")


class SherpaWhisperAsr:
    def __init__(
        self,
        *,
        encoder: Path,
        decoder: Path,
        tokens: Path,
        num_threads: int = 2,
        language: str = "vi",
    ) -> None:
        import sherpa_onnx

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            num_threads=num_threads,
            language=language,
            task="transcribe",
            debug=False,
        )

    def transcribe(self, audio: AudioChunk) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(audio.sample_rate, audio.as_float32())
        self._recognizer.decode_streams([stream])
        return stream.result.text


class SherpaPiperTts:
    def __init__(
        self,
        *,
        model: Path,
        tokens: Path,
        data_dir: Path | None = None,
        lexicon: Path | None = None,
        rule_fsts: Path | None = None,
        num_threads: int = 2,
    ) -> None:
        import sherpa_onnx

        self._sherpa_onnx = sherpa_onnx
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model),
                    tokens=str(tokens),
                    data_dir=str(data_dir or ""),
                    lexicon=str(lexicon or ""),
                ),
                provider="cpu",
                num_threads=num_threads,
            ),
            rule_fsts=str(rule_fsts or ""),
            max_num_sentences=1,
        )
        if not config.validate():
            raise ValueError("invalid sherpa-onnx TTS config")
        self._tts = sherpa_onnx.OfflineTts(config)

    def synthesize(self, text: str) -> AudioChunk:
        import numpy as np

        config = self._sherpa_onnx.GenerationConfig()
        config.sid = 0
        config.speed = 1.0
        config.silence_scale = 0.2
        audio = self._tts.generate(text, config)
        samples = np.asarray(audio.samples, dtype="float32")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return AudioChunk(pcm, sample_rate=int(audio.sample_rate))


def read_wav_pcm(path: Path) -> AudioChunk:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("WAV must be mono")
        if wav.getsampwidth() != 2:
            raise ValueError("WAV must be signed 16-bit PCM")
        return AudioChunk(
            wav.readframes(wav.getnframes()),
            sample_rate=wav.getframerate(),
            channels=wav.getnchannels(),
            sample_width_bytes=wav.getsampwidth(),
        )


def write_wav_pcm(path: Path, audio: AudioChunk) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(audio.channels)
        wav.setsampwidth(audio.sample_width_bytes)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(audio.pcm_s16le)


def build_pipeline(args: argparse.Namespace, *, speaker=None) -> VoicePipeline:
    asr = SherpaWhisperAsr(
        encoder=args.asr_encoder,
        decoder=args.asr_decoder,
        tokens=args.asr_tokens,
        num_threads=args.num_threads,
        language=args.language,
    )
    tts = SherpaPiperTts(
        model=args.tts_model,
        tokens=args.tts_tokens,
        data_dir=args.tts_data_dir,
        lexicon=args.tts_lexicon,
        rule_fsts=args.tts_rule_fsts,
        num_threads=args.num_threads,
    )
    return VoicePipeline(
        PcmRecorder(sample_rate=args.sample_rate),
        asr,
        tts,
        speaker if speaker is not None else AlsaSpeaker(),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Pi 5 push-to-talk voice pipeline")
    parser.add_argument("--asr-encoder", type=Path, required=True)
    parser.add_argument("--asr-decoder", type=Path, required=True)
    parser.add_argument("--asr-tokens", type=Path, required=True)
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--tts-tokens", type=Path, required=True)
    parser.add_argument("--tts-data-dir", type=Path)
    parser.add_argument("--tts-lexicon", type=Path)
    parser.add_argument("--tts-rule-fsts", type=Path)
    parser.add_argument("--language", default="vi")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--wav", type=Path, help="Run one benchmark pass from a mono 16-bit PCM WAV")
    parser.add_argument("--no-play", action="store_true")
    return parser.parse_args(argv)


class NullSpeaker:
    def play(self, audio: AudioChunk) -> None:
        return None


class FixedRecorder:
    def __init__(self, audio: AudioChunk) -> None:
        self.audio = audio

    def start(self) -> None:
        return None

    def stop(self) -> AudioChunk:
        return self.audio


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv or sys.argv[1:])
    speaker = NullSpeaker() if args.no_play else None
    pipeline = build_pipeline(args, speaker=speaker)

    if args.wav:
        pipeline._recorder = FixedRecorder(read_wav_pcm(args.wav))
        pipeline.start_recording()
        result = pipeline.stop_and_process()
        print_result(result)
        return 0

    input("Press Enter to start recording...")
    pipeline.start_recording()
    input("Recording. Press Enter to stop and process...")
    result = pipeline.stop_and_process()
    print_result(result)
    return 0


def print_result(result: PipelineResult) -> None:
    print(f"Transcript: {result.text}")
    print(f"ASR seconds: {result.asr_seconds:.3f}")
    print(f"TTS seconds: {result.tts_seconds:.3f}")
    print(f"Input seconds: {result.input_seconds:.3f}")
    print(f"RTF: {result.rtf:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
