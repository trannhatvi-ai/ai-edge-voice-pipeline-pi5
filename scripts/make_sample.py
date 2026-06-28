from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from voice_edge.app import AudioChunk, SherpaPiperTts, write_wav_pcm


DEFAULT_TEXT = "Xin chào, đây là bài kiểm tra giọng nói tiếng Việt trên thiết bị biên."


def fit_duration(audio: AudioChunk, seconds: float) -> AudioChunk:
    frame_bytes = audio.channels * audio.sample_width_bytes
    target_bytes = int(seconds * audio.sample_rate) * frame_bytes
    pcm = audio.pcm_s16le[:target_bytes]
    if len(pcm) < target_bytes:
        pcm += b"\x00" * (target_bytes - len(pcm))
    return AudioChunk(
        pcm,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        sample_width_bytes=audio.sample_width_bytes,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fixed-duration Vietnamese WAV sample")
    parser.add_argument("--tts-model", type=Path, required=True)
    parser.add_argument("--tts-tokens", type=Path, required=True)
    parser.add_argument("--tts-data-dir", type=Path)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("samples/vi_5s.wav"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    tts = SherpaPiperTts(
        model=args.tts_model,
        tokens=args.tts_tokens,
        data_dir=args.tts_data_dir,
        num_threads=args.num_threads,
    )
    audio = fit_duration(tts.synthesize(args.text), args.seconds)
    write_wav_pcm(args.out, audio)
    print(f"Wrote {args.out} ({audio.duration_seconds:.3f}s, {audio.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
