from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

from voice_edge.app import AudioChunk, SherpaWhisperAsr, read_wav_pcm


def normalize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹ]+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        curr = [i]
        for j, hyp_word in enumerate(hyp, 1):
            cost = 0 if ref_word == hyp_word else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1] / len(ref)


def load_transcripts(path: Path) -> dict[str, str]:
    transcripts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        filename, text = line.split(maxsplit=1)
        transcripts[filename] = text
    return transcripts


def transcribe(asr: SherpaWhisperAsr, audio: AudioChunk) -> tuple[str, float]:
    start = time.perf_counter()
    text = asr.transcribe(audio)
    return text, time.perf_counter() - start


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Whisper Tiny FP32 vs selected quantized ASR WER")
    parser.add_argument("--model-dir", type=Path, default=Path("models/sherpa-onnx-whisper-tiny"))
    parser.add_argument("--wav-dir", type=Path, default=Path("models/sherpa-onnx-whisper-tiny/test_wavs"))
    parser.add_argument("--transcripts", type=Path, default=Path("models/sherpa-onnx-whisper-tiny/test_wavs/trans.txt"))
    parser.add_argument("--quant-encoder", default="tiny-encoder.int8.onnx")
    parser.add_argument("--quant-decoder", default="tiny-decoder.onnx")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/asr_quant_report.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    fp32 = SherpaWhisperAsr(
        encoder=args.model_dir / "tiny-encoder.onnx",
        decoder=args.model_dir / "tiny-decoder.onnx",
        tokens=args.model_dir / "tiny-tokens.txt",
        num_threads=args.num_threads,
        language="",
    )
    quantized = SherpaWhisperAsr(
        encoder=args.model_dir / args.quant_encoder,
        decoder=args.model_dir / args.quant_decoder,
        tokens=args.model_dir / "tiny-tokens.txt",
        num_threads=args.num_threads,
        language="",
    )

    rows = []
    for filename, reference in load_transcripts(args.transcripts).items():
        audio = read_wav_pcm(args.wav_dir / filename)
        fp32_text, fp32_seconds = transcribe(fp32, audio)
        quant_text, quant_seconds = transcribe(quantized, audio)
        rows.append(
            {
                "file": filename,
                "reference": reference,
                "fp32_text": fp32_text,
                "quant_text": quant_text,
                "fp32_wer": wer(reference, fp32_text),
                "quant_wer": wer(reference, quant_text),
                "wer_delta": wer(reference, quant_text) - wer(reference, fp32_text),
                "fp32_seconds": fp32_seconds,
                "quant_seconds": quant_seconds,
            }
        )

    avg_delta = sum(row["wer_delta"] for row in rows) / len(rows)
    report = {
        "summary": {
            "files": len(rows),
            "avg_fp32_wer": sum(row["fp32_wer"] for row in rows) / len(rows),
            "avg_quant_wer": sum(row["quant_wer"] for row in rows) / len(rows),
            "avg_wer_delta": avg_delta,
            "wer_delta_pass": avg_delta <= 0.02,
            "quant_encoder": args.quant_encoder,
            "quant_decoder": args.quant_decoder,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.output}")
    return 0 if report["summary"]["wer_delta_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
