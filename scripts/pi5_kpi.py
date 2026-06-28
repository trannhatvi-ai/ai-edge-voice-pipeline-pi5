from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from voice_edge.app import FixedRecorder, NullSpeaker, build_pipeline, read_wav_pcm


def parse_rss_kb(status_text: str) -> int | None:
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return None


def current_rss_kb() -> int | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    return parse_rss_kb(status.read_text(encoding="utf-8", errors="replace"))


def detect_target() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }
    model = Path("/proc/device-tree/model")
    if model.exists():
        info["device_model"] = model.read_text(errors="replace").strip("\x00\n ")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
        info["cpuinfo_model_name_lines"] = [
            line for line in text.splitlines() if "model name" in line.lower()
        ][:4]
    try:
        result = subprocess.run(
            ["free", "-h"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            info["free_h"] = result.stdout.strip()
    except FileNotFoundError:
        pass
    return info


def summarize_runs(
    runs: Sequence[dict[str, float | int | None]],
    *,
    rtf_limit: float,
    rss_growth_limit_kb: int,
) -> dict[str, object]:
    rtfs = [float(run["rtf"]) for run in runs]
    rss_values = [
        int(run["rss_kb"])
        for run in runs
        if run.get("rss_kb") is not None
    ]
    rss_growth = max(rss_values) - min(rss_values) if len(rss_values) >= 2 else None
    max_rtf = max(rtfs) if rtfs else 0.0
    avg_rtf = sum(rtfs) / len(rtfs) if rtfs else 0.0
    return {
        "runs": len(runs),
        "avg_rtf": avg_rtf,
        "max_rtf": max_rtf,
        "rtf_limit": rtf_limit,
        "rtf_pass": max_rtf < rtf_limit,
        "rss_growth_kb": rss_growth,
        "rss_growth_limit_kb": rss_growth_limit_kb,
        "memory_pass": rss_growth is not None and rss_growth <= rss_growth_limit_kb,
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pi 5 RTF/RAM KPI benchmark")
    parser.add_argument("--wav", type=Path, required=True, help="Mono 16-bit PCM WAV, ideally 5 seconds")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rtf-limit", type=float, default=0.30)
    parser.add_argument("--rss-growth-limit-kb", type=int, default=64 * 1024)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.iterations < 2:
        raise ValueError("--iterations must be >= 2 to measure memory growth")

    audio = read_wav_pcm(args.wav)
    pipeline = build_pipeline(args, speaker=NullSpeaker())
    pipeline._recorder = FixedRecorder(audio)

    runs = []
    for index in range(1, args.iterations + 1):
        pipeline.start_recording()
        result = pipeline.stop_and_process()
        row = {
            "iteration": index,
            "input_seconds": result.input_seconds,
            "asr_seconds": result.asr_seconds,
            "tts_seconds": result.tts_seconds,
            "rtf": result.rtf,
            "rss_kb": current_rss_kb(),
            "text": result.text,
        }
        runs.append(row)
        print(
            f"iter={index} rtf={result.rtf:.3f} "
            f"asr={result.asr_seconds:.3f}s tts={result.tts_seconds:.3f}s "
            f"rss_kb={row['rss_kb']}"
        )

    summary = summarize_runs(
        runs,
        rtf_limit=args.rtf_limit,
        rss_growth_limit_kb=args.rss_growth_limit_kb,
    )
    report = {
        "target": detect_target(),
        "summary": summary,
        "runs": runs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "pi5_kpi_report.json"
    csv_path = args.output_dir / "pi5_kpi_runs.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0 if summary["rtf_pass"] and summary["memory_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
