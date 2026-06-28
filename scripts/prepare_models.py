from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Sequence


ASR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2"
TTS_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-vi_VN-vivos-x_low.tar.bz2"


def model_paths(root: str | Path) -> dict[str, str]:
    root = Path(root).as_posix()
    return {
        "asr_encoder": f"{root}/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx",
        "asr_decoder": f"{root}/sherpa-onnx-whisper-tiny/tiny-decoder.onnx",
        "asr_tokens": f"{root}/sherpa-onnx-whisper-tiny/tiny-tokens.txt",
        "tts_model": f"{root}/vits-piper-vi_VN-vivos-x_low/vi_VN-vivos-x_low.onnx",
        "tts_tokens": f"{root}/vits-piper-vi_VN-vivos-x_low/tokens.txt",
        "tts_data_dir": f"{root}/vits-piper-vi_VN-vivos-x_low/espeak-ng-data",
    }


def download(url: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    print(f"download {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(out)


def extract(archive: Path, out_dir: Path) -> None:
    print(f"extract {archive}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(out_dir, filter="data")


def verify(paths: dict[str, str]) -> None:
    missing = [path for path in paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing model files:\n" + "\n".join(missing))


def print_runtime_command(paths: dict[str, str]) -> None:
    print(
        "\npython -m scripts.pi5_kpi "
        "--wav samples/vi_5s.wav "
        "--iterations 30 "
        f"--asr-encoder {paths['asr_encoder']} "
        f"--asr-decoder {paths['asr_decoder']} "
        f"--asr-tokens {paths['asr_tokens']} "
        f"--tts-model {paths['tts_model']} "
        f"--tts-tokens {paths['tts_tokens']} "
        f"--tts-data-dir {paths['tts_data_dir']} "
        "--num-threads 2"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real ASR/TTS models for the Pi 5 benchmark")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--download-dir", type=Path, default=Path("downloads"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    archives = [
        (ASR_URL, args.download_dir / "sherpa-onnx-whisper-tiny.tar.bz2"),
        (TTS_URL, args.download_dir / "vits-piper-vi_VN-vivos-x_low.tar.bz2"),
    ]
    for url, archive in archives:
        download(url, archive)
        extract(archive, args.model_dir)

    paths = model_paths(args.model_dir)
    verify(paths)
    print_runtime_command(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
