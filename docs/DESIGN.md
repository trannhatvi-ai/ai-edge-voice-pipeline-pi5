# Design Answer

## Backend Choice

I choose `sherpa-onnx` for this Python implementation.

Reason: the recognizer and TTS engine are Python objects created once in `__init__`, so model load and ONNX Runtime session setup happen once at process startup. Each push-to-talk cycle only passes PCM samples through the already-loaded objects.

`whisper.cpp` is also a strong choice on Pi 5, especially with GGUF quantization, but a naive CLI call per button release would reload the model every time. That violates the warm-start requirement unless the app uses the C/C++ API or a persistent service wrapper.

## Quantization Choice

For this repo: **INT8 Whisper Tiny ONNX**.

Why INT8:

- Good CPU speedup on ARM compared with FP32/FP16-style execution.
- Lower RAM than FP16.
- Supported naturally by ONNX Runtime / sherpa-onnx deployment.
- Accuracy loss should be measured against the original model and must stay within the required 2% WER budget.

Why not the other options:

- FP16: useful on hardware with fast FP16 paths, but not the best CPU-only Pi 5 target for latency/RAM.
- Q8/Q5/Q4: these are better framed as whisper.cpp GGUF choices. If using whisper.cpp C++ instead, I would benchmark Q5 first and keep Q8 as the accuracy fallback.
- Q4: smallest and often fastest, but highest WER risk, so it is not my default under a 2% WER constraint.

## Thread Allocation

Default: `num_threads=2`.

Pi 5 has 4 Cortex-A76 cores, but `num_threads=4` is not always fastest. It can lose time to memory bandwidth pressure, cache contention, thread scheduling overhead, and thermal throttling. The production answer is to benchmark `1, 2, 3, 4` threads on the target Pi and keep the lowest stable RTF.

## Low-Latency I/O

ASR receives `AudioChunk.pcm_s16le` in memory. The pipeline does not create `temp.wav`.

If another process absolutely requires an audio file, write it under `/dev/shm` because it is tmpfs-backed RAM on Linux. That avoids MicroSD/SSD wear and gives near-instant read/write.

## Memory Management

`VoicePipeline` receives already-constructed ASR and TTS objects. In the real runtime path:

- `SherpaWhisperAsr.__init__` creates one `OfflineRecognizer`.
- `SherpaPiperTts.__init__` creates one `OfflineTts`.
- Every button cycle reuses those objects.

The smoke tests verify the pipeline passes an in-memory `AudioChunk` to ASR and does not pass a file path.

## KPI Verification

For a 5-second sample, run:

```bash
python -m scripts.pi5_kpi ... --wav samples/vi_5s.wav --iterations 30
```

Pass condition:

```text
max_rtf < 0.300
memory_pass == true
```

The KPI script writes:

- `benchmarks/pi5_kpi_report.json`
- `benchmarks/pi5_kpi_runs.csv`

The report includes the target machine information, every iteration's ASR/TTS time, RTF, RSS, and final pass/fail summary.

## Current Verification Status

Blocked on real target hardware and real model artifacts.

The development machine used to create this repo is Windows x64, not Raspberry Pi 5 ARM64. Running the benchmark on that machine would not validate ARM NEON use, Pi 5 memory bandwidth, CPU thermal behavior, or the assignment's target latency.

No quantized model files are committed in this repo. The intended runtime artifacts are:

- Whisper Tiny ONNX INT8 encoder
- Whisper Tiny ONNX INT8 decoder
- Whisper tokens file
- Piper/VITS Vietnamese ONNX model
- Piper/VITS tokens and optional `espeak-ng-data`

Until those files are placed under `models/` on a Raspberry Pi 5 and `scripts.pi5_kpi` passes, the honest status is **not yet proven to meet RTF or no-memory-leak KPIs**.
