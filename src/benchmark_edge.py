"""
Edge inference benchmark (ONNX Runtime).

Screening devices for community camps and rural clinics do not run on an A100.
They run on a Raspberry Pi in a room with unreliable mains power. A model that
is accurate but takes four seconds per frame on the target hardware is not a
screening tool.

Almost no published imaging repository reports latency on the deployment target.
Reporting it is cheap, and it is the difference between a research artefact and
something a product team can cost out.

Usage:
    python -m src.benchmark_edge --onnx artifacts/model.onnx --runs 200
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def _hardware_summary() -> dict:
    """Best-effort description of what we are actually running on."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    try:  # Pi and most Linux SBCs
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Model"):
                    info["board"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def benchmark(
    onnx_path: str,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    runs: int = 200,
    warmup: int = 20,
    threads: int = 4,
) -> dict:
    """Time single-image inference. Reports the tail, not just the mean.

    p95 is the number that matters for a clinician standing over a patient. A
    good mean with a heavy tail still feels broken in the room.
    """
    model_path = Path(onnx_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if runs <= 0 or warmup < 0 or threads <= 0:
        raise ValueError("runs and threads must be positive; warmup cannot be negative")

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(
        str(model_path),
        opts,
        providers=["CPUExecutionProvider"],
    )
    name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    x = rng.standard_normal(input_shape).astype(np.float32)

    for _ in range(warmup):  # let the allocator settle
        sess.run(None, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "hardware": _hardware_summary(),
        "threads": threads,
        "runs": runs,
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(float(np.percentile(times, 95, method="nearest")), 2),
        "p99_ms": round(float(np.percentile(times, 99, method="nearest")), 2),
        "min_ms": round(times[0], 2),
        "max_ms": round(times[-1], 2),
        "throughput_fps": round(1000.0 / statistics.mean(times), 2),
    }


def format_benchmark(r: dict) -> str:
    hw = r["hardware"]
    board = hw.get("board", hw["machine"])
    return (
        f"Device      {board}\n"
        f"Threads     {r['threads']}   Runs  {r['runs']}\n"
        f"Latency     mean {r['mean_ms']} ms | median {r['median_ms']} ms | "
        f"p95 {r['p95_ms']} ms | p99 {r['p99_ms']} ms\n"
        f"Throughput  {r['throughput_fps']} img/s"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/model.onnx")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    print(format_benchmark(benchmark(args.onnx, runs=args.runs, threads=args.threads)))
