"""The CPU chapter: llama.cpp with a Q4 GGUF.

GGUF is first-class here and second-class in vLLM, where the reference run
measured roughly 93 tok/s and 958 ms TTFT for the same format. The chapter
answers a different question than the GPU grid: what tokens per second cost
on hardware you already own, with no hourly meter running.
"""

from __future__ import annotations

import csv
import io
import random
import shutil
import subprocess
from pathlib import Path


class CpuHarnessError(RuntimeError):
    pass


def parse_llama_bench_csv(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise CpuHarnessError("llama-bench produced no CSV rows")
    return rows


def run_llama_bench(
    binary: str,
    model: Path,
    *,
    pp: int = 512,
    tg: int = 128,
    threads: int | None = None,
    reps: int = 3,
    timeout_s: float = 3600.0,
) -> list[dict]:
    resolved = shutil.which(binary)
    if resolved is None:
        raise CpuHarnessError(
            f"{binary!r} not on PATH; build llama.cpp or point --binary at llama-bench"
        )
    args = [
        resolved,
        "-m",
        str(model),
        "-p",
        str(pp),
        "-n",
        str(tg),
        "-r",
        str(reps),
        "-o",
        "csv",
    ]
    if threads:
        args += ["-t", str(threads)]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=True)
    except subprocess.CalledProcessError as exc:
        raise CpuHarnessError(f"llama-bench failed: {exc.stderr[-400:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CpuHarnessError(f"llama-bench exceeded {timeout_s:g}s") from exc
    return parse_llama_bench_csv(proc.stdout)


def mock_rows(seed: int = 4) -> list[dict]:
    """Synthetic CPU rows in llama-bench's CSV shape.

    Same contract as MockBackend: shapes for plumbing verification,
    magnitudes that mean nothing.
    """
    rng = random.Random(seed)
    rows = []
    for threads in (4, 8):
        base_tg = 9.5 if threads == 8 else 5.2
        rows.append(
            {
                "model": "synthetic-q4_k_m.gguf",
                "t": str(threads),
                "pp512": f"{base_tg * rng.uniform(6.0, 7.0):.2f}",
                "tg128": f"{base_tg * rng.uniform(0.94, 1.06):.2f}",
            }
        )
    return rows


def render_chapter(rows: list[dict], *, measured: bool) -> list[str]:
    lines = [
        "| model | threads | pp512 tok/s | tg128 tok/s |",
        "|---|---|---|---|",
    ]
    for r in rows:
        label = r.get("model", "?")
        lines.append(
            f"| {label} | {r.get('t', '?')} | {r.get('pp512', 'not measured')} "
            f"| {r.get('tg128', 'not measured')} |"
        )
    if not measured:
        lines.insert(0, "Synthetic rows from the mock pipeline. Not measurements.")
    return lines
