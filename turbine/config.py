"""The measurement protocol, fixed before any result exists.

The serve-benchmark shape copies the vLLM blog protocol so numbers stay
comparable to theirs: synthetic prompts at input length 1024 and output
length 512, 300 measured requests behind 5 warmups, request rates
{2, 8, inf}, TPOT and p99 TTFT reported separately. Changing any constant
here changes what every archived result means, which is why they are
constants and not flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

INPUT_TOKENS = 1024
OUTPUT_TOKENS = 512
NUM_PROMPTS = 300
WARMUP_REQUESTS = 5
REQUEST_RATES = (2.0, 8.0, math.inf)
CONCURRENCIES = (1, 8, 32)
REQUEST_TIMEOUT_S = 120.0

MIN_RUNS = 3

PREFIX_REUSE_SWEEP = (1.0, 0.75, 0.5, 0.25, 0.0)
PREFIX_STUDY_CONCURRENCY = 8

TIERS = ("T4", "A10G", "A100-40G")
PRECISIONS = ("fp16", "awq")

DEFAULT_PLANNED_MINUTES = 10


def rate_label(rate: float) -> str:
    return "inf" if math.isinf(rate) else f"{rate:g}"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Cell:
    """One grid position: hardware, precision, load shape."""

    tier: str
    precision: str
    concurrency: int
    rate: float
    prefix_reuse: float = 0.0

    def id(self) -> str:
        reuse = "" if self.prefix_reuse == 0 else f"-p{self.prefix_reuse:g}"
        return (
            f"{self.tier}-{self.precision}-c{self.concurrency}"
            f"-r{rate_label(self.rate)}{reuse}"
        )

    def quantized(self) -> bool:
        return self.precision != "fp16"


def core_grid() -> list[Cell]:
    """T4 vs A10G, fp16 vs AWQ, three concurrencies, protocol rate sweep.

    The A100-40G rows are deliberately absent. The brief budgets one or two
    short A100 runs as a headline check, not a full sweep, so those are run
    by hand with `turbine cell`.
    """
    return [
        Cell(tier=t, precision=p, concurrency=c, rate=r)
        for t in ("T4", "A10G")
        for p in PRECISIONS
        for c in CONCURRENCIES
        for r in REQUEST_RATES
    ]


def prefix_grid() -> list[Cell]:
    """Reuse-rate sweep where prefix caching stops helping.

    Fixed concurrency on the mid tier; only the share of each prompt that
    repeats across requests moves.
    """
    return [
        Cell(
            tier="A10G",
            precision="awq",
            concurrency=PREFIX_STUDY_CONCURRENCY,
            rate=math.inf,
            prefix_reuse=r,
        )
        for r in PREFIX_REUSE_SWEEP
    ]


@dataclass(frozen=True)
class RunEnvironment:
    """What actually served the requests.

    Every field must be filled for a result to count as measured. Mock runs
    carry backend="mock" and are refused at any boundary between results and
    published numbers.
    """

    backend: str
    vllm_version: str
    cuda_version: str
    driver_version: str
    model_id: str
    model_revision: str
    gpu_host: str
    recorded_at: str

    @classmethod
    def synthetic(cls) -> RunEnvironment:
        return cls(
            backend="mock",
            vllm_version="synthetic",
            cuda_version="synthetic",
            driver_version="synthetic",
            model_id="synthetic",
            model_revision="synthetic",
            gpu_host="local",
            recorded_at=now_iso(),
        )

    @classmethod
    def live(
        cls,
        *,
        vllm_version: str,
        cuda_version: str,
        driver_version: str,
        model_id: str,
        model_revision: str,
        gpu_host: str,
    ) -> RunEnvironment:
        missing = [
            k
            for k, v in (
                ("vllm_version", vllm_version),
                ("cuda_version", cuda_version),
                ("driver_version", driver_version),
                ("model_id", model_id),
                ("model_revision", model_revision),
                ("gpu_host", gpu_host),
            )
            if not str(v).strip()
        ]
        if missing:
            raise ValueError(f"unpinned environment, missing: {', '.join(missing)}")
        return cls(
            backend="openai-compat",
            vllm_version=vllm_version,
            cuda_version=cuda_version,
            driver_version=driver_version,
            model_id=model_id,
            model_revision=model_revision,
            gpu_host=gpu_host,
            recorded_at=now_iso(),
        )

    def pinned(self) -> bool:
        return self.backend != "mock" and all(
            getattr(self, f) not in ("", "synthetic")
            for f in (
                "vllm_version",
                "cuda_version",
                "driver_version",
                "model_id",
                "model_revision",
            )
        )
