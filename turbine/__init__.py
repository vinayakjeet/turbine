"""Turbine: inference serving benchmarks on free-tier GPU credit.

TTFT, TPOT, throughput and tokens-per-second-per-dollar across hardware
tiers, precisions, prefix-cache states and load shapes, with a credit
guard that refuses to let a measurement outrun its budget.
"""

__version__ = "0.1.0"
