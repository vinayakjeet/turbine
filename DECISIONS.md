# Decisions

Genuine tradeoffs with the alternatives considered, written when the choice
was made.

## Mock backend on a virtual clock before any GPU run

The brief asks for measurements on Modal's free credit. Every run there
costs real budget, so the pipeline had to be finished and trusted before
the first billed request. The mock preserves the full path: same load
generator, same statistics, same report gates. The one deliberate liberty
is a virtual clock: simulated latencies are advanced instantly and recorded
as-is, so archived synthetic numbers are byte-identical across machines.
Alternatives considered: sleeping in real time (minutes per sweep for no
extra fidelity, since the magnitudes are invented anyway) and jittered wall
clock (non-reproducible numbers that would eventually be mistaken for
hardware results). Rejected: labelling mock rows as GPU-shaped previews;
they are labelled SYNTHETIC at every boundary instead.

## One protocol shape copied from the vLLM blog

Input 1024, output 512, 300 requests, 5 warmups, rates {2, 8, inf}. The
alternative was designing a "better" workload with mixed lengths and bursty
arrivals. Rejected for this build: comparability to the published numbers
is worth more than realism here, and the threats-to-validity section says
plainly what the synthetic workload excludes. Mixed-length workloads remain
a backlog direction, not a silent substitution.

## Concurrency ceiling doubles as closed loop

A finite request rate is Poisson arrivals against a semaphore; an infinite
rate skips arrival gaps and the same semaphore becomes closed-loop
batching. Two regimes, one code path. The alternative, separate schedulers,
would have let timing semantics drift between them.

## Errors counted separately, never retried mid-window

A retry inside a measured window replaces the quantity being measured with
a luckier draw. ShipGate learned this as "errors are counted separately
from scores"; the same failure mode in a benchmark looks like a latency
improvement. Retries during warmup are fine because warmup samples never
enter statistics.

## Kernel recorded per quantized cell, enforced at render time

`--quantization awq` on vLLM dispatches Marlin by default, but a run can
land on the native kernel and lose an order of magnitude silently. The
report renderer refuses any quantized row whose kernel field is missing or
unresolved rather than footnoting it. Alternative considered: trusting the
launch flags. Rejected because the flag describes intent, not what
dispatched; the mapping table lives in `turbine/kernels.py` where it can be
checked against release notes.

## Spend ledger committed to the repo

An append-only JSONL at the root. The alternative is a private log or none
at all; both make the $30 cap unauditable, which defeats the point of
publishing cost numbers. Costs carry as decimal strings so no float drifts
between write and audit.

## Quality delta scored locally in ShipGate's dataset format

The eval set ships in ShipGate's JSONL shape so the same file gates through
ShipGate unchanged. Scoring exact-match locally keeps the quality chapter
runnable without a judge key; if a judged comparison is ever needed, the
dataset migrates without reformatting.

## Break-even driven by fixed costs, not averages

Scale-to-zero billing means a self-hosted GPU wins on every request it
serves and loses on cold starts and reserved idle. So the framework prices
fixed daily costs (cold starts, reserved hours) against the per-request API
premium and reports requests/day, with "always" and "never" as explicit
verdicts. An average-cost-per-token framing would have hidden exactly the
crossover the depth chapter exists to publish.
