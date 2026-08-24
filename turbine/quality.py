"""Quantization quality delta on the ShipGate eval format.

The dataset ships in ShipGate's JSONL shape so the same file gates through
ShipGate unchanged. This module reproduces only exact-match scoring locally,
fp16 endpoint against quantized endpoint over MIN_RUNS repetitions each.

Expect a null result. The reference studies found all 4-bit methods within a
few percent of baseline on task metrics; a measured zero with error bars is
the finding, and most repos claim a quality cliff that is not there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from turbine.backends.base import Backend, CompletionError
from turbine.stats import mean, stdev

INSTRUCTION = (
    "Classify the customer message into one of: billing, technical, "
    "account, other. Reply with exactly one word.\n\nMessage: {prompt}"
)


@dataclass(frozen=True)
class EvalItem:
    id: str
    prompt: str
    expected: str


def load_dataset(path: Path) -> list[EvalItem]:
    items = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            items.append(
                EvalItem(
                    id=raw["id"],
                    prompt=raw["input"]["prompt"],
                    expected=raw["expected"],
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed eval item ({exc})") from exc
    if not items:
        raise ValueError(f"{path}: no eval items")
    return items


def normalize(text: str) -> str:
    return text.strip().lower()


async def score_arm(factory, items: list[EvalItem], *, reps: int) -> dict:
    """One arm, scored reps times. Errors are counted separately from wrong
    answers: a run full of transport failures must not read as a quality
    collapse."""
    scores = []
    errors = 0
    for rep in range(reps):
        backend: Backend = factory(rep)
        correct = 0
        for item in items:
            answer = ""
            try:
                generator = backend.stream(
                    INSTRUCTION.format(prompt=item.prompt),
                    max_tokens=8,
                    request_seed=rep * 1000 + stable_item(item.id),
                )
                try:
                    async for chunk in generator:
                        answer += chunk
                finally:
                    await generator.aclose()
            except CompletionError:
                errors += 1
                continue
            if normalize(answer) == normalize(item.expected):
                correct += 1
        await backend.aclose()
        scores.append(correct / len(items))
    return {
        "scores": [round(s, 4) for s in scores],
        "mean": round(mean(scores), 4),
        "stdev": round(stdev(scores), 4),
        "errors": errors,
        "n": len(items),
    }


def stable_item(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")


async def evaluate_quality(
    fp16_factory, quant_factory, dataset_path: Path, *, quant_label: str = "awq", reps: int = 3
) -> dict:
    items = load_dataset(dataset_path)
    fp16 = await score_arm(fp16_factory, items, reps=reps)
    quant = await score_arm(quant_factory, items, reps=reps)
    delta = round(quant["mean"] - fp16["mean"], 4)
    return {
        "dataset": str(dataset_path),
        "reps": reps,
        "rows": [
            {"arm": "fp16", **fp16},
            {"arm": quant_label, **quant},
        ],
        "delta": delta,
    }
