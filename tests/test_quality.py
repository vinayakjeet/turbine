from pathlib import Path

import pytest

from turbine.backends.base import CompletionError
from turbine.quality import evaluate_quality, load_dataset


def scripted(answers_by_prompt_substring: dict[str, str], default: str = "other"):
    def make(rep: int):
        class Arm:
            name = "scripted"

            def stream(self, prompt, *, max_tokens, request_seed):
                answer = default
                for marker, label in answers_by_prompt_substring.items():
                    if marker in prompt:
                        answer = label
                        break

                async def gen():
                    yield answer + "\n"

                return gen()

            async def aclose(self):
                pass

        return Arm()

    return make


DATASET = Path("datasets/support-intent-quant.jsonl")


def test_dataset_loads_in_shipgate_shape_and_is_balanced():
    items = load_dataset(DATASET)
    assert len(items) >= 20
    from collections import Counter

    counts = Counter(i.expected for i in items)
    assert set(counts) == {"billing", "technical", "account", "other"}
    assert min(counts.values()) == max(counts.values())


def test_malformed_item_names_its_line(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "input": {}}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        load_dataset(bad)


async def test_exact_match_scoring_with_disagreement_between_arms(tmp_path: Path):
    fp16 = scripted({"charged twice": "billing", "502 errors": "technical"}, default="other")
    quant = scripted({"charged twice": "billing"}, default="other")

    result = await evaluate_quality(fp16, quant, DATASET, reps=3)
    rows = {r["arm"]: r for r in result["rows"]}
    assert rows["fp16"]["errors"] == 0
    assert rows["fp16"]["n"] == len(load_dataset(DATASET))
    # fp16 gets the 502 item right (technical); the quant arm answers `other`
    assert rows["fp16"]["mean"] > rows["awq"]["mean"]
    assert result["delta"] < 0


async def test_transport_errors_are_counted_separately_from_wrong_answers():
    class Broken:
        name = "broken"

        def stream(self, *a, **kw):
            async def gen():
                raise CompletionError("down")
                yield ""

            return gen()

        async def aclose(self):
            pass

    result = await evaluate_quality(lambda rep: Broken(), lambda rep: Broken(), DATASET, reps=2)
    items = len(load_dataset(DATASET))
    for row in result["rows"]:
        assert row["errors"] == items * result["reps"]
        assert all(s == 0.0 for s in row["scores"])
