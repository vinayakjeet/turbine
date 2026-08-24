import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import status


def write_live_run(results: Path, tok_per_s: float, tier="A10G", precision="awq"):
    run_dir = results / "live"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell": {
            "tier": tier,
            "precision": precision,
            "concurrency": 8,
            "rate": float("inf"),
            "prefix_reuse": 0.0,
        },
        "kernel": "marlin" if precision != "fp16" else "none",
        "environment": {
            "backend": "openai-compat",
            "vllm_version": "0.11.0",
            "cuda_version": "12.9",
            "driver_version": "580.00",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "abc1234",
            "gpu_host": "modal",
            "recorded_at": "2026-08-24T10:00:00+00:00",
        },
        "terminated": False,
        "cost_usd": "0.0100",
        "command": "turbine cell",
        "reps": [
            {
                "rep": i,
                "tok_per_s": tok_per_s,
                "errors": 0,
                "requests": 300,
                "output_tokens": 150000,
                "duration_s": 100.0,
                "ttft_s": {"n": 1, "mean": 1, "stdev": 1, "p50": 1, "p95": 1, "p99": 1},
                "tpot_s": {"n": 1, "mean": 1, "stdev": 1, "p50": 1, "p95": 1, "p99": 1},
            }
            for i in range(3)
        ],
    }
    name = f"{tier}-{precision}-c8-rinf.json"
    (run_dir / name).write_text(json.dumps(payload))


def test_no_results_means_not_measured_and_exit_zero(tmp_path):
    payload = status.headline(tmp_path)
    assert payload["value"] == "not measured"
    assert payload["reason"]


def test_headline_picks_best_tok_per_dollar(tmp_path):
    write_live_run(tmp_path, 900.0, tier="T4", precision="fp16")
    write_live_run(tmp_path, 2000.0, tier="A10G", precision="awq")
    payload = status.headline(tmp_path)
    assert payload["value"] == round(2000.0 / 1.10, 1)
    assert payload["unit"] == "tok/s/$"
    assert payload["source"] == "scripts/status.py"
    assert payload["n_runs"] == 3
    assert payload["date"] == "2026-08-24"


def test_mock_or_unpinned_results_never_become_the_headline(tmp_path):
    run_dir = tmp_path / "live"
    run_dir.mkdir(parents=True)
    mock_payload = {
        "cell": {"tier": "A10G", "precision": "awq", "concurrency": 8,
                 "rate": float("inf"), "prefix_reuse": 0.0},
        "kernel": "marlin",
        "environment": {"backend": "mock", "recorded_at": "2026-08-24"},
        "terminated": False,
        "command": "",
        "reps": [{"tok_per_s": 99999.0, "errors": 0}],
    }
    (run_dir / "A10G-awq-c8-rinf.json").write_text(json.dumps(mock_payload))
    payload = status.headline(tmp_path)
    assert payload["value"] == "not measured"
