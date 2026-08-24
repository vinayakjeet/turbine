import json
from pathlib import Path

import pytest

from turbine.config import Cell, RunEnvironment, core_grid, prefix_grid
from turbine.report import (
    Unpublishable,
    load_runs,
    render_report,
    validate_publishable,
)


def make_run(
    tmp_path: Path,
    cell: Cell,
    *,
    reps=3,
    env=None,
    kernel=None,
    terminated=False,
) -> Path:
    environment = env or RunEnvironment.live(
        vllm_version="0.11.0",
        cuda_version="12.9",
        driver_version="580.00",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="abc1234",
        gpu_host="modal",
    )
    payload = {
        "cell": {
            "tier": cell.tier,
            "precision": cell.precision,
            "concurrency": cell.concurrency,
            "rate": cell.rate,
            "prefix_reuse": cell.prefix_reuse,
        },
        "kernel": kernel or ("none" if cell.precision == "fp16" else "marlin"),
        "environment": {
            "backend": environment.backend,
            "vllm_version": environment.vllm_version,
            "cuda_version": environment.cuda_version,
            "driver_version": environment.driver_version,
            "model_id": environment.model_id,
            "model_revision": environment.model_revision,
            "gpu_host": environment.gpu_host,
            "recorded_at": environment.recorded_at,
        },
        "terminated": terminated,
        "cost_usd": "0.0100",
        "command": "turbine cell --tier A10G",
        "reps": [
            {
                "rep": i,
                "duration_s": 10.0 + i,
                "requests": 300,
                "errors": 0,
                "output_tokens": 150000,
                "tok_per_s": 900.0 + i * 10,
                "ttft_s": {
                    "n": 300, "mean": 0.12, "stdev": 0.01,
                    "p50": 0.11 + i * 0.001, "p95": 0.2, "p99": 0.25,
                },
                "tpot_s": {
                    "n": 300, "mean": 0.008, "stdev": 0.0005,
                    "p50": 0.008, "p95": 0.009, "p99": 0.01,
                },
            }
            for i in range(reps)
        ],
    }
    path = tmp_path / f"{cell.id()}.json"
    path.write_text(json.dumps(payload))
    return path


def test_valid_run_passes_the_gate(tmp_path: Path):
    run_file = make_run(
        tmp_path,
        Cell(tier="A10G", precision="awq", concurrency=8, rate=float("inf")),
    )
    validate_publishable(load_runs([run_file])[0])


def test_under_three_reps_is_refused_not_averaged(tmp_path: Path):
    run_file = make_run(
        tmp_path, Cell(tier="T4", precision="fp16", concurrency=1, rate=2.0), reps=2
    )
    with pytest.raises(Unpublishable, match="3-run floor"):
        validate_publishable(load_runs([run_file])[0])


def test_unpinned_environment_is_refused(tmp_path: Path):
    cell = Cell(tier="T4", precision="fp16", concurrency=1, rate=8.0)
    run_file = make_run(tmp_path, cell, env=RunEnvironment.synthetic())
    # A run that claims a live backend but carries synthetic pins is exactly
    # the shape the gate exists for; synthetic() marks itself mock, so force
    # the dishonest combination.
    raw = json.loads(run_file.read_text())
    raw["environment"]["backend"] = "openai-compat"
    run_file.write_text(json.dumps(raw))
    with pytest.raises(Unpublishable, match="not fully pinned"):
        validate_publishable(load_runs([run_file])[0])


def test_kernelless_quantized_row_is_refused(tmp_path: Path):
    run_file = make_run(
        tmp_path,
        Cell(tier="A10G", precision="awq", concurrency=32, rate=float("inf")),
        kernel="none",
    )
    with pytest.raises(Unpublishable, match="kernel"):
        validate_publishable(load_runs([run_file])[0])


def test_terminated_cell_is_refused(tmp_path: Path):
    run_file = make_run(
        tmp_path, Cell(tier="T4", precision="fp16", concurrency=1, rate=8.0), terminated=True
    )
    with pytest.raises(Unpublishable, match="terminated"):
        validate_publishable(load_runs([run_file])[0])


def test_mock_report_carries_the_synthetic_banner(tmp_path: Path):
    make_run(
        tmp_path,
        Cell(tier="A10G", precision="awq", concurrency=8, rate=float("inf")),
        env=RunEnvironment.synthetic(),
    )
    text = render_report([tmp_path], grid=core_grid() + prefix_grid())
    assert "SYNTHETIC" in text.splitlines()[2]
    assert "marlin" in text
    assert "not measured" in text


def test_mixing_mock_and_measured_in_one_report_is_refused(tmp_path: Path):
    make_run(
        tmp_path,
        Cell(tier="A10G", precision="awq", concurrency=8, rate=float("inf")),
        env=RunEnvironment.synthetic(),
    )
    other = tmp_path / "live"
    other.mkdir()
    make_run(
        other,
        Cell(tier="T4", precision="fp16", concurrency=8, rate=float("inf")),
    )
    with pytest.raises(Unpublishable, match="[Mm]ixed"):
        render_report([tmp_path, other])


def test_empty_report_dir_is_refused(tmp_path: Path):
    with pytest.raises(Unpublishable, match="no result files"):
        render_report([tmp_path])
