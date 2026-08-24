import shutil
from pathlib import Path

import pytest

import turbine.cpu as cpu
from turbine.cpu import CpuHarnessError, parse_llama_bench_csv, render_chapter

CSV = (
    "build,sha,test,t,pp512,tg128,m\n"
    "4123,abcdef,pp+tg,4,320.5,5.1,qwen2.5-0.5b-q4_k_m.gguf\n"
    "4123,abcdef,pp+tg,8,610.2,9.7,qwen2.5-0.5b-q4_k_m.gguf\n"
)


def test_csv_parse_keeps_numeric_strings():
    rows = parse_llama_bench_csv(CSV)
    assert len(rows) == 2
    assert rows[1]["t"] == "8"
    assert float(rows[1]["tg128"]) == pytest.approx(9.7)


def test_empty_output_is_an_error_not_a_blank_chapter():
    with pytest.raises(CpuHarnessError):
        parse_llama_bench_csv("")


def test_missing_binary_names_the_fix():
    if shutil.which("llama-bench-not-installed"):
        pytest.skip("unexpectedly present")
    with pytest.raises(CpuHarnessError, match="llama-bench"):
        cpu.run_llama_bench("llama-bench-not-installed", Path("m.gguf"))


def test_mock_rows_are_labelled_synthetic_in_rendering():
    lines = render_chapter(cpu.mock_rows(), measured=False)
    assert any("Synthetic" in line for line in lines)
    assert any("Not measurements" in line for line in lines)
    assert sum(1 for line in lines if line.startswith("| synthetic")) == 2



def test_measured_render_has_no_synthetic_banner():
    lines = render_chapter(parse_llama_bench_csv(CSV), measured=True)
    assert not any("Synthetic" in line for line in lines)

