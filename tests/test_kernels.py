import pytest

from turbine.kernels import (
    FP16_KERNEL,
    MissingKernel,
    UnknownQuantArg,
    require_publishable,
    resolve_kernel,
)


def test_fp16_needs_no_kernel():
    assert resolve_kernel("fp16", None) == FP16_KERNEL


def test_awq_defaults_to_the_marlin_path():
    # The default matters: the native AWQ kernel is the 10x slower path, so
    # an unflagged run must not silently look like a Marlin one.
    assert resolve_kernel("awq", None) == "marlin"
    assert resolve_kernel("awq", "awq_marlin") == "marlin"


def test_native_awq_is_reported_as_itself():
    assert resolve_kernel("awq", "awq") == "awq"


def test_unknown_quantization_arg_names_the_problem():
    with pytest.raises(UnknownQuantArg):
        resolve_kernel("awq", "bitsandbytes")


def test_publishable_gate_blocks_kernelless_quantized_rows():
    require_publishable("fp16", "none")
    require_publishable("awq", "marlin")
    with pytest.raises(MissingKernel):
        require_publishable("awq", "none")
