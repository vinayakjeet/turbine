"""Kernel reporting.

The same AWQ weights served through different kernels differ by an order of
magnitude, roughly 10x for the Marlin path over the native one in the
reference studies. A quantized number without its kernel is not
reproducible, so this module refuses to let one into a table.
"""

from __future__ import annotations

FP16_KERNEL = "none"

# vLLM --quantization values mapped to the kernel that actually dispatches.
QUANT_ARG_TO_KERNEL = {
    "awq_marlin": "marlin",
    "gptq_marlin": "marlin",
    "marlin": "marlin",
    "machete": "machete",
    "exllamav2": "exllamav2",
    "awq": "awq",
    "gptq": "gptq",
}

PRECISION_DEFAULT_ARG = {"awq": "awq_marlin", "gptq": "gptq_marlin"}


class UnknownQuantArg(ValueError):
    pass


class MissingKernel(ValueError):
    pass


def resolve_kernel(precision: str, quant_arg: str | None) -> str:
    if precision == "fp16":
        return FP16_KERNEL
    arg = quant_arg or PRECISION_DEFAULT_ARG.get(precision)
    if arg is None or arg not in QUANT_ARG_TO_KERNEL:
        raise UnknownQuantArg(
            f"no kernel known for precision {precision!r}, --quantization {arg!r}"
        )
    return QUANT_ARG_TO_KERNEL[arg]


def require_publishable(precision: str, kernel: str) -> None:
    if precision != "fp16" and kernel == FP16_KERNEL:
        raise MissingKernel(
            f"{precision} row has no recorded kernel; "
            "numbers without kernels are unreproducible"
        )
