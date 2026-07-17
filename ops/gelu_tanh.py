"""
Ascend 310 TBE custom operator: GELU with tanh approximation.
GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

This is used by the Qwen3.5 vision encoder.
"""
import math
from te import tik
from te.lang import cce

GELU_CONST = math.sqrt(2.0 / math.pi)


def gelu_tanh_compute(input_x, output_y, kernel_name="gelu_tanh"):
    """Compute GELU_tanh activation using Ascend Vector Unit.

    Uses te.lang.cce for element-wise operations mapped to Vector Unit.
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    # Use the high-level API for element-wise ops
    # These map to efficient Vector Unit instructions

    # x^3
    x_cube = cce.vmul(input_x, input_x)
    x_cube = cce.vmul(x_cube, input_x)

    # 0.044715 * x^3
    x_cube_scaled = cce.vmuls(x_cube, 0.044715)

    # x + 0.044715 * x^3
    inner = cce.vadd(input_x, x_cube_scaled)

    # sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = cce.vmuls(inner, GELU_CONST)

    # tanh(...)
    tanh_out = cce.vtanh(inner)

    # 1 + tanh(...)
    one = cce.broadcast(tanh_out, 1.0)
    plus_one = cce.vadd(tanh_out, one)

    # 0.5 * x * (1 + tanh(...))
    result = cce.vmul(input_x, plus_one)
    result = cce.vmuls(result, 0.5)

    return result


def gelu_tanh(x):
    """Python-level GELU_tanh for CPU fallback."""
    import numpy as np
    return 0.5 * x * (1.0 + np.tanh(GELU_CONST * (x + 0.044715 * x ** 3)))
