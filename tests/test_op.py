import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
from mlx_fused_rmsnorm import residual_rms_norm


@pytest.mark.parametrize("shape", [(1, 32), (7, 257), (2, 3, 1024)])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_matches_mlx(shape, dtype):
    kind = getattr(mx, dtype)
    x = mx.random.normal(shape).astype(kind)
    residual = mx.random.normal(shape).astype(kind)
    weight = mx.random.normal((shape[-1],)).astype(kind)

    actual = residual_rms_norm(x, residual, weight)
    expected = mx.fast.rms_norm(x + residual, weight, 1e-5)
    mx.eval(actual, expected)

    assert actual.dtype == kind
    tolerance = {"float32": 2e-5, "float16": 2e-2, "bfloat16": 3e-2}[dtype]
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=tolerance,
        atol=tolerance,
    )


def test_noncontiguous_input():
    x = mx.random.normal((4, 128))[:, ::2]
    residual = mx.random.normal((4, 128))[:, ::2]
    weight = mx.random.normal((64,))

    actual = residual_rms_norm(x, residual, weight)
    expected = mx.fast.rms_norm(x + residual, weight, 1e-5)
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), rtol=2e-5, atol=2e-5
    )


def test_custom_epsilon():
    x = mx.random.normal((3, 64))
    residual = mx.random.normal((3, 64))
    weight = mx.ones((64,))

    actual = residual_rms_norm(x, residual, weight, eps=0.01)
    expected = mx.fast.rms_norm(x + residual, weight, 0.01)
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), rtol=2e-5, atol=2e-5
    )


def test_rejects_invalid_weight():
    x = mx.ones((2, 64))
    with pytest.raises(ValueError):
        residual_rms_norm(x, x, mx.ones((32,)))


def test_rejects_invalid_epsilon():
    x = mx.ones((2, 64))
    with pytest.raises(ValueError):
        residual_rms_norm(x, x, mx.ones((64,)), eps=0)
