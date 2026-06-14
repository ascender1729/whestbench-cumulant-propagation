"""Switchable compute backend for the numpy kprop port.

Default: plain numpy (verification environment, scored as residual wall time).
When use_flopscope is enabled (estimator environment), the hot tensor ops
(einsum / matmul) and the Gaussian pdf/cdf are routed through flopscope so
they are FLOP-counted analytically. Operands are wrapped with fnp.asarray
(free) and results converted back to plain base-class numpy arrays with
np.asarray (free), so the rest of the port never sees flopscope arrays.

ASCII only. No scipy import at module level (grader has no scipy).
"""
import math
import numpy as np

use_flopscope = False
_fnp = None
_flops = None


def enable_flopscope():
    """Turn on flopscope-counted tensor ops. Idempotent."""
    global use_flopscope, _fnp, _flops
    if use_flopscope:
        return
    import flopscope as flops
    import flopscope.numpy as fnp
    _flops = flops
    _fnp = fnp
    use_flopscope = True


def disable_flopscope():
    global use_flopscope
    use_flopscope = False


def wrapped_einsum(np_expr, *tensors):
    """einsum with single-letter numpy syntax (no spaces)."""
    if use_flopscope:
        ops = [_fnp.asarray(t) for t in tensors]
        out = _fnp.einsum(np_expr, *ops)
        return np.asarray(out)
    return np.einsum(np_expr, *tensors, optimize=True)


def wrapped_matmul(a, b):
    if use_flopscope:
        out = _fnp.matmul(_fnp.asarray(a), _fnp.asarray(b))
        return np.asarray(out)
    return np.matmul(a, b)


def wrapped_multiply(a, b):
    """Elementwise (broadcasting) multiply, FLOP-counted when enabled."""
    if use_flopscope:
        out = _fnp.multiply(_fnp.asarray(a), _fnp.asarray(b))
        return np.asarray(out)
    return np.multiply(a, b)


def wrapped_add(a, b):
    """Elementwise (broadcasting) add, FLOP-counted when enabled."""
    if use_flopscope:
        out = _fnp.add(_fnp.asarray(a), _fnp.asarray(b))
        return np.asarray(out)
    return np.add(a, b)


def wrapped_divide(a, b):
    """Elementwise (broadcasting) divide, FLOP-counted when enabled."""
    if use_flopscope:
        out = _fnp.divide(_fnp.asarray(a), _fnp.asarray(b))
        return np.asarray(out)
    return np.divide(a, b)


def wrapped_allclose(a, b, rtol=1e-05, atol=1e-08):
    """allclose, FLOP-counted (as pointwise ops + a reduction) when enabled."""
    if use_flopscope:
        fn = getattr(_fnp, "allclose", None)
        if fn is not None:
            return bool(fn(_fnp.asarray(a), _fnp.asarray(b), rtol=rtol, atol=atol))
        fa = _fnp.asarray(a)
        fb = _fnp.asarray(b)
        diff = _fnp.abs(_fnp.subtract(fa, fb))
        thr = _fnp.add(atol, _fnp.multiply(rtol, _fnp.abs(fb)))
        return bool(np.asarray(_fnp.all(_fnp.less_equal(diff, thr))))
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def wrapped_concatenate(arrays, axis=0):
    """Concatenate (0 FLOPs, but executes inside flopscope when enabled)."""
    if use_flopscope:
        out = _fnp.concatenate([_fnp.asarray(a) for a in arrays], axis=axis)
        return np.asarray(out)
    return np.concatenate(arrays, axis=axis)


def wrapped_copy(a):
    """Array copy (0 FLOPs, but executes inside flopscope when enabled)."""
    if use_flopscope:
        return np.asarray(_fnp.asarray(a).copy())
    return np.copy(a)


_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

_scipy_erf = None


def _np_erf(x):
    """erf fallback without scipy (vectorized math.erf)."""
    global _scipy_erf
    if _scipy_erf is None:
        try:
            from scipy.special import erf as scipy_erf
            _scipy_erf = scipy_erf
        except ImportError:
            _scipy_erf = np.vectorize(math.erf, otypes=[np.float64])
    return _scipy_erf(x)


def norm_pdf(x):
    if use_flopscope:
        return np.asarray(_flops.stats.norm.pdf(_fnp.asarray(x)))
    return np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2) * _INV_SQRT_2PI


def norm_cdf(x):
    if use_flopscope:
        return np.asarray(_flops.stats.norm.cdf(_fnp.asarray(x)))
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + _np_erf(x / _SQRT2))
