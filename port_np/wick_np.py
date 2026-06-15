"""NumPy port of mlp_kprop.wick.relu_wick_coef (the ReLU Wick coefficients).
Verified against the torch reference. For the flopscope estimator, norm_pdf/cdf
and the Hermite recurrence map directly to flopscope.numpy + flops.stats.norm.

The polynomial *coefficients* (Hermite He basis, ReLU Wick polynomials) are
shape-independent scalars computed once via a tiny pure-Python polynomial type
(_Poly), so they work identically whether ``numpy`` is real numpy or
flopscope.numpy (which lacks numpy.polynomial). Polynomial *evaluation* (Horner)
runs on the data array via the active numpy backend.
"""
import numpy as np
import math
from functools import cache

from port_np._backend import norm_cdf, norm_pdf  # flopscope-aware; scipy/math.erf fallback


class _Poly:
    """Minimal dense univariate polynomial over floats, coefficients ascending
    (coef[i] is the x**i coefficient). Pure Python: no numpy.polynomial.
    Matches numpy.polynomial.Polynomial for the operations this module uses
    (+, *, **, scalar lmul/rmul, sum() starting at int 0, .coef)."""

    __slots__ = ("coef",)

    def __init__(self, coef):
        c = [float(x) for x in coef]
        if not c:
            c = [0.0]
        # trim trailing zeros (keep at least one term), matching Polynomial
        while len(c) > 1 and c[-1] == 0.0:
            c.pop()
        self.coef = c

    def __add__(self, other):
        if isinstance(other, _Poly):
            a, b = self.coef, other.coef
            n = max(len(a), len(b))
            return _Poly([(a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0)
                          for i in range(n)])
        # scalar
        c = list(self.coef)
        c[0] += float(other)
        return _Poly(c)

    __radd__ = __add__  # so sum([...]) (starts from int 0) works

    def __mul__(self, other):
        if isinstance(other, _Poly):
            a, b = self.coef, other.coef
            out = [0.0] * (len(a) + len(b) - 1)
            for i, ai in enumerate(a):
                if ai == 0.0:
                    continue
                for j, bj in enumerate(b):
                    out[i + j] += ai * bj
            return _Poly(out)
        # scalar
        s = float(other)
        return _Poly([ci * s for ci in self.coef])

    __rmul__ = __mul__

    def __pow__(self, n):
        n = int(n)
        assert n >= 0
        result = _Poly([1.0])
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result


@cache
def _he_coef(n):
    """Probabilists' Hermite He_n coefficients (ascending), as a _Poly.
    Recurrence: He_0 = 1, He_1 = x, He_{k+1} = x*He_k - k*He_{k-1}."""
    if n <= 0:
        return _Poly([1.0])
    h0 = _Poly([1.0])
    h1 = _Poly([0.0, 1.0])
    x = _Poly([0.0, 1.0])
    for k in range(1, n):
        h2 = x * h1 + (-float(k)) * h0
        h0, h1 = h1, h2
    return h1


def He(n, x):
    # probabilists' Hermite He_n evaluated on data via recurrence
    if not hasattr(x, "ndim"):
        x = np.asarray(x, dtype=np.float64)  # skip the wasted fnp call when x is already an array
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0 = np.ones_like(x)
    h1 = x.copy()
    for k in range(1, n):
        h2 = x * h1 - k * h0
        h0, h1 = h1, h2
    return h1


@cache
def He_poly(n):
    return _he_coef(n)


def eval_poly(poly, x):
    if not hasattr(x, "ndim"):
        x = np.asarray(x, dtype=np.float64)  # skip the wasted fnp call when x is already an array
    y = np.zeros_like(x)
    for ck in poly.coef[::-1]:
        y = y * x + ck
    return y


@cache
def _relu_wick_poly(p, k):
    alpha = _Poly([0.0, 1.0])

    def binom(inner):
        return math.prod(range(p - k + 1, p + 1)) * sum(
            math.comb(p - k, j) * alpha ** (p - k - j) * inner(j) for j in range(p - k + 1)
        )

    def inner_1(j):
        return sum(
            math.comb(j, 2 * m) * math.prod(range(1, 2 * m, 2)) * He_poly(j - 2 * m - 1) * (-1) ** (j - 1)
            for m in range((j - 1) // 2 + 1)
        )

    def inner_2(j):
        return math.prod(range(1, j, 2)) if j % 2 == 0 else _Poly([0.0])

    return binom(inner_1), binom(inner_2)


def relu_wick_coef(mean, var, k, p=1, _setup=None):
    # _setup=(mean, var, sigma, alpha) lets the caller hoist this shared setup out
    # of the per-(k,p) loop (sigma/alpha are constant per layer). Recomputing it
    # per call cost ~5 flopscope ops each (asarray x2, maximum, sqrt, divide).
    if _setup is None:
        mean = np.asarray(mean, dtype=np.float64)
        # np.clip delegates to numpy._core.fromnumeric.clip at call time; under the
        # grader's numpy-shim (sys.modules["numpy"]=flopscope.numpy) that lazy import
        # resolves to fnp and dies. np.maximum is a native flopscope ufunc -> safe.
        var = np.maximum(np.asarray(var, dtype=np.float64), 1e-10)
        sigma = np.sqrt(var)
        alpha = mean / sigma
        _setup = (mean, var, sigma, alpha)
    else:
        mean, var, sigma, alpha = _setup
    if k < p:
        P1, P2 = _relu_wick_poly(p, k)
        return sigma ** (p - k) * (eval_poly(P1, alpha) * norm_pdf(alpha) + eval_poly(P2, alpha) * norm_cdf(alpha))
    elif p > 1:
        return math.factorial(p) * relu_wick_coef(mean, var, k - p + 1, 1, _setup=_setup)
    else:
        if k == 0:
            return sigma * norm_pdf(alpha) + mean * norm_cdf(alpha)
        elif k == 1:
            return norm_cdf(alpha)
        else:
            return (-1) ** (k - 2) * sigma ** (-(k - 1)) * He(k - 2, alpha) * norm_pdf(alpha)


if __name__ == "__main__":
    import torch
    from mlp_kprop.wick import relu_wick_coef as torch_wick
    rng = np.random.default_rng(0)
    maxerr = 0.0
    for trial in range(20):
        n = 64
        mean = rng.standard_normal(n) * rng.uniform(0.1, 2)
        var = rng.uniform(0.05, 3, n)
        for p in [1, 2, 3]:
            for k in range(0, 5):
                a = relu_wick_coef(mean, var, k, p)
                b = torch_wick(torch.tensor(mean), torch.tensor(var), k, p).numpy()
                e = float(np.max(np.abs(a - b)))
                maxerr = max(maxerr, e)
    print(f"wick port max abs error vs torch over 20 trials x (p=1..3,k=0..4): {maxerr:.3e}")
    print("PASS" if maxerr < 1e-9 else "FAIL")
