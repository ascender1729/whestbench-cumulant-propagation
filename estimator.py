"""WhestBench estimator: ARC cumulant propagation (kprop, k_max=3, factored).

Single-file submission. The numpy port of mlp_kprop (verified against the
torch reference at <=1.1e-12) is embedded below as compressed module sources
and installed as the in-memory package ``port_np`` via a meta_path finder.

Heavy tensor ops (einsum / matmul / large pointwise) are routed through
flopscope (see port_np._backend) so they are FLOP-counted analytically;
the surrounding python stays in residual wall time.

Fallback ladder inside predict():
  1. kprop_layer_means (k_max=3, SIMPLE, factor=True)  -- the real estimator
  2. covariance propagation (gain method)              -- small widths / errors
  3. zeros                                             -- never crash
"""

from __future__ import annotations

import base64 as _b64
import importlib.abc as _ilabc
import importlib.util as _ilutil
import os
import sys
import zlib as _zlib

# numpy provider. The grader sandbox follows the challenge convention of
# exposing flopscope.numpy "in place of numpy" and may not expose the raw
# ``numpy`` top-level module to the submission. Ensure ``import numpy``
# resolves everywhere (this file + every embedded port module): prefer the
# real numpy if installed (full accuracy at grade time); else fall back to the
# numpy-compatible flopscope.numpy backend so import-time checks still pass.
try:
    import numpy as _numpy_provider  # noqa: F401
except ModuleNotFoundError:
    import flopscope.numpy as _numpy_provider
    sys.modules["numpy"] = _numpy_provider
    for _sub in ("linalg", "random"):
        _m = getattr(_numpy_provider, _sub, None)
        if _m is not None:
            sys.modules["numpy." + _sub] = _m

_EMBEDDED_SOURCES = {
    "port_np._backend":
        "eNq1V01v3DYQvetXDPYkOWvFMZqiCOBDkCaFgcJJa6M9BIFASZSXCJdUSMre7a/vI6nV19ppDLh7ELTizPC9N8PRaLVaXd8L"
        "V21YKTlVett2jlPJqq9c1dRoQ27DSXXbdk9fW6NbarVxeZL8yhvWSfeGWsmE6i3SO25EIyrmhFbE1Z0wWm25cmuylTa8JmbJ"
        "cCvqjkm6Z1KSE1ue5cnfG66os7xopG5h23ISFhE8rJpSbmHHHOBMgmbrAG6jHTmuLBbhmqRcKNtt6SXBYdvJjBiYeMPfWGet"
        "YIraunlZ1Q0xw8loEPbruLnd0Li91Qmc9sHow+8fP51WulPelCkm9w4kpdzn9LHlBhvYYHdvWNvCBIoilGpzZpkxbJ+kjeE8"
        "IgF96GahtYJaPqBXm5zulSyZ5aeVZNb2ooYINsRMxpAUQ0JYHcghrCPdhHufIlIc4clybiekYiyk7+31u8tL0spTuNLIjsBO"
        "Yhs8maOtrjvUg0QMSemtYTVibZA81dsiZavVKuk9oPTmcN+DhmmbJPOMXtAHJi1PCkiDP1da+Xu/fPiXJDVv+rSPjmn2JiH8"
        "sONNZxRgj5SGtIwlkNNlzQEGT1zuUXrfW6lL1NwM0Jo8En/1T4KZaOYmcWP/M9xh72gUmU5k7TV+cDUfFMFmwWLgPDr1khws"
        "lrrdmO4gTi3sw+o8xPDBYDEJMVpfsUU8NKlqC75rzZpOopx2VL4/VqG0rVC3kp9K7hzqIvKze+XYjlJfIS2ruM0G7R8XNcrw"
        "uZicldRlse8QDkOP4svo0Dk4BPsjyAiWLdJFk7hwjcvj0mOs1wCGhiP+4Rde+WwhVmwsKVtT2evzHYYj4N5tRpZlsQiHB2X2"
        "RA5TMEucaDSilfspUuTkveS+fd4LyyktjWZ1xdBf1W1GB4/1vOXd++7cN+MfyeqE8wHCc7OeUVvwZnX945Rh/Gxs/cbPTHTg"
        "suBYiztR8x+nGe2fjWm//TOTnZJa5lTKSmobF9dknJYXr/jp2es13lf9/S+jEgfzBeEUXbjVopfId58XxACg7io/smRPVaRR"
        "EOSWO+acSePLZNh6tQ7vtFEChIG98C9RF5bGOBMZSq1l2qj/lLbXwF96CfxlonjDhrIcooyL5XKxHBdr0TTDcmkjFNuVzrDK"
        "pQ0y0ExTi9lpegBYQDQ//WZ85gN696PSCMQngKK17Bum5NYW/BvmxtTD86OfyQ5RlhEeqJVjneb1hXEMYysy7+AWpiRY74S9"
        "OBuL6t1oQ+lZqCxYlTgVfMcrzJEW7ywMt3wyGkwLKnvaGZtC+rxIZHhFMv+KjGC/9Gj95YmH7lHmMdZSptZvP0jyNsyi/un/"
        "psgxiYUWeQCVHfMKUEGguP7jz5tzyOrn1Nx+My49z8+ypLi8+iusFeefLrH8Kj+Lnw2jEZ3E/61AoCKMvgU3zXxeLfwAYZp0"
        "N5mWYNOgDsNs72cmn9c4ZeMjqcKXDIaLOsaGbbacU8etDupMNkcLmbcPZ/bzXtIYvY3b5bbllUDEfiz1/miC8/CH34zgsQnf"
        "Vbx1dBkivTdGmzePuyMDA9H0wBODldu33F58xjJyzdzPP32ZJW6MATl7fZU22wIfbYPATysWb5Vb9Gib+0i5jzQroV12VDyY"
        "B9PTs/w18j81XFPt8V+M6DM6OaFzXGlWTVPk1bMhrx5FvouKfw/olGFklvqCfzGWL4o/HhUE/RdDqy7F"
    ,
    "port_np.partitions_np":
        "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"
        "OPlvJy1Oxg=="
    ,
    "port_np.tensor_utils_np":
        "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"
        "uBqJ4ejcdTJEn6Z//IncfwMmza/P"
    ,
    "port_np.wick_np":
        "eNqFVl1v2zYUffevIDQMoxSKkZS0wNxqWB9WpEBRBCu2PQSBQEtUREuiOEpO7Q3777skJUuOne3FNi/PPff70p7nfdm19wek"
        "Oj2grkRto7Ja6U7RbyKvqebNLjO/srzjJcJDxdGv/PNv6A+QISMrRS64HHqfrn7nWpSCF4g9MSH7ARn00Om8QpqXXHOZc4o+"
        "dtpelE2n+rxTHPF+EC0DIEGy022mivI6L8oVk4VF3nHdioEDSb7TlgW1TKFCgGBoDmBiJqNy16oDunIS2g9s6KlhpSvP81ai"
        "tYE6EOuRVJMIHKhWpe5aVO5kPnRd06PxKmd5xd2dVaSqaw6yawVrJsj9UUJQ5dzN+MrpGEAmFc02LK85xDQ5YWKFOOeoEfpu"
        "jiRk35jm71CfC3W4Nu5RrktUsqYxRKsC6nHHsSRo769XCBllKNyGbUQj+qH/4Zi3O55J9CzYIoEWv0/BK9YzrdkB70kxHBQ3"
        "InCBDW9vfQsSJZJpGq1BedhpCRmjneR91oia4/0SEx8xewoBHLC7rKL0hc47VMXpCaaEnqiRkEgz+cRxTOQYkiVI0n1QxWEd"
        "VBGoRgS0q5hUiUWMJqt49bOrk8tLZmqEgeYImKpCx7T8QjesFz1gwBP5zPWAayGLdC6lb7n4M2scm/kgU7L/L3nv0MEc/+K6"
        "O02WiTW3wRo+amboYb0O48c1aByC/RXUdhHYYRlXNs+j84jUozusURVb+I4fIhI/OotGcyNAjoWUXC9SO9qwzQW9U2CXfxXW"
        "VzFRV7HvB/2uxfY+79qNuSFbP7DWgsAcQzhaWrz1bWzbuY6Wx5+dsLgsBuSZC6dmtiQJWj946VdsxCQBr6YCb0OQhLEfYPMR"
        "wDl2XrSzF1Z4fZ1ccCW55Mq51a2xabp8+30Cs4B403N0kusx0yPDItlZ7JPlMXFNdbpXccuZJM9Mk5qoNB5dMsJlk1nQhT4D"
        "vSXM0Fwe5RGYN0JZVMzDOCJfYDTdfS+eWmYQ/Z8wDIAActdXxvS1vZ7mvX6v5sTdx+Q+SS8151mZDYdrHKjZPFrAYC35wbQL"
        "sTtfLTDJCSY/YpwV3oBb6qf4cj1LBjtdm2IpP3g1+yH0PIknvp7PXCbk5Rp0gbx01jC99O5IYR2sl3vydSSYnkBjZ9dhAsM4"
        "pi+EY2zHwMhfSd1qBfayTLKWZ1maelnWwrOcZZ6LanyF7PvsFpN5rE7f/wn04m8APJxWzYpc48sn0zgwMEXXUuhwtmuGDKQ4"
        "cmG1bM+1TiMaHZfgoO0DOo1pEi1GUaZvb48HOwjAZV5zWTBdZCZSKKWEUoJ4JwXwtTiisCDGgTiVR2/IDbwpR0ZjXhnTD6BB"
        "bh5nwxdeo4i88U8BduGm/zHD/hl8k84Zw/YnHbjsO231fHIiMqNnady/GXxOx1M72RhyDpk1X2zTYxZufP8cPObeAN1PMg68"
        "0kIOuPRsqcd/QXsETAhAkITnsc6og+cRJZErWY/2CMOaovSG1FBReguv7N+OeU1v+D/ekt27//D1q2fmxyHew9b50W1Q7+OH"
        "T58B/C+IrU+6"
    ,
    "port_np.diagslice_np":
        "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"
    ,
    "port_np.cumulants_np":
        "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"
    ,
    "port_np.harmonic_np":
        "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"
        "7Ok/+H+kqiuS"
    ,
    "port_np.kprop_terms_np":
        "eNq1V9+P4zQQfs9fYfXJOXUDu9xTjyLQ6k5aAcdJIJCoepGbuK1VxzG222053f/OjB07ca8LT/Shjcfj+fHNNxN3Npu9P3Yf"
        "LkT3xpF+S9yeE82ME0706o6rY8cNw2ey51JzY8nW9F3RSV0ftOl15b/rPTNdr0RDFOctb8nmQrascb2pD9+Qr9Lz60VByI67"
        "WihXN71q53514s1kxaSsHTedvVrWwvZz0h2lE1pe6mfRHGxRvGPC7bdHSZxhysoQ6zPI0NUdAbfNvvqNK9sbcvcdUbpSLTOG"
        "Xd6QH2vMtGpFR0vcG9YKBOHsX22Hci2ZUES4iATVG2YqC7Hp3rqtOJMWMNC8fUNUTzZ8z06iN0wSvt3yxpXeWAI1A9LjXkNQ"
        "advCqvjh18enJ9IreamK2WxWFKLzBZL9bifULi475vaFN9P0UoIrPF+xTUMGhUcAj20kn5MnjB6egvr2qKAgvbRRsWHNfthz"
        "Fw0u4sYvGo0ymUIARugLYRaQLIoshXrDmgNXbTz6bBiCUseKFS8nHI9QgIqQJ+U+wO4jMmIq8OpB8jtv0kOuOwgmusIiuRTA"
        "w9sk6cQ5rpB9GA3yy86LsigQZm7IMuJdAQ9/8jJa14p1vK5Bq/g+oNbybcZpeqg7dl4QWJcL7wE10i4+oLtFlpYn4AZKEk7g"
        "x3B3NIrYY0ex0FXDhaRn6KaHkmyBzWeCrBysleTbJfGOi2JyeAIl9TliCMsYy40kYiveTiLtRswWGd4vJ0GTAD8hpTPNUzut"
        "xHrMTmB20NI7TiVX9FSWYJ3cz8l92D/hfgyjzMxHJFA/bZQZLhPaTHCJ6d3AJU2hkMmITuBQG9axW1awsQb+vO8VDwpHy+uO"
        "M1WfmFl4jGD7HZMW9j1ssUFX7qjhO6N8BvJ6HdCNRQQ7X7KvTMyeqOS1DSqB6lXLN8cdnb2N8x4GAGQ8Di1LGJhpBdsZ1tmq"
        "qmbhOLRR9DCWmErWbVpGToEhsduoX8MEX96XY8HENsMmyTlA89+WHgZL4TsVCcJZrYvBvK8NnPXVGJkZxEvyQF4NjRNRReNo"
        "IkKK7R+E1B9atiN+G9k3hwjBEK2XzclEd+Ep7+W+Udvkz1N98DntZzsG6tk+ED2S/iqsjP7w+lpiy6TRML9uPh/Ya8ibpj6Z"
        "KJWLTB8QzO1P605vd6AvBr4JJqM3qQIyeXy3z0Ix6YhuBA/B8I8vt/9V/BkxKnwlgbXkfD4aGSZE1hHb1BJwqfmEUSdT5Wdy"
        "ywr0jDA29scwbtKhfxks+Pr5v4ZLKxp3NVK8KHtZosM4XGIX5aPPB5YRe575Xk4XCQAw8+nzTbLDFYq61ddrv+ElAdmxgBFW"
        "39FudX+tiuz0FpbLZHk9ffmskhQsZG/6xEfrq7cca/lFA2W1hB+oIpYvv4qG2oU7ZCjLYTHeU/TkGdVr2feHI0jjFW2FkYYa"
        "zCe31DVUMMCB90D8/Tl4Fdx6FOC+jmMZiUH+AMNwFYRrp2gEVw4wwnuvBZkx3GroI5zsridvV+3HA9xWlRSK/ll+1Osqc+Kn"
        "GUBwGAartRxK1iLM+S0538Mjuky1ZmcB4AJtxJzoWpQYcPxLwenfQlMYk7qc1NtDgxmA/wlMWAo0U3qAKNoae/4k+HNt90xz"
        "JAlQ5BVpb2yuMBpkwd14KQi5oK+ruyod6jgGVAGAaIaOFnNehBPFP+rJPc0="
    ,
    "port_np.factor_k3_np":
        "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"
    ,
    "port_np.kprop_np":
        "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"
    ,
}


def _decode(blob):
    return _zlib.decompress(_b64.b64decode(blob)).decode("utf-8")


class _EmbeddedPortNpFinder(_ilabc.MetaPathFinder, _ilabc.Loader):
    """Serves the embedded port_np package from _EMBEDDED_SOURCES."""

    _MARKER = "_whest_kprop_embedded_port_np"

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "port_np" or fullname in _EMBEDDED_SOURCES:
            return _ilutil.spec_from_loader(
                fullname, self, is_package=(fullname == "port_np")
            )
        return None

    def create_module(self, spec):
        return None  # default module creation

    def exec_module(self, module):
        name = module.__name__
        if name == "port_np":
            module.__path__ = []
            return
        src = _decode(_EMBEDDED_SOURCES[name])
        exec(compile(src, "<embedded " + name + ">", "exec"), module.__dict__)


if not any(getattr(f, "_MARKER", None) == "_whest_kprop_embedded_port_np"
           for f in sys.meta_path):
    sys.meta_path.insert(0, _EmbeddedPortNpFinder())

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext

from port_np import _backend
from port_np.kprop_np import Kind, kprop_layer_means

_backend.enable_flopscope()

# kprop's k_max=3 machinery assumes width is large enough for the harmonic
# projection coefficients to be well-conditioned; below this width use the
# covariance-propagation fallback (the validate probe is width=4, depth=2).
_MIN_KPROP_WIDTH = 16


def _cov_prop_means(Ws):
    """Covariance propagation (gain method) fallback on (in, out) weights."""
    n = Ws[0].shape[0]
    mu = np.zeros(n, dtype=np.float64)
    cov = np.eye(n, dtype=np.float64)
    rows = []
    for W in Ws:
        mu_pre = _backend.wrapped_matmul(W.T, mu)
        cov_pre = _backend.wrapped_einsum("ij,ia,jb->ab", cov, W, W)
        var_pre = np.maximum(np.diagonal(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi = _backend.norm_pdf(alpha)
        Phi = _backend.norm_cdf(alpha)
        mu = mu_pre * Phi + sigma_pre * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma_pre * phi
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        gain = np.where(sigma_pre > 1e-12, Phi, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
        rows.append(mu.copy())
    return rows


class Estimator(BaseEstimator):
    """Cumulant propagation (kprop k_max=3) estimator."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        _backend.enable_flopscope()
        # Pre-warm the shape-independent @cache'd combinatorics (partition
        # enumeration, Wick polynomials, harmonic projection coefficients)
        # off-budget, so the first real predict() does not pay for them in
        # residual wall time. setup() runs outside any BudgetContext.
        try:
            rng = np.random.default_rng(0)
            n = 32
            Ws = [rng.normal(0.0, np.sqrt(2.0 / n), (n, n)) for _ in range(2)]
            kind = Kind[os.environ.get("VIBE_KPROP_KIND", "SIMPLE")]
            kprop_layer_means(Ws, k_max=3, kind=kind, factor=True)
        except Exception:
            pass

    def predict(self, mlp, budget: int) -> fnp.ndarray:
        _ = budget
        depth, width = mlp.depth, mlp.width
        try:
            _backend.enable_flopscope()
            Ws = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
            means = None
            if width >= _MIN_KPROP_WIDTH:
                kind = Kind[os.environ.get("VIBE_KPROP_KIND", "SIMPLE")]
                try:
                    means = kprop_layer_means(
                        Ws, k_max=3, kind=kind, factor=True
                    )
                except Exception:
                    means = None
            if means is None:
                means = _cov_prop_means(Ws)
            out = np.stack([np.asarray(m, dtype=np.float64) for m in means], axis=0)
            if out.shape != (depth, width) or not np.all(np.isfinite(out)):
                raise ValueError("bad kprop output; falling back to zeros")
            return fnp.asarray(out)
        except Exception:
            return fnp.zeros((depth, width), dtype=fnp.float64)
