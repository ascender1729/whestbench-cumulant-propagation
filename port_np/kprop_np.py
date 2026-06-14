"""NumPy port of the driver flow of mlp_kprop.kprop_harmonic.

Ports: Kind, get_r_x, get_d_max, coerce_input, clone_tower, linear_kprop,
nonlin_kprop (full body incl. the factor=True delegation to
factored_nonlin_kprop_k3), plus a whestbench-friendly entry point
kprop_layer_means.

Faithful translation with:
  - torch.Tensor -> np.ndarray (float64); device/dtype plumbing dropped
  - tqdm dropped; flop_name is a no-op
  - partition enumeration (get_int_cond / get_vec_cond / get_all_terms /
    get_all_terms_iso / multiply_wicks) imported from port_np.kprop_terms_np
ASCII only. py3.10.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from functools import cache
from typing import Optional

import numpy as np

from port_np.cumulants_np import DS_pK_to_K
from port_np.diagslice_np import DSTensor, DSTower, eval_part
from port_np.harmonic_np import DS_harmonic_proj, HTensor, proj_geq_r
from port_np.kprop_terms_np import get_all_terms_iso, multiply_wicks
from port_np.partitions_np import check_vec_partition
from port_np.tensor_utils_np import flop_name, symmetrize
from port_np.wick_np import relu_wick_coef

logger = logging.getLogger(__name__)

# HTower: dict mapping cumulant degree d -> HTensor (or FactoredTensor when
# factor=True).
HTower = dict


class Kind(Enum):
    OLD = 1
    SIMPLE = 2
    AUGMENT = 3
    BASE = 4  # Just for ablation tests; not expected to get good enough MSE.


OLD, SIMPLE, AUGMENT, BASE = Kind.OLD, Kind.SIMPLE, Kind.AUGMENT, Kind.BASE


def get_r_x(d: int, k_max: int, kind: Kind = SIMPLE) -> int:
    """
    Given budget parameter k_max, we track the d-th cumulant going into the
    linear step as Sym(A otimes I^{otimes r_x}), where A is an order d-2*r_x
    tensor. Power counting says r_x = d-k_max.
    Return value of -1 means the entire cumulant should be discarded.
    """
    if kind == SIMPLE:
        if d > k_max:
            if d == k_max + 1 and d % 2 == 0:
                return d // 2
            else:
                return -1
        else:
            return 0
    elif kind == AUGMENT:
        if d > k_max:
            if d == k_max + 1:
                return 1
            elif d == k_max + 2 and d % 2 == 0:
                return d // 2
            else:
                return -1
        else:
            return 0
    elif kind == OLD:
        r = max(d - k_max, 0)
        if 2 * r > d:
            return -1
        else:
            return r
    elif kind == BASE:
        if d > k_max:
            return -1
        return 0
    else:
        raise ValueError(f"Unknown kind: {kind}")


def get_d_max(k_max, kind: Kind) -> int:
    """
    Returns the maximum cumulant degree tracked given budget parameter k_max
    and kind.
    """
    if kind == SIMPLE:
        return k_max + 1 if k_max % 2 == 1 else k_max
    elif kind == AUGMENT:
        return k_max + 2 if k_max % 2 == 0 else k_max + 1
    elif kind == BASE:
        return k_max
    else:
        # Maximum possible degree of diagslice satisfying ceil(alpha/2)
        # int_cond is 2*k_max
        return 2 * k_max


def _coerce_layer_bias(bias, *, out_dim: int):
    if bias is None:
        return None
    bias = np.asarray(bias, dtype=np.float64)
    if bias.ndim != 1 or bias.shape[0] != out_dim:
        raise ValueError(f"bias must have shape ({out_dim},), got {tuple(bias.shape)}")
    return bias


def linear_kprop(
    K: HTower,
    W,
    k_max: int,
    d_max: Optional[int] = None,
    *,
    set_metric=None,
    bias=None,
) -> HTower:
    """
    Linear step of cumulant propagation: contracts each K[d] with W.
    Used before the nonlinear step.

    Args:
        set_metric: Metric to set on the output HTensors. If None, the metric
            is computed as W @ old_metric @ W^T (see contract_W).
            Set to diag(E[WW^T]) = mlp.init_scale[layer]*I when using average
            metric.
    """
    W = np.asarray(W, dtype=np.float64)
    n_out = W.shape[0]

    if set_metric is None and k_max == 1:
        # When k=1, the only non-loop edge is in the leading order (1, 1)
        # partition for variance which gets traced out in the projection after
        # the nonlinearity, so we only need the diagonal. This also saves us
        # from going over budget when k=1.
        with flop_name('metric'):
            set_metric = (W ** 2).sum(axis=1)

    WK: HTower = {}
    for d, K_d in K.items():
        if d_max is not None and d > d_max:
            continue
        if isinstance(K_d, HTensor):
            assert K_d.has_identity_metric(), (
                f"linear_kprop expects identity metric on input HTensors, got {K_d}"
            )
            WK[d] = K_d.contract_W(W, set_metric=set_metric)
        elif hasattr(K_d, "contract_W"):
            WK[d] = K_d.contract_W(W)
        else:
            raise TypeError(f"Unsupported tensor type in linear_kprop: {type(K_d)!r}")

    if bias is not None:
        bias_vec = _coerce_layer_bias(bias, out_dim=n_out)
        if bias_vec is not None:
            WK[1] = WK[1].clone()
            WK[1].core = WK[1].core + bias_vec
    return WK


def nonlin_kprop(
    K_in: HTower,
    nonlin_wick_coef: Callable[[float, float, int, int], float],
    k_max: int,
    kind: Kind = SIMPLE,
    use_pK: bool = True,
    factor: bool = False,
    mean_only: bool = False,
) -> HTower:
    """
    Propagate cumulants through nonlinearity.
    We first compute power cumulants via Wick expansion around a Gaussian with
    matching mean and variance (so the sum is over 2-mixed partitions); then we
    convert back to ordinary cumulants.

    Args:
        K_in: Input cumulants (from the linear step; may have non-identity
            metric)
        k_max: Budget parameter. We want final error O(n^{-k_max}).
        nonlin_wick_coef: 1d Wick coefficients wrt a Gaussian.
            (mean, var, k, p) -> E_{Z~N(mean,var)}[d^k nonlin(Z)^p]
        factor: Use a factorized representation for the top-degree cumulant.
            Only supported for k_max=3 in this port.

    Returns:
        K_out: Output cumulants (with identity metric)
    """
    if not use_pK and kind != BASE:
        raise NotImplementedError("not use_pK ablation only implemented for kind=BASE.")

    n = K_in[1].n

    if factor:
        if k_max > 4:
            raise NotImplementedError("Factored nonlin_kprop only implemented for k_max=3 or 4")
        assert kind in (SIMPLE, AUGMENT, BASE), (
            "Factored nonlin_kprop only implemented for kind=SIMPLE, AUGMENT, or BASE"
        )
        if k_max == 3:
            from port_np.factor_k3_np import factored_nonlin_kprop_k3
            return factored_nonlin_kprop_k3(
                K_in=K_in,
                nonlin_wick_coef=nonlin_wick_coef,
                augment=(kind == AUGMENT),
                base=(kind == BASE),
                use_pK=use_pK,
                mean_only=mean_only,
            )
        elif k_max == 4:
            raise NotImplementedError("factor_k4 is not ported to numpy (factor_k3 only).")
        else:
            logger.debug("nonlin_kprop with factor=True called with k_max<=2. Identical to unfactored.")

    # 1. Get propagated mean and variance
    with flop_name('get_mean_var'):
        assert K_in[1].r == 0
        mean = K_in[1].core
        if k_max == 1:
            if 2 not in K_in:
                # This only happens on k=1 kind=BASE
                var = np.ones_like(mean)
            else:
                assert K_in[2].r == 1
                var_metric = K_in[2].metric
                var = K_in[2].core * (
                    var_metric if var_metric.ndim == 1 else np.diagonal(var_metric)
                )
        else:
            assert K_in[2].r == 0
            var = np.diagonal(K_in[2].core)
        assert mean.ndim == 1, "Mean must be a vector."
        assert var.ndim == 1, "Variance must be a vector."

    @cache
    @flop_name('get_wick_coef')
    def get_wick_coef(k: int, p: int):
        return nonlin_wick_coef(mean=mean, var=var, k=k, p=p)

    # 2. Compute pK
    terms_iso = get_all_terms_iso(k_max, d_max=get_d_max(k_max, kind))
    terms_iso = [
        (int_part, vec_part, count)
        for int_part, vec_part_dict in terms_iso.items()
        for vec_part, count in vec_part_dict.items()
        # If not use_pK, only need (1, ..., 1) int_parts since we're not going
        # to zero the diagonal
        if all(p == 1 for p in int_part) or use_pK
    ]
    pK_slices = defaultdict(lambda: 0.0)
    for int_part, vec_part, count in terms_iso:
        with flop_name('nonlin_sum'):
            term = eval_part(K_in, vec_part, len(int_part), output_zero_repeated=use_pK)
            if term is None:
                continue
            pK_slices[int_part] += count * multiply_wicks(
                term,
                # check_vec_partition returns sum of partition vectors
                check_vec_partition(vec_part, len(int_part)),
                p=int_part,
                wick_lookup=get_wick_coef,
            )
    # Since we sum over iso classes * count instead of all terms, each slice is
    # not symmetric wrt its int_part. So we symmetrize here.
    with flop_name('symmetrize'):
        for int_part in pK_slices:
            pK_slices[int_part] = symmetrize(pK_slices[int_part], vec=int_part)

    # If not use_pK, pK_slices already contain our cumulant estimate. So
    # immediately project to harmonic and return.
    if not use_pK:
        ret = {}
        for d in range(1, get_d_max(k_max, kind) + 1):
            part = (1,) * d
            if part not in pK_slices:
                continue
            ret[d] = proj_geq_r(pK_slices[part], n=n, r_out=get_r_x(d, k_max, kind=kind))
        return ret

    # 3. Convert pK to K
    pK_out_ds = DSTower.from_slices(pK_slices, autozero=True)
    K_out_ds = DS_pK_to_K(pK_out_ds)

    # 4. Project to harmonic form
    K_out: HTower = {}
    for d, K_d_ds in K_out_ds.items():
        r_x = get_r_x(d, k_max, kind=kind)
        if r_x == -1:
            continue
        K_out[d] = DS_harmonic_proj(K_d_ds, r_x)
    return K_out


def relu_kprop(K_in: HTower, k_max: int, kind: Kind = SIMPLE) -> HTower:
    return nonlin_kprop(K_in, nonlin_wick_coef=relu_wick_coef, k_max=k_max, kind=kind)


@flop_name('coerce_input')
def coerce_input(K: dict, k_max: int, kind: Kind = SIMPLE) -> HTower:
    d_max = max(K.keys())
    n = np.asarray(K[d_max]).shape[0] if not isinstance(K[d_max], (HTensor, DSTensor)) else K[d_max].n
    K_out: HTower = {}
    for d, K_d in K.items():
        r = get_r_x(d, k_max, kind=kind)
        if r == -1:
            continue
        if isinstance(K_d, HTensor):
            K_out[d] = K_d.to(dtype=np.float64)
        elif isinstance(K_d, DSTensor):
            K_out[d] = DS_harmonic_proj(K_d.to(dtype=np.float64), r_out=r)
        else:
            K_d = np.asarray(K_d, dtype=np.float64)
            assert K_d.shape == (n,) * d, f"K[{d}] must have shape (n,)*{d}."
            K_out[d] = proj_geq_r(K_d, n=n, r_out=r)
    return K_out


@flop_name('clone_tower')
def clone_tower(K: HTower, d_max: Optional[int] = None) -> HTower:
    if d_max is None:
        d_max = max(K.keys())
    return {d: K_d.clone() for d, K_d in K.items() if d <= d_max}


def kprop_layer_means(
    Ws,
    k_max: int = 3,
    kind: Kind = SIMPLE,
    factor: bool = True,
    metric_by_layer=None,
):
    """
    Whestbench-friendly entry point: cumulant propagation through an
    all-ReLU MLP, returning the post-ReLU mean vector after every layer.

    Args:
        Ws: list of L numpy (n, n) weight matrices in the WHESTBENCH
            convention: stored (in, out), forward pass h_new = relu(h @ W).
            Internally converted to the ARC convention W_arc = W.T (out, in).
            Every weight matrix is followed by a ReLU (no trailing linear
            layer, no biases).
        k_max: budget parameter. We want final error O(n^{-k_max}).
        kind: which cumulants to track given k_max (see get_r_x).
        factor: use the factorized top-degree representation (k_max=3 only).
        metric_by_layer: per-layer average metric diag(E[WW^T]) scalar.
            Defaults to [2.0]*L (whestbench He N(0, 2/n) gives E[WW^T] = 2I).

    Returns:
        List (length L) of plain (n,) numpy mean vectors, one per post-ReLU
        activation.
    """
    L = len(Ws)
    assert L >= 1, "Need at least one weight matrix."
    n = np.asarray(Ws[0]).shape[0]
    if metric_by_layer is None:
        metric_by_layer = [2.0] * L
    assert len(metric_by_layer) == L, "metric_by_layer must have one entry per layer."

    K_in = {1: np.zeros(n, dtype=np.float64), 2: np.eye(n, dtype=np.float64)}
    K = coerce_input(K_in, k_max=k_max, kind=kind)

    means = []
    last_l = L - 1
    for l, W in enumerate(Ws):
        W_arc = np.asarray(W, dtype=np.float64).T  # whestbench (in,out) -> ARC (out,in)
        K = linear_kprop(K, W_arc, k_max=k_max, set_metric=metric_by_layer[l])
        # Final layer: only the post-nonlin mean K[1] is consumed downstream
        # (the score reads the last layer mean), so skip producing its unused
        # K2/K3/K4 cumulants. Only valid for the factored k_max=3 path.
        ml = factor and (k_max == 3) and (l == last_l)
        K = nonlin_kprop(
            K,
            nonlin_wick_coef=relu_wick_coef,
            k_max=k_max,
            kind=kind,
            factor=factor,
            mean_only=ml,
        )
        mean_ht = K[1]
        assert isinstance(mean_ht, HTensor) and mean_ht.r == 0
        means.append(np.array(mean_ht.core, dtype=np.float64, copy=True))
    return means
