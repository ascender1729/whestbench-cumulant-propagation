"""NumPy port of the partition-enumeration helpers from
mlp_kprop.kprop_harmonic needed by factor_k3 / factor_k4:
  get_int_cond, get_vec_cond, get_all_terms, get_all_terms_iso, multiply_wicks

Faithful translation with:
  - torch.Tensor -> np.ndarray; K_part.dim() -> K_part.ndim
  - tqdm -> plain iteration (pbar.set_postfix dropped; no behavioral effect)
  - partition helpers from port_np.partitions_np
ASCII only.
"""

import logging
import math
from collections.abc import Callable, Iterable
from functools import cache
from typing import Optional

import numpy as np

from port_np._backend import wrapped_multiply
from port_np.partitions_np import (
    IntPartCond,
    IntPartition,
    Vec,
    VecPartCond,
    VecPartition,
    is_connected,
    is_mixed,
    vec_part_isos,
)

logger = logging.getLogger(__name__)


@cache
def get_int_cond(k_max: int):
    def int_cond(int_part: IntPartition) -> bool:
        return sum(math.ceil(x / 2) for x in int_part) <= k_max

    return IntPartCond(part_cond=int_cond)


@cache
def get_vec_cond(k_max: int):
    def vec_cond(vec_part: VecPartition) -> bool:
        return (
            sum(max(sum(math.ceil(v[i] / 2) for i in range(len(v))) - 1, 1) for v in vec_part)
            <= k_max - 1
        )

    return VecPartCond(part_cond=vec_cond)


@cache
def get_all_terms(
    k_max: int,
    d_max: Optional[int] = None,
    use_mean_var: bool = False,
) -> Iterable[tuple[IntPartition, VecPartition]]:
    int_cond = get_int_cond(k_max)
    vec_cond = get_vec_cond(k_max)
    logger.debug("Enumerating all partitions and diagrams...")
    mix_cond = (
        (lambda vpart: is_mixed(vpart, m=1))
        if use_mean_var
        else (lambda vpart: is_mixed(vpart, m=2))
    )
    all_terms = []
    if d_max is None:
        d_max = 2 * k_max
    int_parts = int_cond.get_parts(d_max=d_max)
    block_cond = lambda block, d_max=d_max: sum(block) <= d_max
    for int_part in int_parts:
        for vec_part in vec_cond.get_parts(
            dim=len(int_part),
            sum_max=4 * (k_max - 1),
        ):
            if (
                mix_cond(vec_part)
                and is_connected(vec_part, d=len(int_part))
                and all(block_cond(block) for block in vec_part)
            ):
                all_terms.append((int_part, vec_part))

    logger.debug(f"Enumerated {len(all_terms)} (int_part, vec_part) pairs.")
    return all_terms


@cache
def get_all_terms_iso(
    k_max: int,
    d_max: Optional[int] = None,
    use_mean_var: bool = False,
) -> dict[IntPartition, dict[VecPartition, int]]:
    terms = get_all_terms(k_max, d_max=d_max, use_mean_var=use_mean_var)
    ret = {}
    for int_part in set(t[0] for t in terms):
        vec_parts = [t[1] for t in terms if t[0] == int_part]
        ret[int_part] = vec_part_isos(vec_parts, vec=int_part, dim=len(int_part))
    return ret


def multiply_wicks(
    K_part,
    k: Vec,
    p: Vec,
    wick_lookup: Callable[[int, int], np.ndarray],
):
    """
    Multiplies in the diagonal Wick coefficient tensors corresponding to E[d^k nonlin(Z)^p].
    """
    d = len(k)
    assert d == K_part.ndim
    assert d == len(p)
    for axis, (k_i, p_i) in enumerate(zip(k, p)):
        wick_coef = wick_lookup(int(k_i), int(p_i))
        view_shape = [1] * d
        view_shape[axis] = -1
        K_part = wrapped_multiply(K_part, wick_coef.reshape(view_shape))
    return K_part
