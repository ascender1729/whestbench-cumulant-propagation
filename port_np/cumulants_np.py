"""NumPy port of mlp_kprop.cumulants (moment <-> cumulant <-> power-cumulant
conversions over Towers / DSTowers).

Faithful translation of src/mlp_kprop/cumulants.py with:
  - torch.Tensor -> np.ndarray (float64), torch.zeros[_like] -> np.zeros[_like]
  - Tower = dict[int, np.ndarray] (d -> (n,)*d), matching mlp_kprop.tensor_utils
  - DSTensor / DSTower / partition helpers from port_np siblings
  - flop_name from tensor_utils_np (no-op decorator / ctx manager)

SKIPPED (sampling-only utilities used only by tests/ and scripts/, not by
kprop_harmonic.py, factor_k3.py, factor_k4.py or kprop_ds.py; they depend on
torch sample streams and the MLP class is not imported here):
  - Samples / SampleStream type aliases
  - stream_tensor, finish, _moment_expr, moment_gen_slice
  - DS_moment_gen, DS_cumulant_gen, DS_moment, DS_cumulant
ASCII only.
"""

import logging
import math
from functools import cache

import numpy as np

from port_np._backend import wrapped_add, wrapped_multiply
from port_np.diagslice_np import DSTensor, DSTower
from port_np.partitions_np import set_partitions, vec_part_coef, vector_partitions
from port_np.tensor_utils_np import expand, flop_name

logger = logging.getLogger(__name__)

# Tower = dict[int, np.ndarray]  # d -> tensor of shape (n,)*d


def part_sum(A, coef, d=None):
    """
    Transforms A by summing over set partitions with given coefficient function.
    I.e., returns tA[i_1,...,i_k] = sum_pi coef(pi) prod_{B in pi} A[B].
    """
    if d is None:
        return {d: part_sum(A, coef, d) for d in A}

    for i in range(1, d + 1):
        assert i in A, f"A must contain all orders up to d={d}."

    n = A[1].shape[0]
    dtype = A[1].dtype
    out = np.zeros([n] * d, dtype=dtype)

    for part in set_partitions(d):
        out += coef(part) * math.prod(expand(A[len(block)], tuple(block), d) for block in part)

    assert list(out.shape) == [n] * d
    return out


def M_to_K(M, d=None):
    """
    Converts moment tensors M to dth cumulant tensor K via Moebius inversion.
    """
    coef = lambda part: math.factorial(len(part) - 1) * ((-1) ** (len(part) - 1))
    return part_sum(M, coef, d)


def K_to_M(K, d=None):
    """
    Converts cumulant tensors K to dth moment tensor M via standard partition expansion.
    """
    return part_sum(K, lambda part: 1, d)


def DS_part_sum(A, coef, strict=True):
    """
    Same as part_sum, but for DSTensors, computed per diagonal slice.
    We compute precisely the diagonal slices of the output that appear in A.
    Note that A has the necessary diagonal slices for this if it is downward closed.
    Partitions of a diagonal slice (n_1, ..., n_k) correspond to set partitions of
    sqcup_{i=1}^k [(i, 1),...,(i,n_i)], which correspond to vector partitions of
    (n_1, ..., n_k) multiplied by the fiber size vec_part_coef(divide_fac=False).
    """
    if strict:
        assert A.is_downward_closed(), "A must be downward closed."
    R = DSTower()  # to return

    def get_block(block):
        # Handle zero indices by dropping and expanding
        nonzeros = tuple([i for i, x in enumerate(block) if x > 0])
        block_nz = tuple(block[i] for i in nonzeros)
        try:
            return expand(A[sum(block)].get_slice(block_nz, strict=True), nonzeros, len(block))
        except Exception:
            assert not strict, f"Missing slice {block} in A."
            return 0.0

    for d in range(1, max(A.keys()) + 1):
        Rd_slices = dict()
        for int_part in A[d].slices:
            Rd_slices[int_part] = np.zeros_like(A[d].slices[int_part])
            for vpart in vector_partitions(int_part):
                blocks = [get_block(block) for block in vpart]
                term = blocks[0]
                for blk in blocks[1:]:
                    term = wrapped_multiply(term, blk)
                term = wrapped_multiply(
                    term, vec_part_coef(vpart, divide_fac=False) * coef(vpart)
                )
                Rd_slices[int_part] = wrapped_add(Rd_slices[int_part], term)
        R[d] = DSTensor(Rd_slices, autozero=True)
    return R


def DS_K_to_M(K):
    """
    Cumulants to moments via the usual partition formula, applied to each diagonal slice separately.
    """
    return DS_part_sum(K, lambda vpart: 1)


def DS_M_to_K(M):
    """
    Moments to cumulants via the usual Moebius inversion, applied to each diagonal slice separately.
    """
    coef = lambda part: math.factorial(len(part) - 1) * ((-1) ** (len(part) - 1))
    return DS_part_sum(M, coef)


def DS_pK_to_M(pK):
    """
    Converts power cumulants pK to moments M as a DSTensor.
    We compute precisely the diagonal slices of M that appear in pK.
    We form the moments via the standard partition formula applied to *each diagonal
    slice separately*, e.g. E[X_1^2 X_2 X_3] is treated as an entry in the 3rd moment
    tensor of (X^2, X, X), as opposed to the 4th of (X, X, X, X).
    """
    assert pK.is_downward_closed(), "pK must be downward closed."
    M = DSTower()
    for d in range(1, max(pK.keys()) + 1):
        # Compute moments for each partition separately
        # By symmetry we only need to compute one per integer partition
        Md_slices = dict()
        for int_part in pK[d].slices:
            # Do K_to_M formula, using the correct multiplicities of each pK. i.e., partition int_part itself
            Md_slices[int_part] = np.zeros_like(pK[d].slices[int_part])
            for blocks in set_partitions(
                tuple(enumerate(int_part))
            ):  # need to track int block idx to know how to expand
                Md_slices[int_part] += math.prod(
                    expand(
                        pK.get_slice(sorted(tuple(t[1] for t in B), reverse=True)),
                        tuple(t[0] for t in B),  # idxs of entries in block B
                        len(int_part),
                    )
                    for B in blocks
                )
        M[d] = DSTensor(Md_slices, autozero=True)
    return M


@flop_name('pK_to_K')
def _DS_pK_to_K_old(pK):
    """
    Converts power cumulants pK to cumulants K as a DSTensor.
    We first convert pK to moments M via DS_pK_to_M, then M to K via DS_M_to_K.
    """
    # We want to skip as many (1, ..., 1) computations as possible, since pK->K is a no-op on them.
    # We greedily remove as many as possible while maintaining that pK is downward closed (which is required by pK->M).
    pK = pK.clone()
    ones_slices = dict()
    for d in sorted(pK.keys(), reverse=True):
        ones = (1,) * d
        if ones not in pK[d].slices:
            continue
        ones_slices[d] = pK[d].slices[ones]
        pK[d].slices.pop(ones)
        if not pK.is_downward_closed():
            # Put the slice back
            pK[d].slices[ones] = ones_slices[d]
            ones_slices.pop(d)
            break
    K = DS_M_to_K(DS_pK_to_M(pK))
    for d in ones_slices:
        K[d].slices[(1,) * d] = ones_slices[d]
    return K


@cache
def _pK_to_K_coef(vpart):
    """
    Computes sum_{tau >= rho; sigma /\\ tau <= rho} mu(tau, 1)
    where mu is the Moebius function on the partition lattice.
    We think of (rho, sigma) as the diagram (i.e. vector partition) vpart in the usual way.
    """
    def is_disconnected(tau):
        for mblock in tau:
            for i in range(len(vpart[0])):
                if len([j for j in mblock if vpart[j][i] > 0]) > 1:
                    return False
        return True
    ret = 0
    for tau in set_partitions(len(vpart)):
        if is_disconnected(tau):
            ret += (-1) ** (len(tau) - 1) * math.factorial(len(tau) - 1)
    return ret


@flop_name('pK_to_K')
def DS_pK_to_K(pK, strict=True):
    """
    Does the pK -> K conversion directly.

    Composing the pK -> M and M -> K formulae yields:
        pK[X_{i_1},...,X_{i_d}] = sum_{rho} (prod_{B in rho} pK[X_{i_B}])
            * (sum_{tau >= rho; sigma /\\ tau <= rho} mu(tau, 1))
    where sigma is the type of the partition (i_1, ..., i_d), and mu is the
    Moebius function on the partition lattice.
    """
    return DS_part_sum(pK, _pK_to_K_coef, strict=strict)
