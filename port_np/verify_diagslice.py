"""Verify port_np.diagslice_np (numpy) against mlp_kprop.diagslice (torch).

Run from repo root: uv run python port_np/verify_diagslice.py
Exercises the full public API surface used downstream by cumulants.py,
harmonic.py, factor_k3.py, factor_k4.py, kprop_ds.py, kprop_harmonic.py:
  _zero_repeated/zero_repeated, _diagslice_view, _einsum_delta, _merge_legs,
  EinsumCond(+get_parts), DSTensor (from_tensor, to_tensor, get_slice/get_dslice,
  has_slice, clone, prune, item, einsum incl. ein_cond/out_cond/in_symmetric/
  return_part_contributions, arithmetic incl. inplace and coercion, clamp),
  DSTower (from_tower, from_slices, get_slice, arithmetic, clone, coerce, prune,
  is_downward_closed, to_tower), diagslice (tensor + DSTensor paths),
  expand_dslice, eval_part.
ASCII only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itertools import permutations, product

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
np.seterr(all="ignore")

import mlp_kprop.diagslice as T  # torch reference
import port_np.diagslice_np as N  # numpy port
from mlp_kprop.partitions import (
    IntPartCond,
    int_partitions,
    set_partitions,
    set_to_int_partition,
    vector_partitions,
    weak_compositions,
)
from port_np.tensor_utils_np import symmetrize as np_symmetrize

MAXERR = 0.0
SEC = {}


def rec(section, t, a):
    global MAXERR
    if isinstance(t, torch.Tensor):
        t = t.detach().numpy()
    t = np.asarray(t, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    assert t.shape == a.shape, f"{section}: shape mismatch {t.shape} vs {a.shape}"
    if t.size == 0:
        return
    both_nan = np.isnan(t) & np.isnan(a)
    d = np.abs(t - a)
    d = np.where(both_nan, 0.0, d)
    assert not np.isnan(d).any(), f"{section}: one-sided nan"
    err = float(d.max())
    SEC[section] = max(SEC.get(section, 0.0), err)
    MAXERR = max(MAXERR, err)


def cmp_ds(section, tds, nds, check_full=True):
    assert isinstance(nds, N.DSTensor), f"{section}: not a DSTensor port object"
    assert set(tds.slices.keys()) == set(nds.slices.keys()), (
        f"{section}: slice keys {set(tds.slices.keys())} vs {set(nds.slices.keys())}"
    )
    assert tds.n == nds.n and tds.d == nds.d, f"{section}: n/d mismatch"
    for part in tds.slices:
        rec(section, tds.slices[part], nds.slices[part])
    if check_full:
        rec(section + ".to_tensor", tds.to_tensor(), nds.to_tensor())


def cmp_tower(section, tt, nt):
    assert set(tt.keys()) == set(nt.keys()), f"{section}: degree keys mismatch"
    for deg in tt.keys():
        cmp_ds(f"{section}[{deg}]", tt[deg], nt[deg])


def sym(rng, n, d):
    return np.ascontiguousarray(np_symmetrize(rng.standard_normal((n,) * d)))


rng = np.random.default_rng(20260612)

# ---------------------------------------------------------------------------
# 1. Core slicing per (n, d)
# ---------------------------------------------------------------------------
for n in [4, 6]:
    for d in [1, 2, 3, 4]:
        A = sym(rng, n, d)
        At = torch.tensor(A)

        rec("zero_repeated", T.zero_repeated(At), N.zero_repeated(A))

        for sp_ in set_partitions(d):
            rec(
                "_diagslice_view",
                T._diagslice_view(At, sp_).clone(),
                N._diagslice_view(A, sp_).copy(),
            )

        # in-place view write semantics
        Bt, Bn = At.clone(), A.copy()
        sp0 = set_partitions(d)[0]
        T._diagslice_view(Bt, sp0).zero_()
        N._diagslice_view(Bn, sp0)[...] = 0.0
        rec("_diagslice_view.write", Bt, Bn)

        tds = T.DSTensor.from_tensor(At)
        nds = N.DSTensor.from_tensor(A)
        cmp_ds("from_tensor", tds, nds)
        rec("roundtrip_vs_input", torch.tensor(A), nds.to_tensor())

        # clone + prune
        cmp_ds("clone", tds.clone(), nds.clone())
        cmp_ds("prune", tds.clone().prune(), nds.clone().prune())

        # get_slice over all permutations of all partitions + has_slice
        for ip in int_partitions(d):
            assert tds.has_slice(ip) == nds.has_slice(ip)
            for pp in set(permutations(ip)):
                rec("get_slice", tds.get_slice(pp), nds.get_slice(pp))
                rec("get_dslice", tds.get_dslice(pp), nds.get_dslice(pp))

        # filtered from_tensor + non-strict get_slice
        if d >= 2:
            pc = lambda part: min(part) > 1
            tdf = T.DSTensor.from_tensor(At, part_cond=pc)
            ndf = N.DSTensor.from_tensor(A, part_cond=pc)
            cmp_ds("from_tensor_filtered", tdf, ndf)
            for ip in int_partitions(d):
                rec(
                    "get_slice_nonstrict",
                    tdf.get_slice(ip, strict=False),
                    ndf.get_slice(ip, strict=False),
                )
            # strict failure parity
            missing = [ip for ip in int_partitions(d) if min(ip) == 1]
            if missing:
                t_raised = n_raised = False
                try:
                    tdf.get_slice(missing[0])
                except AssertionError:
                    t_raised = True
                try:
                    ndf.get_slice(missing[0])
                except AssertionError:
                    n_raised = True
                assert t_raised and n_raised, "strict get_slice should raise on both sides"

        # diagslice: tensor path and DSTensor path (test_upslice analog)
        for ip in int_partitions(d):
            for pp in set(permutations(ip)):
                for ozr in [False, True]:
                    rec(
                        "diagslice_tensor",
                        T.diagslice(At, pp, output_zero_repeated=ozr),
                        N.diagslice(A, pp, output_zero_repeated=ozr),
                    )
                    rec(
                        "diagslice_ds",
                        T.diagslice(tds, pp, output_zero_repeated=ozr),
                        N.diagslice(nds, pp, output_zero_repeated=ozr),
                    )

        # expand_dslice over all length-3 weak compositions of d
        for vec in weak_compositions(3, d):
            if all(v == 0 for v in vec):
                continue
            for ozr in [True, False]:
                rec(
                    "expand_dslice_tensor",
                    T.expand_dslice(At, vec, output_zero_repeated=ozr),
                    N.expand_dslice(A, vec, output_zero_repeated=ozr),
                )
                rec(
                    "expand_dslice_ds",
                    T.expand_dslice(tds, vec, output_zero_repeated=ozr),
                    N.expand_dslice(nds, vec, output_zero_repeated=ozr),
                )

        # to(dtype) round trip
        cmp_ds("to_dtype", tds.to(device="cpu", dtype=torch.float64), nds.to(dtype=np.float64))

# ---------------------------------------------------------------------------
# 2. Arithmetic ops
# ---------------------------------------------------------------------------
for n, d in [(4, 2), (4, 3), (6, 2)]:
    A = sym(rng, n, d)
    B = sym(rng, n, d)
    D = np.ascontiguousarray(np_symmetrize(rng.uniform(1.0, 2.0, (n,) * d)))  # divisor away from 0
    At, Bt, Dt = torch.tensor(A), torch.tensor(B), torch.tensor(D)
    tA, tB, tD = (T.DSTensor.from_tensor(x) for x in (At, Bt, Dt))
    nA, nB, nD = (N.DSTensor.from_tensor(x) for x in (A, B, D))

    cmp_ds("op_add", tA + tB, nA + nB)
    cmp_ds("op_sub", tA - tB, nA - nB)
    cmp_ds("op_mul", tA * tB, nA * nB)
    cmp_ds("op_div", tA / tD, nA / nD)
    cmp_ds("op_add_scalar", tA + 2.5, nA + 2.5)
    cmp_ds("op_radd_scalar", 2.5 + tA, 2.5 + nA)
    cmp_ds("op_rsub_scalar", 1.0 - tA, 1.0 - nA)
    cmp_ds("op_div_scalar", tA / 2.0, nA / 2.0)
    cmp_ds("op_pow_scalar", tA ** 2, nA ** 2)
    cmp_ds("op_neg", -tA, -nA)
    cmp_ds("op_abs", abs(tA), abs(nA))
    cmp_ds("op_clamp", tA.clamp(min=-0.5, max=0.5), nA.clamp(min=-0.5, max=0.5))
    cmp_ds("op_coerce_tensor", tA + Bt, nA + B)
    cmp_ds("op_rcoerce_tensor", tA.__radd__(Bt), nA.__radd__(B))

    ti, ni = tA.clone(), nA.clone()
    ti += tB
    ni += nB
    cmp_ds("op_iadd", ti, ni)
    ti, ni = tA.clone(), nA.clone()
    ti *= Bt
    ni *= B
    cmp_ds("op_imul_tensor", ti, ni)
    ti, ni = tA.clone(), nA.clone()
    ti -= 0.25
    ni -= 0.25
    cmp_ds("op_isub_scalar", ti, ni)

# item()
v = rng.standard_normal((1,))
tv = T.DSTensor.from_tensor(torch.tensor(v))
nv = N.DSTensor.from_tensor(v)
rec("item", np.array(tv.item()), np.array(nv.item()))

# ---------------------------------------------------------------------------
# 3. _einsum_delta and _merge_legs
# ---------------------------------------------------------------------------
n = 4
Ae = rng.standard_normal((n, n))
Be = rng.standard_normal((n, n))
Aet, Bet = torch.tensor(Ae), torch.tensor(Be)
for expr in [
    "a b, b c -> a c",
    "a b, b c -> a a",
    "a b, b c -> a c c",
    "a b, c d -> a a b b b c d d",
]:
    rec("_einsum_delta", T._einsum_delta(Aet, Bet, expr), N._einsum_delta(Ae, Be, expr))
ve = rng.standard_normal(n)
vet = torch.tensor(ve)
for dd in range(2, 6):
    expr = "a -> " + " ".join(("a",) * dd)
    rec("_einsum_delta_diag", T._einsum_delta(vet, expr), N._einsum_delta(ve, expr))


def sp(part):
    return tuple(frozenset(block) for block in part)


def sps(*parts):
    return tuple(sp(part) for part in parts)


merge_cases = [
    ("a b c -> a b c", sps(((0,), (1,), (2,)), ((0,), (1,), (2,)))),
    ("a b c -> a b c", sps(((0, 1), (2,)), ((0, 1), (2,)))),
    ("a b c -> a b c", sps(((0, 1), (2,)), ((0,), (1, 2)))),
    ("a b c -> a b c", sps(((0, 1, 2),), ((0,), (1,), (2,)))),
    ("a b, b c -> a c", sps(((0,), (1,)), ((0,), (1,)), ((0,), (1,)))),
    ("a b, b c -> a c", sps(((0, 1),), ((0,), (1,)), ((0,), (1,)))),
    ("a b, b c -> a c", sps(((0, 1),), ((0, 1),), ((0,), (1,)))),
    (
        "a b c d, a i, b j, c k, d l -> i j k l",
        sps(((0, 1), (2, 3)), ((0,), (1,)), ((0,), (1,)), ((0,), (1,)), ((0,), (1,)), ((0, 1), (2, 3))),
    ),
    (
        "a b c d, a i, b j, c k, d l -> i j k l",
        sps(((0, 1), (2, 3)), ((0,), (1,)), ((0,), (1,)), ((0,), (1,)), ((0,), (1,)), ((0, 2), (1, 3))),
    ),
]
for einexpr, parts in merge_cases:
    assert T._merge_legs(einexpr, parts) == N._merge_legs(einexpr, parts), (
        f"_merge_legs mismatch for {einexpr} {parts}"
    )

# ---------------------------------------------------------------------------
# 4. EinsumCond.get_parts parity
# ---------------------------------------------------------------------------
for aritys, is_dstensor, out_symmetric in [
    ((2, 2), (True,), True),
    ((2, 2), (True,), False),
    ((2, 2), (False,), False),
    ((3, 3, 3), (True, True), False),
    ((3, 3, 3), (True, True), True),
    ((3, 4, 3), (True, False), True),
    ((3, 4, 3), (False, True), True),
]:
    input_iparts = [
        int_partitions(aritys[i]) if b else ((1,) * aritys[i],) for i, b in enumerate(is_dstensor)
    ]
    t_conds = tuple(IntPartCond(parts=set(ip)) for ip in input_iparts) + (
        T.trivial_int_cond,
    )
    n_conds = tuple(IntPartCond(parts=set(ip)) for ip in input_iparts) + (
        N.trivial_int_cond,
    )
    for in_symmetric in [False, True]:
        t_ec = T.EinsumCond(einsum_cond=lambda parts: True)
        n_ec = N.EinsumCond(einsum_cond=lambda parts: True)
        t_ret = t_ec.get_parts(
            aritys=tuple(aritys), arg_conds=t_conds,
            out_symmetric=out_symmetric, in_symmetric=in_symmetric,
        )
        n_ret = n_ec.get_parts(
            aritys=tuple(aritys), arg_conds=n_conds,
            out_symmetric=out_symmetric, in_symmetric=in_symmetric,
        )
        assert [p for p, c in t_ret] == [p for p, c in n_ret], "EinsumCond parts mismatch"
        for (tp, tc), (np_, nc) in zip(t_ret, n_ret):
            rec("einsumcond_coefs", np.array(float(tc)), np.array(float(nc)))

# ---------------------------------------------------------------------------
# 5. DSTensor.einsum
# ---------------------------------------------------------------------------
n = 4
mats = []
for _ in range(3):
    M = rng.standard_normal((n, n))
    mats.append(np.ascontiguousarray(M + M.T))
A2, B2, C2 = mats
X, Y = rng.standard_normal((n, n)), rng.standard_normal((n, n))
D3 = np.einsum("ia,ib,ic->abc", X, X, X)
E3 = np.einsum("ia,ib,ic->abc", Y, Y, Y)
exprs = [
    ((A2, A2, A2), "a b, b c, c d -> a d"),
    ((A2, B2, C2), "a b, a b, a b -> a b"),
    ((A2, A2, A2), "a b, a c, a d -> b c d"),
    ((D3, A2, A2, A2), "a b c, a d, b e, c f -> d e f"),
    ((D3, E3), "a b c, a b c -> a b c"),
]
for inputs, expr in exprs:
    t_inputs = [torch.tensor(x) for x in inputs]
    t_ds = [T.DSTensor.from_tensor(x) for x in t_inputs]
    n_ds = [N.DSTensor.from_tensor(x) for x in inputs]
    for flags in product([False, True], repeat=len(inputs)):
        t_args = [t_ds[i] if f else t_inputs[i] for i, f in enumerate(flags)]
        n_args = [n_ds[i] if f else inputs[i] for i, f in enumerate(flags)]
        t_ret = T.DSTensor.einsum(*t_args, expr)
        n_ret = N.DSTensor.einsum(*n_args, expr)
        cmp_ds("einsum", t_ret, n_ret)

# linear_kprop pattern: einsum(K, W, ..., W)
m = 3
W = rng.standard_normal((n, m))
Wt = torch.tensor(W)
for dd in range(1, 5):
    K = sym(rng, n, dd)
    Kt = torch.tensor(K)
    expr = " ".join(f"i{i}" for i in range(dd))
    expr += ", " + ", ".join(f"i{i} j{i}" for i in range(dd))
    expr += " -> " + " ".join(f"j{i}" for i in range(dd))
    for in_symmetric in [False, True]:
        t_ret = T.DSTensor.einsum(T.DSTensor.from_tensor(Kt), *(Wt,) * dd, expr, in_symmetric=in_symmetric)
        n_ret = N.DSTensor.einsum(N.DSTensor.from_tensor(K), *(W,) * dd, expr, in_symmetric=in_symmetric)
        cmp_ds("einsum_linear", t_ret, n_ret)

# return_part_contributions
K2 = sym(rng, 3, 2)
W2 = rng.standard_normal((2, 3))
expr = "i0 i1, j0 i0, j1 i1 -> j0 j1"
t_ret, t_contribs = T.DSTensor.einsum(
    T.DSTensor.from_tensor(torch.tensor(K2)), torch.tensor(W2), torch.tensor(W2), expr,
    return_part_contributions=True,
)
n_ret, n_contribs = N.DSTensor.einsum(
    N.DSTensor.from_tensor(K2), W2, W2, expr, return_part_contributions=True
)
cmp_ds("einsum_contrib_ret", t_ret, n_ret)
assert set(t_contribs.keys()) == set(n_contribs.keys())
for out_part in t_contribs:
    assert set(t_contribs[out_part].keys()) == set(n_contribs[out_part].keys())
    for in_parts in t_contribs[out_part]:
        rec("einsum_contribs", t_contribs[out_part][in_parts], n_contribs[out_part][in_parts])

# ein_cond / out_cond filtering (test_einsumcond2 analog), d=4
d4, n4 = 4, 4
A4 = sym(rng, n4, d4)
A4t = torch.tensor(A4)
int_parts = list(int_partitions(d4))[:3]
set_parts = [p for p in set_partitions(d4) if set_to_int_partition(p) in int_parts]
t_dsA = T.DSTensor.from_tensor(A4t)
n_dsA = N.DSTensor.from_tensor(A4)
args = " ".join(f"i{i}" for i in range(d4))
expr = f"{args} -> {args}"

cmp_ds("einsum_identity", T.DSTensor.einsum(t_dsA, expr), N.DSTensor.einsum(n_dsA, expr))
cmp_ds(
    "einsum_outcond",
    T.DSTensor.einsum(t_dsA, expr, out_cond=IntPartCond(parts=int_parts)),
    N.DSTensor.einsum(n_dsA, expr, out_cond=IntPartCond(parts=int_parts)),
)
cmp_ds(
    "einsum_eincond_out",
    T.DSTensor.einsum(t_dsA, expr, ein_cond=T.EinsumCond(einsum_cond=lambda parts: parts[-1] in set_parts)),
    N.DSTensor.einsum(n_dsA, expr, ein_cond=N.EinsumCond(einsum_cond=lambda parts: parts[-1] in set_parts)),
)
cmp_ds(
    "einsum_eincond_in",
    T.DSTensor.einsum(t_dsA, expr, ein_cond=T.EinsumCond(einsum_cond=lambda parts: parts[0] in set_parts)),
    N.DSTensor.einsum(n_dsA, expr, ein_cond=N.EinsumCond(einsum_cond=lambda parts: parts[0] in set_parts)),
)

# ---------------------------------------------------------------------------
# 6. eval_part
# ---------------------------------------------------------------------------
n5 = 5
tK, nK = {}, {}
for dd in [1, 2, 3]:
    Kd = sym(rng, n5, dd)
    tK[dd] = T.DSTensor.from_tensor(torch.tensor(Kd))
    nK[dd] = N.DSTensor.from_tensor(Kd)
for vp in vector_partitions((1, 1, 1)):
    for ozr in [True, False]:
        t_out = T.eval_part(tK, vp, 3, output_zero_repeated=ozr)
        n_out = N.eval_part(nK, vp, 3, output_zero_repeated=ozr)
        rec("eval_part", t_out, n_out)
for vp in vector_partitions((2, 1)):
    for ozr in [True, False]:
        t_out = T.eval_part(tK, vp, 2, output_zero_repeated=ozr)
        n_out = N.eval_part(nK, vp, 2, output_zero_repeated=ozr)
        if t_out is None or n_out is None:
            assert t_out is None and n_out is None
        else:
            rec("eval_part2", t_out, n_out)
# empty partition + missing-order None path
rec("eval_part_empty", T.eval_part(tK, (), 2), N.eval_part(nK, (), 2))
assert T.eval_part(tK, ((4, 0),), 2) is None and N.eval_part(nK, ((4, 0),), 2) is None

# ---------------------------------------------------------------------------
# 7. DSTower
# ---------------------------------------------------------------------------
v1 = np.array([1.0, 2.0])
m1 = np.array([[2.0, 0.5], [0.5, 3.0]])
v2 = np.array([3.0, 4.0])
c1 = np.ones((2, 2, 2))
t_t1 = T.DSTower.from_tower({1: torch.tensor(v1), 2: torch.tensor(m1)})
t_t2 = T.DSTower.from_tower({1: torch.tensor(v2), 3: torch.tensor(c1)})
n_t1 = N.DSTower.from_tower({1: v1, 2: m1})
n_t2 = N.DSTower.from_tower({1: v2, 3: c1})

cmp_tower("tower_from_tower", t_t1, n_t1)
cmp_tower("tower_add", t_t1 + t_t2, n_t1 + n_t2)
cmp_tower("tower_sub", t_t1 - t_t2, n_t1 - n_t2)
cmp_tower("tower_mul_scalar", t_t1 * 2, n_t1 * 2)
cmp_tower("tower_neg", -t_t2, -n_t2)
cmp_tower("tower_clone", t_t1.clone(), n_t1.clone())

t_i, n_i = t_t1.clone(), n_t1.clone()
t_i += t_t2
n_i += n_t2
cmp_tower("tower_iadd", t_i, n_i)

assert t_t1.is_downward_closed() == n_t1.is_downward_closed()
t_gap = T.DSTower.from_slices({(2, 1): torch.tensor(np_symmetrize(rng.standard_normal((2, 2))))}, autozero=True)
n_gap = N.DSTower.from_slices({(2, 1): np_symmetrize(rng.standard_normal((2, 2)))}, autozero=True)
assert t_gap.is_downward_closed() == n_gap.is_downward_closed() == False  # noqa: E712

slices_np = {(1,): np.array([1.0, 2.0]), (2,): np.array([3.0, 4.0])}
t_rebuilt = T.DSTower.from_slices({k: torch.tensor(v) for k, v in slices_np.items()}, autozero=True)
n_rebuilt = N.DSTower.from_slices({k: v.copy() for k, v in slices_np.items()}, autozero=True)
cmp_tower("tower_from_slices", t_rebuilt, n_rebuilt)

# tower get_slice (incl. unsorted partition) and to_tower
n6, d6 = 4, 3
KT = {dd: sym(rng, n6, dd) for dd in [1, 2, 3]}
t_tw = T.DSTower.from_tower({dd: torch.tensor(KT[dd]) for dd in KT})
n_tw = N.DSTower.from_tower(dict(KT))
for part in [(1,), (2,), (1, 1), (3,), (2, 1), (1, 2), (1, 1, 1)]:
    rec("tower_get_slice", t_tw.get_slice(part), n_tw.get_slice(part))
    rec("tower_get_dslice", t_tw.get_dslice(part), n_tw.get_dslice(part))
t_full = t_tw.to_tower()
n_full = n_tw.to_tower()
assert set(t_full.keys()) == set(n_full.keys())
for dd in t_full:
    rec("tower_to_tower", t_full[dd], n_full[dd])

# coerce with part_cond + prune
pc = lambda part: len(part) <= 2
cmp_tower(
    "tower_coerce",
    t_tw.coerce(part_cond=pc, prune=True),
    n_tw.coerce(part_cond=pc, prune=True),
)

# prune drops zero slices identically
t_z = T.DSTower.from_slices(
    {(1,): torch.tensor([0.0, 0.0]), (2,): torch.tensor([1.0, 2.0])}, autozero=True
).prune()
n_z = N.DSTower.from_slices(
    {(1,): np.array([0.0, 0.0]), (2,): np.array([1.0, 2.0])}, autozero=True
).prune()
assert set(t_z[1].slices.keys()) == set(n_z[1].slices.keys())
assert set(t_z[2].slices.keys()) == set(n_z[2].slices.keys())

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
for k in sorted(SEC):
    print(f"{k:32s} {SEC[k]:.3e}")
print(f"MAXERR {MAXERR:.6e}")
print("PASS" if MAXERR < 1e-10 else "FAIL")
