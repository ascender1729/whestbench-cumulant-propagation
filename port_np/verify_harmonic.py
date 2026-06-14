"""Verify port_np.harmonic_np (numpy) against mlp_kprop.harmonic (torch).

Run from repo root: uv run python port_np/verify_harmonic.py
Exercises the API surface used downstream by kprop_harmonic.py, factor_k3.py,
factor_k4.py:
  proj_coef, lap, rad (None/vector/matrix/scalar metric), compose,
  _multigraph_coef, _lap_m_prod_einexpr, _lap_m_prod, _lap_m_dslice,
  _harmonic_diagslice_einexpr, harmonic_diagslice (identity/vector/full metric),
  HTensor (construct, d/s/shape/ndim, to_tensor, get_dslice incl. unsorted
  parts + cache invalidation on core mutation, clone, to, repr,
  has_identity_metric, contract_W method), contract_W (default metric update,
  set_metric vector/scalar, NotImplementedError on non-identity metric),
  contract_W_proj (all valid r_out incl. non-identity-metric rejection),
  proj_geq_r (all valid r_out, core + to_tensor round trip),
  DS_harmonic_proj (geq True/False, torch DSTensor vs port DSTensor), HTower.
Adapted from tests/test_harmonic.py. ASCII only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from itertools import product

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
np.seterr(all="ignore")

import mlp_kprop.harmonic as T
import mlp_kprop.diagslice as TD
import port_np.harmonic_np as N
import port_np.diagslice_np as ND
from port_np.partitions_np import int_partitions, multigraphs
from port_np.tensor_utils_np import symmetrize as np_symmetrize
from port_np.tensor_utils_np import contract_W_basic as np_cwb

rng = np.random.default_rng(20260612)

MAXERR = 0.0
SECTION_ERR = {}
FAILED_CHECKS = []


def chk(section, name, t_val, n_val):
    global MAXERR
    if isinstance(t_val, torch.Tensor):
        t_arr = t_val.numpy()
    else:
        t_arr = np.asarray(t_val, dtype=np.float64)
    n_arr = np.asarray(n_val, dtype=np.float64)
    if t_arr.shape != n_arr.shape:
        FAILED_CHECKS.append(f"{section}/{name}: shape {t_arr.shape} vs {n_arr.shape}")
        SECTION_ERR[section] = float("inf")
        return
    err = float(np.max(np.abs(t_arr - n_arr))) if t_arr.size else 0.0
    SECTION_ERR[section] = max(SECTION_ERR.get(section, 0.0), err)
    MAXERR = max(MAXERR, err)
    if err > 1e-10:
        FAILED_CHECKS.append(f"{section}/{name}: err={err:.3e}")


def chk_eq(section, name, t_val, n_val):
    if t_val != n_val:
        FAILED_CHECKS.append(f"{section}/{name}: {t_val!r} != {n_val!r}")
        SECTION_ERR[section] = float("inf")
    else:
        SECTION_ERR.setdefault(section, 0.0)


def sym_rand(shape):
    return np_symmetrize(rng.standard_normal(shape) if shape else rng.standard_normal())


def t(x):
    return torch.tensor(np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------- proj_coef
for n in [4, 6]:
    for d in range(1, 7):
        for r in range(d // 2 + 1):
            chk("proj_coef", f"n{n}_d{d}_r{r}", T.proj_coef(n, d, r), N.proj_coef(n, d, r))

# ---------------------------------------------------------------- lap / rad
for n in [4, 6]:
    for d in range(0, 5):
        A = sym_rand((n,) * d)
        chk("lap", f"n{n}_d{d}", T.lap(t(A), strict=True), N.lap(A, strict=True))
        # rad: identity, vector, full, scalar metrics
        chk("rad", f"id_n{n}_d{d}", T.rad(t(A), n=n, strict=True), N.rad(A, n=n, strict=True))
        mv = rng.random(n) + 0.5
        chk("rad", f"vec_n{n}_d{d}",
            T.rad(t(A), n=n, metric=t(mv), strict=True),
            N.rad(A, n=n, metric=mv, strict=True))
        mf = np_symmetrize(rng.standard_normal((n, n)))
        chk("rad", f"full_n{n}_d{d}",
            T.rad(t(A), n=n, metric=t(mf), strict=True),
            N.rad(A, n=n, metric=mf, strict=True))
        chk("rad", f"scalar_n{n}_d{d}",
            T.rad(t(A), n=n, metric=torch.tensor(0.7), strict=True),
            N.rad(A, n=n, metric=0.7, strict=True))

# --------------------------------------- _multigraph_coef / _lap_m_prod_einexpr
for aritys in [(2,), (2, 2), (3, 2), (1, 1, 1), (3, 3, 2)]:
    for m in range(0, 3):
        for graph in multigraphs(len(aritys), m):
            chk("multigraph_coef", f"{aritys}_{m}_{graph}",
                np.float64(T._multigraph_coef(graph, list(aritys))),
                np.float64(N._multigraph_coef(graph, list(aritys))))
            chk("multigraph_coef", f"nolap_{aritys}_{m}_{graph}",
                np.float64(T._multigraph_coef(graph, list(aritys), lap_coef=False)),
                np.float64(N._multigraph_coef(graph, list(aritys), lap_coef=False)))
            chk_eq("lap_m_prod_einexpr", f"{aritys}_{m}_{graph}",
                   T._lap_m_prod_einexpr(graph, list(aritys)),
                   N._lap_m_prod_einexpr(graph, list(aritys)))

# ---------------------------------------------------------------- _lap_m_prod
n = 5
for aritys, m in [
    ((1,), 0), ((2,), 0), ((1,), 1), ((2,), 2), ((1,), 3), ((2,), 3),
    ((0,), 0), ((0, 2), 1),
    ((2, 2), 0), ((2, 2), 1), ((2, 3), 0), ((2, 3), 1),
    ((3, 3), 0), ((3, 3), 1), ((1, 1, 1), 2), ((2, 3, 2), 1), ((3, 3, 3), 1),
]:
    As = [sym_rand((n,) * a) for a in aritys]
    chk("lap_m_prod", f"{aritys}_{m}",
        T._lap_m_prod(m, [t(A) for A in As], strict=True),
        N._lap_m_prod(m, As, strict=True))

# --------------------------------------------------------------- _lap_m_dslice
n = 6
for part, m in product(
    [(1, 1, 1), (2, 1), (3,), (2, 2), (3, 1), (4,), (1,), (2,)],
    range(0, 4),
):
    if 2 * m > sum(part) + 2:  # allow some over-capped cases too
        continue
    if sum(part) - 2 * m < 0:
        continue
    ds = ND.zero_repeated(rng.standard_normal((n,) * len(part)))
    chk("lap_m_dslice", f"{part}_{m}",
        T._lap_m_dslice(m, t(ds), part),
        N._lap_m_dslice(m, ds, part))

# ------------------------------------------------------------------ proj_geq_r
for n in [4, 6]:
    for d in range(1, 5):
        A = sym_rand((n,) * d)
        for r_out in range(d // 2 + 1):
            ht = T.proj_geq_r(t(A), n=n, r_out=r_out, strict=True)
            hn = N.proj_geq_r(A, n=n, r_out=r_out, strict=True)
            chk_eq("proj_geq_r", f"r_n{n}_d{d}_{r_out}", ht.r, hn.r)
            chk_eq("proj_geq_r", f"d_n{n}_d{d}_{r_out}", ht.d, hn.d)
            chk("proj_geq_r", f"core_n{n}_d{d}_{r_out}", ht.core, hn.core)
            chk("proj_geq_r", f"tot_n{n}_d{d}_{r_out}",
                ht.to_tensor(strict=True), hn.to_tensor(strict=True))
            chk("proj_geq_r", f"metric_n{n}_d{d}_{r_out}", ht.metric, hn.metric)

# ------------------------------------------ HTensor: to_tensor, dslices, misc
for n in [4, 6]:
    for d in range(1, 5):
        for r in range(d // 2 + 1):
            s = d - 2 * r
            core = sym_rand((n,) * s)
            mv = rng.random(n) + 0.5
            W = rng.standard_normal((n, n))
            mf = W @ W.T
            metrics = [("id", None), ("vec", mv), ("full", mf), ("scalar", 0.5)]
            for mname, metric in metrics:
                ht = T.HTensor(t(core), r=r, n=n,
                               metric=None if metric is None else t(np.asarray(metric)),
                               strict=True)
                hn = N.HTensor(core, r=r, n=n, metric=metric, strict=True)
                tag = f"n{n}_d{d}_r{r}_{mname}"
                chk_eq("htensor", f"hasid_{tag}",
                       ht.has_identity_metric(), hn.has_identity_metric())
                chk_eq("htensor", f"s_{tag}", ht.s, hn.s)
                chk_eq("htensor", f"shape_{tag}", tuple(ht.shape), tuple(hn.shape))
                chk_eq("htensor", f"ndim_{tag}", ht.ndim, hn.ndim)
                chk("htensor", f"tot_{tag}",
                    ht.to_tensor(strict=True), hn.to_tensor(strict=True))
                chk("htensor", f"clone_{tag}",
                    ht.clone().to_tensor(), hn.clone().to_tensor())
                chk("htensor", f"metric_{tag}", ht.metric, hn.metric)
                repr(hn)  # smoke
                # dslices for every integer partition of d, incl. unsorted orders
                for ipart in int_partitions(d):
                    chk("htensor_dslice", f"{tag}_{ipart}",
                        ht.get_dslice(ipart), hn.get_dslice(ipart))
                    rev = tuple(reversed(ipart))
                    chk("htensor_dslice", f"{tag}_{rev}",
                        ht.get_dslice(rev), hn.get_dslice(rev))
                    # direct harmonic_diagslice on sorted part
                    spart = tuple(sorted(ipart, reverse=True))
                    chk("harmonic_diagslice", f"{tag}_{spart}",
                        T.harmonic_diagslice(ht, spart),
                        N.harmonic_diagslice(hn, spart))

# ------------------------------------------------ dslice cache invalidation
n = 6
core = sym_rand((n, n))
ht = T.HTensor(t(core), r=0, n=n, strict=True)
hn = N.HTensor(core.copy(), r=0, n=n, strict=True)
before_t = ht.get_dslice((2,)).clone()
before_n = hn.get_dslice((2,)).copy()
chk("cache_inval", "before", before_t, before_n)
ht.core += 1
hn.core += 1
chk("cache_inval", "after_setattr", ht.get_dslice((2,)), hn.get_dslice((2,)))
# pure in-place element mutation (no attribute rebind)
ht.core[0, 0] += 3.0
ht.core[1, 1] += 3.0  # keep symmetric-ish on diagonal
hn.core[0, 0] += 3.0
hn.core[1, 1] += 3.0
chk("cache_inval", "after_inplace", ht.get_dslice((2,)), hn.get_dslice((2,)))
chk("cache_inval", "recompute",
    T.harmonic_diagslice(ht, (2,)), hn.get_dslice((2,)))

# ---------------------------------------------------------------- contract_W
for n_in, n_out in [(5, 4), (4, 6)]:
    W = rng.standard_normal((n_out, n_in))
    for d_core, r in [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0)]:
        core = sym_rand((n_in,) * d_core)
        mv = rng.random(n_in) + 0.5
        mf = np_symmetrize(rng.standard_normal((n_in, n_in)))
        for mname, metric in [("id", None), ("vec", mv), ("full", mf)]:
            ht = T.HTensor(t(core), r=r, n=n_in,
                           metric=None if metric is None else t(metric), strict=True)
            hn = N.HTensor(core, r=r, n=n_in, metric=metric, strict=True)
            wt = T.contract_W(ht, t(W))
            wn = N.contract_W(hn, W)
            tag = f"{n_in}to{n_out}_s{d_core}_r{r}_{mname}"
            chk("contract_W", f"core_{tag}", wt.core, wn.core)
            chk("contract_W", f"metric_{tag}", wt.metric, wn.metric)
            chk_eq("contract_W", f"r_{tag}", wt.r, wn.r)
            chk_eq("contract_W", f"n_{tag}", wt.n, wn.n)
            # method form
            wn2 = hn.contract_W(W)
            chk("contract_W", f"method_{tag}", wt.core, wn2.core)
        # set_metric on identity-metric HTensor: vector and scalar
        ht = T.HTensor(t(core), r=r, n=n_in, strict=True)
        hn = N.HTensor(core, r=r, n=n_in, strict=True)
        sv = rng.random(n_out) + 0.5
        wt = T.contract_W(ht, t(W), set_metric=t(sv))
        wn = N.contract_W(hn, W, set_metric=sv)
        chk("contract_W", f"setm_vec_core_{n_in}_{d_core}_{r}", wt.core, wn.core)
        chk("contract_W", f"setm_vec_metric_{n_in}_{d_core}_{r}", wt.metric, wn.metric)
        wt = T.contract_W(ht, t(W), set_metric=torch.tensor(2.0 / n_out))
        wn = N.contract_W(hn, W, set_metric=2.0 / n_out)
        chk("contract_W", f"setm_sc_metric_{n_in}_{d_core}_{r}", wt.metric, wn.metric)
        # set_metric must reject non-identity metric on both sides
        ht_bad = T.HTensor(t(core), r=r, n=n_in, metric=t(mv), strict=True)
        hn_bad = N.HTensor(core, r=r, n=n_in, metric=mv, strict=True)
        t_raised = n_raised = False
        try:
            T.contract_W(ht_bad, t(W), set_metric=t(sv))
        except NotImplementedError:
            t_raised = True
        try:
            N.contract_W(hn_bad, W, set_metric=sv)
        except NotImplementedError:
            n_raised = True
        chk_eq("contract_W", f"setm_reject_{n_in}_{d_core}_{r}", t_raised, n_raised)

# ------------------------------------------------------------- contract_W_proj
n = 5
W = rng.standard_normal((n, n))
for s_in, r_in in product(range(0, 4), range(0, 3)):
    core = sym_rand((n,) * s_in)
    d = s_in + 2 * r_in
    for r_out in range(d // 2 + 1):
        ht = T.contract_W_proj(T.HTensor(t(core), r=r_in, n=n, strict=True),
                               t(W), r_out=r_out, strict=True)
        hn = N.contract_W_proj(N.HTensor(core, r=r_in, n=n, strict=True),
                               W, r_out=r_out, strict=True)
        chk("contract_W_proj", f"s{s_in}_r{r_in}_ro{r_out}", ht.core, hn.core)
        chk_eq("contract_W_proj", f"r_s{s_in}_r{r_in}_ro{r_out}", ht.r, hn.r)
# non-identity metric must be rejected on both sides
core = sym_rand((n, n))
mv = rng.random(n) + 0.5
t_raised = n_raised = False
try:
    T.contract_W_proj(T.HTensor(t(core), r=1, n=n, metric=t(mv), strict=True),
                      t(W), r_out=0, strict=True)
except NotImplementedError:
    t_raised = True
try:
    N.contract_W_proj(N.HTensor(core, r=1, n=n, metric=mv, strict=True),
                      W, r_out=0, strict=True)
except NotImplementedError:
    n_raised = True
chk_eq("contract_W_proj", "reject_metric", t_raised, n_raised)

# ------------------------------------------------------------ DS_harmonic_proj
for n in [4, 6]:
    for d in range(1, 5):
        A = sym_rand((n,) * d)
        dsa_t = TD.DSTensor.from_tensor(t(A))
        dsa_n = ND.DSTensor.from_tensor(A)
        for r_out in range(d // 2 + 1):
            for geq in [True, False]:
                ht = T.DS_harmonic_proj(dsa_t, r_out=r_out, geq=geq, strict=True)
                hn = N.DS_harmonic_proj(dsa_n, r_out=r_out, geq=geq, strict=True)
                tag = f"n{n}_d{d}_ro{r_out}_geq{geq}"
                chk("DS_harmonic_proj", f"core_{tag}", ht.core, hn.core)
                chk("DS_harmonic_proj", f"tot_{tag}", ht.to_tensor(), hn.to_tensor())
                chk_eq("DS_harmonic_proj", f"r_{tag}", ht.r, hn.r)

# ---------------------------------------------- HTower alias smoke + report
tower_n = N.HTower()
tower_n[2] = N.HTensor(sym_rand((4, 4)), r=0, n=4, strict=True)
assert isinstance(tower_n, dict)

print("Per-section max abs error (inf = mismatch/shape/eq failure):")
for sec in sorted(SECTION_ERR):
    print(f"  {sec:24s} {SECTION_ERR[sec]:.3e}")
if FAILED_CHECKS:
    print(f"{len(FAILED_CHECKS)} failed checks (first 20):")
    for f in FAILED_CHECKS[:20]:
        print("  " + f)
print(f"MAXERR {MAXERR:.3e}")
print("PASS" if MAXERR < 1e-10 and not FAILED_CHECKS else "FAIL")
