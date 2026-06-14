"""Verify port_np.factor_k3_np (+ port_np.kprop_terms_np) against torch
mlp_kprop.factor_k3 / mlp_kprop.kprop_harmonic.

Run from repo root: uv run python port_np/verify_factor_k3.py

Sections:
  0. term enumeration: get_all_terms_iso / get_int_cond / get_vec_cond /
     multiply_wicks vs torch reference
  1. FactoredTensor unit tests (to_tensor, get_dslice all order-3 partitions
     + permutations + higher-d parts from tests/test_factor_k3.py, contract_W,
     contract_wick incl. populated cache, add_factors, __add__, get_repeated,
     from_dstensor)
  2. integration: full kprop layer step at k_max=3 factor=True via torch
     nonlin_kprop(..., factor=True) vs port factored_nonlin_kprop_k3, for
     kind in {SIMPLE, AUGMENT, BASE} x use_avg_metric {True, False}
     x use_pK ({True, False} for BASE)
  3. multi-layer: 2 sequential linear+nonlin steps with He-scaled W (n=6),
     comparing all cumulants incl. final means (inside section 2 loop, depth=2)
ASCII only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from itertools import permutations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
np.seterr(all="ignore")

import mlp_kprop.factor_k3 as TF
import mlp_kprop.diagslice as TD
import mlp_kprop.kprop_harmonic as TKH
from mlp_kprop.kprop_harmonic import Kind, coerce_input, linear_kprop, nonlin_kprop
from mlp_kprop.wick import relu_wick_coef as t_relu_wick

import port_np.factor_k3_np as NF
import port_np.kprop_terms_np as NT
import port_np.diagslice_np as ND
import port_np.harmonic_np as NH
from port_np.wick_np import relu_wick_coef as n_relu_wick
from port_np.tensor_utils_np import symmetrize as n_symmetrize

rng = np.random.default_rng(20260612)

MAXERR = 0.0
SECTION_ERR = {}
FAILED = []


def rec(section, name, err):
    global MAXERR
    err = float(err)
    MAXERR = max(MAXERR, err)
    SECTION_ERR[section] = max(SECTION_ERR.get(section, 0.0), err)
    if err > 1e-10:
        FAILED.append(f"{section}/{name}: {err:.3e}")


def chk(section, name, t_val, n_val):
    if isinstance(t_val, torch.Tensor):
        t_arr = t_val.numpy()
    else:
        t_arr = np.asarray(t_val, dtype=np.float64)
    n_arr = np.asarray(n_val, dtype=np.float64)
    if t_arr.shape != n_arr.shape:
        FAILED.append(f"{section}/{name}: shape {t_arr.shape} vs {n_arr.shape}")
        SECTION_ERR[section] = float("inf")
        return
    err = float(np.max(np.abs(t_arr - n_arr))) if t_arr.size else 0.0
    rec(section, name, err)


def tt(a):
    return torch.tensor(np.asarray(a, dtype=np.float64))


# ----------------------------------------------------------------------
# Section 0: term enumeration helpers
# ----------------------------------------------------------------------
sec = "0-terms"
for k_max in [2, 3]:
    for d_max in [None, 3, 4]:
        t_terms = TKH.get_all_terms(k_max, d_max=d_max)
        n_terms = NT.get_all_terms(k_max, d_max=d_max)
        assert list(t_terms) == list(n_terms), f"get_all_terms mismatch k={k_max} d={d_max}"
        t_iso = TKH.get_all_terms_iso(k_max, d_max=d_max)
        n_iso = NT.get_all_terms_iso(k_max, d_max=d_max)
        assert t_iso == n_iso, f"get_all_terms_iso mismatch k={k_max} d={d_max}"
rec(sec, "enumeration-dicts", 0.0)

# get_int_cond / get_vec_cond agreement on sampled partitions
t_ic, n_ic = TKH.get_int_cond(3), NT.get_int_cond(3)
assert t_ic.get_parts(d_max=6) == n_ic.get_parts(d_max=6), "int_cond parts mismatch"
t_vc, n_vc = TKH.get_vec_cond(3), NT.get_vec_cond(3)
assert t_vc.get_parts(dim=3, sum_max=8) == n_vc.get_parts(dim=3, sum_max=8), "vec_cond parts mismatch"
rec(sec, "cond-parts", 0.0)

# multiply_wicks numeric check
n_ = 5
mean_np = rng.standard_normal(n_)
var_np = rng.uniform(0.2, 2.0, n_)
t_lookup = lambda k, p: t_relu_wick(tt(mean_np), tt(var_np), k, p)
n_lookup = lambda k, p: n_relu_wick(mean_np, var_np, k, p)
for kvec, pvec in [((1, 1), (1, 2)), ((2, 1, 1), (2, 1, 1)), ((1, 2, 1), (1, 1, 1))]:
    d = len(kvec)
    K_np = rng.standard_normal((n_,) * d)
    t_out = TKH.multiply_wicks(tt(K_np), kvec, pvec, t_lookup)
    n_out = NT.multiply_wicks(K_np, kvec, pvec, n_lookup)
    chk(sec, f"multiply_wicks{kvec}{pvec}", t_out, n_out)

# ----------------------------------------------------------------------
# Section 1: FactoredTensor unit tests
# ----------------------------------------------------------------------
sec = "1-FactoredTensor"
for n in [4, 6]:
    for r in [2, 5]:
        # --- order-3 build ---
        facs = [rng.standard_normal((n, r)) for _ in range(3)]
        tF = TF.FactoredTensor(n, 3, tuple(tt(f) for f in facs))
        nF = NF.FactoredTensor(n, 3, tuple(facs))
        chk(sec, f"to_tensor n{n} r{r}", tF.to_tensor(), nF.to_tensor())

        # get_dslice: all order-3 partitions (sorted + permutations), except all-1s
        parts3 = set()
        for base_part in [(3,), (2, 1)]:
            for p in permutations(base_part):
                parts3.add(p)
        for part in sorted(parts3):
            chk(sec, f"dslice{part} n{n} r{r}", tF.get_dslice(part), nF.get_dslice(part))
        # (1, 1, 1) must raise on both
        raised_t = raised_n = False
        try:
            TF._factored_get_dslice(tuple(tt(f) for f in facs), (1, 1, 1))
        except NotImplementedError:
            raised_t = True
        try:
            NF._factored_get_dslice(tuple(facs), (1, 1, 1))
        except NotImplementedError:
            raised_n = True
        assert raised_t and raised_n, "(1,1,1) dslice should raise on both sides"

        # higher-order parts from tests/test_factor_k3.py::test_dslice
        for part in [(2,), (2, 1), (2, 2, 1), (3,)]:
            d = sum(part)
            hfacs = [rng.standard_normal((n, r)) for _ in range(d)]
            tH = TF.FactoredTensor(n, d, tuple(tt(f) for f in hfacs))
            nH = NF.FactoredTensor(n, d, tuple(hfacs))
            chk(sec, f"dslice-hd{part} n{n} r{r}", tH.get_dslice(part), nH.get_dslice(part))
            # cross-check against zero_repeated(diagslice(to_tensor)) on port side
            ref = ND.zero_repeated(ND.diagslice(nH.to_tensor(), part))
            chk(sec, f"dslice-hd-ref{part} n{n} r{r}", ref, nH.get_dslice(part))

        # --- contract_W ---
        W = rng.standard_normal((n, n))
        tW = tF.contract_W(tt(W))
        nW = nF.contract_W(W)
        chk(sec, f"contract_W n{n} r{r}", tW.to_tensor(), nW.to_tensor())

        # --- contract_wick (cache populated by get_dslice above) ---
        wick = rng.standard_normal(n)
        tC = tF.contract_wick(tt(wick))
        nC = nF.contract_wick(wick)
        chk(sec, f"contract_wick n{n} r{r}", tC.to_tensor(), nC.to_tensor())
        for part in [(3,), (2, 1)]:
            chk(sec, f"contract_wick-cache{part} n{n} r{r}", tC.get_dslice(part), nC.get_dslice(part))

        # --- add_factors (cache update path) ---
        extra = [rng.standard_normal((n, 3)) for _ in range(3)]
        tA = tF.add_factors(tuple(tt(f) for f in extra))
        nA = nF.add_factors(tuple(extra))
        chk(sec, f"add_factors n{n} r{r}", tA.to_tensor(), nA.to_tensor())
        for part in [(3,), (2, 1)]:
            chk(sec, f"add_factors-cache{part} n{n} r{r}", tA.get_dslice(part), nA.get_dslice(part))

        # --- __add__ ---
        facs2 = [rng.standard_normal((n, 2)) for _ in range(3)]
        tG = TF.FactoredTensor(n, 3, tuple(tt(f) for f in facs2))
        nG = NF.FactoredTensor(n, 3, tuple(facs2))
        chk(sec, f"__add__ n{n} r{r}", (tF + tG).to_tensor(), (nF + nG).to_tensor())

        # --- get_repeated ---
        tR = tF.get_repeated()
        nR = nF.get_repeated()
        chk(sec, f"get_repeated n{n} r{r}", tR.to_tensor(), nR.to_tensor())
        chk(
            sec,
            f"get_repeated-identity n{n} r{r}",
            nF.to_tensor(),
            ND.zero_repeated(nF.to_tensor()) + nR.to_tensor(),
        )

# --- from_dstensor (tests/test_factor_k3.py::test_from_dstensor) ---
for n in [4, 6]:
    A = n_symmetrize(rng.standard_normal((n, n, n)))
    t_ds = TD.DSTensor.from_tensor(tt(A))
    t_ds.slices.pop((1, 1, 1))
    n_ds = ND.DSTensor.from_tensor(A)
    n_ds.slices.pop((1, 1, 1))
    tFD = TF.FactoredTensor.from_dstensor(t_ds)
    nFD = NF.FactoredTensor.from_dstensor(n_ds)
    chk(sec, f"from_dstensor n{n}", tFD.to_tensor(), nFD.to_tensor())
    chk(sec, f"from_dstensor-roundtrip n{n}", t_ds.to_tensor(), nFD.to_tensor())

# ----------------------------------------------------------------------
# Sections 2+3: integration -- full kprop chains, depth=2 (multi-layer)
# ----------------------------------------------------------------------
sec = "2-integration"


def np_linear(K, W, set_metric=None):
    """Port-side equivalent of linear_kprop for this chain (no bias)."""
    WK = {}
    for d, Kd in K.items():
        if isinstance(Kd, NH.HTensor):
            assert Kd.has_identity_metric()
            WK[d] = Kd.contract_W(W, set_metric=set_metric)
        else:
            WK[d] = Kd.contract_W(W)
    return WK


configs = []
for kind in [Kind.SIMPLE, Kind.AUGMENT, Kind.BASE]:
    for use_avg_metric in [True, False]:
        for use_pK in ([True, False] if kind == Kind.BASE else [True]):
            configs.append((kind, use_avg_metric, use_pK))

depth = 2
for n in [4, 6]:
    for kind, use_avg_metric, use_pK in configs:
        tag = f"n{n} {kind.name} avg={use_avg_metric} pK={use_pK}"
        Ws = [rng.standard_normal((n, n)) * math.sqrt(2.0 / n) for _ in range(depth)]

        # torch chain (factor=True drives mlp_kprop.factor_k3 internally)
        K_t = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3)
        # port chain
        K_n = {1: NH.HTensor(np.zeros(n), r=0), 2: NH.HTensor(np.eye(n), r=0)}

        for l in range(depth):
            metric_t = 2.0 * torch.ones(n) if use_avg_metric else None
            metric_n = 2.0 * np.ones(n) if use_avg_metric else None

            WK_t = linear_kprop(K_t, tt(Ws[l]), k_max=3, set_metric=metric_t)
            K_t = nonlin_kprop(
                WK_t,
                nonlin_wick_coef=t_relu_wick,
                k_max=3,
                kind=kind,
                use_pK=use_pK,
                factor=True,
            )

            WK_n = np_linear(K_n, Ws[l], set_metric=metric_n)
            K_n = NF.factored_nonlin_kprop_k3(
                K_in=WK_n,
                nonlin_wick_coef=n_relu_wick,
                augment=(kind == Kind.AUGMENT),
                base=(kind == Kind.BASE),
                use_pK=use_pK,
            )

            assert set(K_t.keys()) == set(K_n.keys()), (
                f"{tag} layer {l}: keys {set(K_t.keys())} vs {set(K_n.keys())}"
            )
            for d in sorted(K_t.keys()):
                chk(sec, f"{tag} L{l} K[{d}]", K_t[d].to_tensor(), K_n[d].to_tensor())
            # d=3 factored repeated-cache slices too
            if 3 in K_t:
                for part in [(3,), (2, 1)]:
                    chk(sec, f"{tag} L{l} K[3].dslice{part}", K_t[3].get_dslice(part), K_n[3].get_dslice(part))

        # Section 3: final-layer mean (state-threading check)
        chk("3-chain-mean", f"{tag} final mean", K_t[1].core, K_n[1].core)

# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
print("Per-section max abs errors:")
for s in sorted(SECTION_ERR):
    print(f"  {s}: {SECTION_ERR[s]:.3e}")
if FAILED:
    print("FAILED CHECKS:")
    for f in FAILED:
        print(f"  {f}")
print(f"MAXERR {MAXERR:.3e}")
print("PASS" if MAXERR < 1e-10 and not FAILED else "FAIL")
