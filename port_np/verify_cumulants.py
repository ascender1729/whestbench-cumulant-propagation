"""Verify port_np.cumulants_np (numpy) against mlp_kprop.cumulants (torch).

Run from repo root: uv run python port_np/verify_cumulants.py

Covers the ported API surface:
  part_sum (incl. d=None dict path), M_to_K, K_to_M,
  DS_part_sum (custom coef), DS_K_to_M, DS_M_to_K, DS_pK_to_M,
  _DS_pK_to_K_old, DS_pK_to_K (strict=True and strict=False with missing
  slices), _pK_to_K_coef parity.

Towers are built from the SAME numpy arrays on both sides:
  torch side via mlp_kprop.diagslice.DSTower, numpy side via
  port_np.diagslice_np.DSTower (from_tower on symmetric tensors, plus
  downward-closed from_slices subsets for realistic sparse structures).
Sampling utilities (stream_tensor/finish/moment_gen_slice/DS_moment_gen/
DS_cumulant_gen/DS_moment/DS_cumulant) were skipped in the port and are not
verified; torch DS_cumulant is still used here to produce one set of
realistic cumulant/power-cumulant inputs (adapted from tests/test_cumulants.py
test_pK_to_K).
ASCII only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from itertools import product

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

import mlp_kprop.cumulants as TC  # torch reference
import mlp_kprop.diagslice as TD
import port_np.cumulants_np as NC  # numpy port
import port_np.diagslice_np as ND
from mlp_kprop.partitions import int_partitions, vector_partitions
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
    err = float(np.abs(t - a).max())
    SEC[section] = max(SEC.get(section, 0.0), err)
    MAXERR = max(MAXERR, err)


def cmp_ds(section, tds, nds):
    assert isinstance(nds, ND.DSTensor), f"{section}: not a port DSTensor"
    assert set(tds.slices.keys()) == set(nds.slices.keys()), (
        f"{section}: slice keys {set(tds.slices.keys())} vs {set(nds.slices.keys())}"
    )
    assert tds.n == nds.n and tds.d == nds.d, f"{section}: n/d mismatch"
    for part in tds.slices:
        rec(section, tds.slices[part], nds.slices[part])


def cmp_tower(section, tt, nt):
    assert set(tt.keys()) == set(nt.keys()), (
        f"{section}: degree keys {set(tt.keys())} vs {set(nt.keys())}"
    )
    for deg in tt.keys():
        cmp_ds(f"{section}", tt[deg], nt[deg])


def sym(rng, n, d):
    return np.ascontiguousarray(np_symmetrize(rng.standard_normal((n,) * d)))


def make_towers(rng, n, d_max):
    """Full symmetric Tower (dict) + paired torch/np DSTowers from SAME arrays."""
    arrs = {d: sym(rng, n, d) for d in range(1, d_max + 1)}
    t_tower = {d: torch.tensor(arrs[d]) for d in arrs}
    t_ds = TD.DSTower.from_tower(t_tower)
    n_ds = ND.DSTower.from_tower({d: arrs[d].copy() for d in arrs})
    return arrs, t_tower, t_ds, n_ds


def filter_towers(t_ds, n_ds, keep):
    """Downward-closed sparse DSTowers from slice subsets of paired full towers."""
    t_slices, n_slices = {}, {}
    for d in t_ds.keys():
        for part in t_ds[d].slices:
            if keep(part):
                t_slices[part] = t_ds[d].slices[part].clone()
                n_slices[part] = n_ds[d].slices[part].copy()
    t_f = TD.DSTower.from_slices(t_slices, autozero=True)
    n_f = ND.DSTower.from_slices(n_slices, autozero=True)
    assert t_f.is_downward_closed() and n_f.is_downward_closed()
    return t_f, n_f


rng = np.random.default_rng(20260612)

# coefficient functions defined once, used verbatim on both sides
import math  # noqa: E402

custom_set_coef = lambda part: 1.0 / (1.0 + len(part)) + sum(len(b) for b in part)
custom_vec_coef = lambda vpart: 0.5 * len(vpart) + math.prod(sum(v) for v in vpart) * 0.25

for n in [4, 6]:
    for d_max in [2, 3, 4]:
        arrs, t_tower, t_ds, n_ds = make_towers(rng, n, d_max)
        n_tower = {d: arrs[d] for d in arrs}

        # ----------------- Tower-level part_sum / M_to_K / K_to_M -----------
        for d in range(1, d_max + 1):
            rec("part_sum_custom", TC.part_sum(t_tower, custom_set_coef, d),
                NC.part_sum(n_tower, custom_set_coef, d))
            rec("M_to_K", TC.M_to_K(t_tower, d), NC.M_to_K(n_tower, d))
            rec("K_to_M", TC.K_to_M(t_tower, d), NC.K_to_M(n_tower, d))
        # d=None dict path
        t_all = TC.M_to_K(t_tower)
        n_all = NC.M_to_K(n_tower)
        assert set(t_all.keys()) == set(n_all.keys())
        for d in t_all:
            rec("M_to_K_dictpath", t_all[d], n_all[d])

        # ----------------- DS-level conversions on full towers --------------
        cmp_tower("DS_part_sum_custom", TC.DS_part_sum(t_ds, custom_vec_coef),
                  NC.DS_part_sum(n_ds, custom_vec_coef))
        cmp_tower("DS_K_to_M", TC.DS_K_to_M(t_ds), NC.DS_K_to_M(n_ds))
        cmp_tower("DS_M_to_K", TC.DS_M_to_K(t_ds), NC.DS_M_to_K(n_ds))
        cmp_tower("DS_pK_to_M", TC.DS_pK_to_M(t_ds), NC.DS_pK_to_M(n_ds))
        cmp_tower("DS_pK_to_K", TC.DS_pK_to_K(t_ds), NC.DS_pK_to_K(n_ds))
        cmp_tower("_DS_pK_to_K_old", TC._DS_pK_to_K_old(t_ds), NC._DS_pK_to_K_old(n_ds))

        # ----------------- sparse downward-closed slice structures ----------
        for name, keep in [("len2", lambda p: len(p) <= 2), ("max2", lambda p: max(p) <= 2)]:
            t_f, n_f = filter_towers(t_ds, n_ds, keep)
            cmp_tower(f"DS_K_to_M_{name}", TC.DS_K_to_M(t_f), NC.DS_K_to_M(n_f))
            cmp_tower(f"DS_M_to_K_{name}", TC.DS_M_to_K(t_f), NC.DS_M_to_K(n_f))
            cmp_tower(f"DS_pK_to_M_{name}", TC.DS_pK_to_M(t_f), NC.DS_pK_to_M(n_f))
            cmp_tower(f"DS_pK_to_K_{name}", TC.DS_pK_to_K(t_f), NC.DS_pK_to_K(n_f))
            cmp_tower(f"_DS_pK_to_K_old_{name}", TC._DS_pK_to_K_old(t_f),
                      NC._DS_pK_to_K_old(n_f))

        # ----------------- strict=False with missing slices -----------------
        if d_max >= 3:
            # Drop the (1,1) slice from degree 2: not downward closed anymore;
            # blocks needing (1,1) hit the missing-slice 0.0 path on both sides.
            t_slices, n_slices = {}, {}
            for d in t_ds.keys():
                for part in t_ds[d].slices:
                    if part == (1, 1):
                        continue
                    t_slices[part] = t_ds[d].slices[part].clone()
                    n_slices[part] = n_ds[d].slices[part].copy()
            t_m = TD.DSTower.from_slices(t_slices, autozero=True)
            n_m = ND.DSTower.from_slices(n_slices, autozero=True)
            assert not t_m.is_downward_closed() and not n_m.is_downward_closed()
            cmp_tower("DS_pK_to_K_nonstrict", TC.DS_pK_to_K(t_m, strict=False),
                      NC.DS_pK_to_K(n_m, strict=False))

# ---------------------------------------------------------------------------
# _pK_to_K_coef parity (pure combinatorics)
# ---------------------------------------------------------------------------
for vec in [(1, 1), (2,), (1, 1, 1), (2, 1), (2, 2), (3, 1), (1, 1, 1, 1), (2, 1, 1)]:
    for vp in vector_partitions(vec):
        tc = TC._pK_to_K_coef(vp)
        nc = NC._pK_to_K_coef(vp)
        assert tc == nc, f"_pK_to_K_coef mismatch on {vp}: {tc} vs {nc}"
        rec("_pK_to_K_coef", np.array(float(tc)), np.array(float(nc)))

# ---------------------------------------------------------------------------
# Realistic case adapted from tests/test_cumulants.py::test_pK_to_K
# (torch DS_cumulant used only to BUILD inputs; both ports run on same arrays)
# ---------------------------------------------------------------------------
n, N, d_max = 4, 400, 4
X_np = rng.standard_normal((N, n)) @ rng.standard_normal((n, n)) + rng.standard_normal((1, n)) * 5
X = torch.tensor(X_np)
K_ref = TC.DS_cumulant(X, d_max=d_max).to_tower()
X_pows = torch.concat([X**k for k in range(1, d_max + 1)], dim=1)
K_pows = TC.DS_cumulant(X_pows, d_max=d_max).to_tower()
pK = {d: torch.zeros_like(K_ref[d]) for d in range(1, d_max + 1)}
for d in range(1, d_max + 1):
    for idxs in product(range(n), repeat=d):
        counts = defaultdict(int)
        for i in idxs:
            counts[i] += 1
        pK[d][idxs] = K_pows[len(set(idxs))][tuple(i + (counts[i] - 1) * n for i in counts)]
pK_arrs = {d: pK[d].numpy().copy() for d in pK}
t_dspK = TD.DSTower({d: TD.DSTensor.from_tensor(torch.tensor(pK_arrs[d])) for d in pK_arrs})
n_dspK = ND.DSTower({d: ND.DSTensor.from_tensor(pK_arrs[d].copy()) for d in pK_arrs})
t_K = TC.DS_pK_to_K(t_dspK)
n_K = NC.DS_pK_to_K(n_dspK)
cmp_tower("DS_pK_to_K_realistic", t_K, n_K)
cmp_tower("_DS_pK_to_K_old_realistic", TC._DS_pK_to_K_old(t_dspK), NC._DS_pK_to_K_old(n_dspK))
# Semantic sanity (vs true cumulants, loose tolerance; parity is the real test)
for d in range(1, d_max + 1):
    sem = float((t_K[d].to_tensor() - K_ref[d]).abs().max())
    assert sem < 1e-6, f"semantic check failed at d={d}: {sem}"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
for k in sorted(SEC):
    print(f"{k:28s} {SEC[k]:.3e}")
print(f"MAXERR {MAXERR:.6e}")
print("PASS" if MAXERR < 1e-10 else "FAIL")
