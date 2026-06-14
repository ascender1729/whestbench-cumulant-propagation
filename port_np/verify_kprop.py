"""Verify port_np.kprop_np (driver port) against torch mlp_kprop.kprop_harmonic.

Run from repo root: uv run python port_np/verify_kprop.py

Phase A (n=6, L=3, threshold 1e-10):
  torch MLP(6,6,6, num_layers=4, relu, he) -> 3 ReLU activations act0..act2.
  Reference: mlp_kprop(..., k_max=3, output_all=True, output_d_max=1).
  Port: kprop_layer_means(Ws_whest, k_max=3, metric_by_layer=mlp.init_scale[:3])
  Configs: (SIMPLE, factor=True), (AUGMENT, factor=True), (SIMPLE, factor=False),
           (AUGMENT, factor=False).

Phase B (n=256, L=8, threshold 1e-8):
  torch MLP num_layers=9, k_max=3 SIMPLE factor=True only.
  Prints port wall-time for the full 8-layer pass.

ASCII only.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
np.seterr(all="ignore")

import mlp_kprop.kprop_harmonic as TKH
from mlp_kprop.mlp import MLP

from port_np.kprop_np import Kind, kprop_layer_means

MAXERR = 0.0
FAILED = []


def rec(name, err, thresh):
    global MAXERR
    err = float(err)
    MAXERR = max(MAXERR, err)
    status = "ok" if err <= thresh else "FAIL"
    if err > thresh:
        FAILED.append((name, err))
    print(f"  {name:60s} err={err:.3e} [{status}]")


def torch_ref_means(mlp, n, L, kind_t, factor):
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    K_by_layer = TKH.mlp_kprop(
        mlp,
        K_in,
        k_max=3,
        output_all=True,
        kind=kind_t,
        factor=factor,
        output_d_max=1,
        up_to_layer=f"act{L - 1}",
    )
    means = []
    for i in range(L):
        ht = K_by_layer[f"act{i}"][1]
        assert ht.r == 0
        means.append(ht.core.detach().numpy().copy())
    return means


def run_config(mlp, n, L, kind_name, factor, thresh):
    kind_t = getattr(TKH.Kind, kind_name)
    kind_n = getattr(Kind, kind_name)
    ref = torch_ref_means(mlp, n, L, kind_t, factor)

    Ws = [mlp.Ws[i].weight.detach().numpy().T.copy() for i in range(L)]  # whestbench (in,out)
    t0 = time.time()
    port = kprop_layer_means(
        Ws,
        k_max=3,
        kind=kind_n,
        factor=factor,
        metric_by_layer=list(mlp.init_scale[:L]),
    )
    dt = time.time() - t0

    assert len(port) == L
    cfg_err = 0.0
    for i in range(L):
        err = float(np.abs(ref[i] - port[i]).max())
        cfg_err = max(cfg_err, err)
        rec(f"n={n} L={L} {kind_name} factor={factor} act{i} mean", err, thresh)
    return dt, cfg_err


def main():
    # ---------------- Phase A: small exact ----------------
    print("Phase A: n=6, L=3, threshold 1e-10")
    torch.manual_seed(20260612)
    mlp_a = MLP(input_dim=6, hidden_dim=6, output_dim=6, num_layers=4,
                nonlin="relu", init_kind="he")
    assert not mlp_a.has_bias(), "Phase A MLP unexpectedly has biases."
    phase_a_err = 0.0
    for kind_name, factor in [
        ("SIMPLE", True),
        ("AUGMENT", True),
        ("SIMPLE", False),
        ("AUGMENT", False),
    ]:
        _, cfg_err = run_config(mlp_a, n=6, L=3, kind_name=kind_name,
                                factor=factor, thresh=1e-10)
        phase_a_err = max(phase_a_err, cfg_err)
    print(f"Phase A max err: {phase_a_err:.3e}")

    # ---------------- Phase B: full size ----------------
    print("Phase B: n=256, L=8, k_max=3 SIMPLE factor=True, threshold 1e-8")
    torch.manual_seed(20260613)
    mlp_b = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=9,
                nonlin="relu", init_kind="he")
    assert not mlp_b.has_bias(), "Phase B MLP unexpectedly has biases."
    dt, phase_b_err = run_config(mlp_b, n=256, L=8, kind_name="SIMPLE",
                                 factor=True, thresh=1e-8)
    print(f"Phase B port wall-time (kprop_layer_means, n=256 L=8): {dt:.2f}s")
    print(f"Phase A max err: {phase_a_err:.3e}")
    print(f"Phase B max err: {phase_b_err:.3e}")

    print(f"MAXERR {MAXERR:.6e}")
    if FAILED:
        print(f"FAIL ({len(FAILED)} comparisons over threshold)")
        for name, err in FAILED:
            print(f"  FAILED: {name} err={err:.3e}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
