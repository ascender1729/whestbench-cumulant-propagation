"""Self-check: kprop k=3 matches Monte-Carlo on a random He MLP, on plain numpy.
No torch, no flopscope. Run: python demo.py"""
import numpy as np
from port_np.kprop_np import kprop_layer_means, Kind


def he_mlp(width, depth, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, np.sqrt(2.0 / width), (width, width)) for _ in range(depth)]


def mc_means(Ws, n_samples=1_000_000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, Ws[0].shape[0]))
    out = []
    for W in Ws:
        x = np.maximum(x @ W, 0.0)
        out.append(x.mean(0))
    return out


def demo():
    Ws = he_mlp(width=128, depth=4)
    k3 = kprop_layer_means(Ws, k_max=3, kind=Kind.SIMPLE, factor=True)
    k2 = kprop_layer_means(Ws, k_max=2, kind=Kind.SIMPLE, factor=True)
    mc = mc_means(Ws)
    err3 = float(np.max(np.abs(np.asarray(k3[-1]) - mc[-1])))
    err2 = float(np.max(np.abs(np.asarray(k2[-1]) - mc[-1])))
    print(f"final-layer max|kprop - MC|:  k3={err3:.2e}  k2={err2:.2e}")
    assert err3 < 2e-2, f"k3 too far from MC: {err3}"   # MC noise + finite-width at w=128
    assert err3 <= err2 + 1e-9, "k3 should be at least as good as k2"
    print("OK")


if __name__ == "__main__":
    demo()
