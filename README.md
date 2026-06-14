# White-Box MLP Estimation: Cumulant Propagation (NumPy/flopscope port)

An independent, dependency-light **NumPy / `flopscope.numpy` reimplementation**
of the Alignment Research Center's *cumulant propagation* (`kprop`) method for
analytically estimating the expected post-ReLU activations of a wide random MLP
without sampling. Built for the
[ARC White-Box Estimation Challenge 2026](https://www.alignment.org/blog/announcing-the-arc-white-box-estimation-challenge/)
(WhestBench, AIcrowd).

## What this is

Given the weights of a deep ReLU MLP (width 256, depth 8, He init), predict the
per-neuron expected activation `E[ReLU(z_l)]` under standard-normal inputs. ARC's
method propagates **cumulants** (up to order `k`) layer by layer using a
Wick/Hermite expansion over factored symmetric tensors. The reference
implementation is in PyTorch; this is a clean port that runs on plain NumPy and
on the challenge's analytical FLOP-counting backend `flopscope.numpy`.

## Results

- **Bit-exact to the PyTorch reference**: max abs difference `<= 1e-12` across all
  ported modules (see `port_np/verify_*.py`).
- **k_max=3 (SIMPLE, factored)**: final-layer MSE ~`1.2e-6` at ~`1.7e10`
  analytical FLOPs, ~17x lower error than the covariance (k=2) baseline.
- Runs under the grader's constrained backend (no raw `numpy`, FLOP budget,
  symmetry-tracked tensors). See `RESULTS.md` and the report for the negative
  results (energy-compression and skew shortcuts that do **not** work) and the
  large-N / QFT reasons depth-8 resists the cheap shortcuts.

## Layout

```
port_np/            the port: 10 modules, bit-exact vs torch
  kprop_np.py         entry point: kprop_layer_means(Ws, k_max, kind, factor)
  wick_np.py          ReLU Wick/Hermite coefficients
  harmonic_np.py      factored symmetric tensors (harmonic decomposition)
  diagslice_np.py     diagonal-slice tensor algebra
  factor_k3_np.py     CP-factored 3rd-cumulant propagation
  _gen_single.py      bundles the package into a single-file estimator.py
  verify_*.py         module-by-module correctness checks against the torch ref
demo.py             numpy-only self-check: kprop k3 vs Monte-Carlo
estimator.py        the self-contained challenge submission (generated)
paper/              technical report (LaTeX + PDF) and figures
RESULTS.md          method notes and measured results
```

## Quick check

Runs the k=3 estimator against Monte-Carlo on a random He MLP (plain NumPy, no
PyTorch, no flopscope):

```bash
pip install -r requirements.txt
python demo.py
# final-layer max|kprop - MC|:  k3=6.06e-03  k2=2.35e-02
# OK
```

## Build the submission

```bash
cd port_np && python _gen_single.py   # writes ../estimator.py
```

`estimator.py` (checked in) is the generated single-file submission: it embeds
the `port_np` package and runs on the challenge's `flopscope.numpy` backend with
raw `numpy` blocked. The `verify_*.py` scripts cross-check each module against
ARC's PyTorch reference and therefore require that reference package installed.

## Attribution

The cumulant-propagation **method** is ARC's:
[*Estimating the Expected Output of Wide Random MLPs More Efficiently than
Sampling*](https://arxiv.org/abs/2605.05179) and
[alignment-research-center/mlp_cumulant_propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation).
This repository is an **independent port** (the algorithm in NumPy + a writeup),
not affiliated with ARC. Licensed MIT to match the reference.

## Author

Pavan Kumar Dubasi, [github.com/ascender1729](https://github.com/ascender1729)
