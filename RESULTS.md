# ARC White-Box Estimation Challenge (WhestBench 2026) - working notes

Date: 2026-06-11
Challenge: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026
Starter kit: whest-starterkit/ (cloned, uv-synced, Python 3.10, whestbench 0.10.0, flopscope 0.5.0)

## Task
Given weights of a random ReLU MLP (width 256, depth 8, He init), predict the
expected post-ReLU activation of every neuron under X ~ N(0, I). Leaderboard
score = final-layer MSE x max(0.1, effective_compute / 6.8e10), lower better.
Effective compute = analytical FLOPs + 1e11 * residual_wall_time_seconds.

Timeline: warm-up open now; Phase 1 Jun 18 - Jul 31; Phase 2 ends Sep 19;
prizes $50k/$20k/$10k + $20k algorithmic. 50 submissions/team/day.

## Results (mini split of arc-whestbench-public-2026@v1-warmup, first 20 MLPs, seed 42)

| Estimator | final_layer_mse | adjusted score | util | notes |
|---|---|---|---|---|
| zeros (template) | ~8.3e-01 | ~8.3e-02 | ~0 | shipped |
| 02 mean propagation | ~7.5e-04 | ~7.5e-05 | <1% | shipped |
| 03 covariance propagation | 3.43e-05 | 3.43e-06 | 1.5% | shipped baseline |
| 10 Hermite-8 covariance | 3.30e-05 | 3.30e-06 | 2.9% | ours; exact bivariate ReLU cov via Hermite series |
| 11 global-surrogate CV | 1.99e-05 | 5.58e-06 | 28% | ours; killed by residual wall-time charge |
| 12 layerwise telescoping CV | 1.25e-05 | **2.49e-06** | 19.9% | ours; BEST adjusted; packaged |
| 13 full-budget telescoping CV | **3.49e-06** | 2.63e-06 | 76% | ours; best raw accuracy (~10x baseline) |
| 14 tuned (16k chunks) | 3.49e-06 | 2.70e-06 | 78% | bigger chunks raised GC residual; worse |

## Key findings
1. The shipped covariance baseline's "gain" off-diagonal update is the k=1
   term of the exact Hermite expansion; extending to k=8 (exact bivariate
   Gaussian covariance) improves MSE only ~4% -> the dominant error is the
   joint-Gaussianity assumption itself, not the covariance update.
2. Layerwise telescoping control variate: E_hat_l = A_l W^T E_hat_{l-1} + b_l
   + mean_k[ReLU(z_l) - (A_l z_l + b_l)] with z_l the true pre-activations and
   (A_l, b_l) the mech linearization. Provably unbiased for ANY fixed (A_l,b_l);
   reuses the true forward pass (no surrogate matmuls). Var per layer ~4x lower
   than raw activation sampling, residuals weakly correlated across layers.
3. Scoring economics: for a biased estimator the optimum sits at the 10%
   multiplier floor; for an unbiased one score ~ (a/k)(C/B) which DECREASES in
   C until residual-time overhead flattens it. With residual wall time charged
   at 1e11 FLOP/s, Python/GC overhead is the binding constraint on this laptop
   (60-270 ms/MLP = 0.6-2.7e10 FLOP-equiv). On the grader's 16-vCPU box the
   residual charge will shrink and both 12 and 13 should improve.
4. Empirical-Bayes shrinkage (per-layer scalar) between the unbiased CV
   estimate and the biased mech estimate is cheap and adaptive: weight from
   two-replicate variance estimate.
5. Variance constant measured: Var(f - CV) ~ 0.094 avg on final layer ->
   MSE ~ 0.114/k. To go materially below ~2e-6 adjusted needs a structurally
   better method (higher-cumulant mechanistic propagation per ARC's
   arXiv:2605.05179), not more samples. Quadratic surrogate gains only ~4%
   (Hermite spectrum of ReLU is kink-dominated). Antithetic useless (residual
   is even). Empirical slope refit gains ~3%.

## Artifacts
- whest-starterkit/examples/10_hermite_covariance.py
- whest-starterkit/examples/11_hermite_cv.py
- whest-starterkit/examples/12_layerwise_cv.py  <- designated candidate
- whest-starterkit/examples/13_layerwise_cv_full.py
- whest-starterkit/estimator.py (= copy of 12)
- whest-starterkit/submission_layerwise_cv.tar.gz  <- ready to upload
- report_11/12/13/14.json (local score reports)

## Next steps
- Operator: create AIcrowd account / team, upload tarball to warm-up round.
- Re-tune _RESERVE_FRACTION on grader feedback (residual charge differs).
- Phase 1 (real leaderboard) starts Jun 18.
- Research direction for the $20k algorithmic prize: implement 4th-cumulant /
  Edgeworth-corrected propagation (ARC paper) with structured (low-rank +
  diagonal) cumulant tensors to fix the depth-driven Gaussianity breakdown;
  keep the telescoping CV as an unbiased wrapper around ANY mech core.
- Consider Lambda/AWS for wide hyperparameter sweeps over the full 100-MLP
  mini split (local 20-MLP runs take ~1-3 min each).
