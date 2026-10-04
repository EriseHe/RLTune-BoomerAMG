# Anchored scheduled relaxation: mathematical and implementation review

The supplied derivation is correct under its stated SPD, full-row
ℓ₁-Jacobi and fixed exact two-grid assumptions. **Use (2.6, 1) as the
prescribed fine-grid baseline.** The result supports the coefficient choice;
it does not prove a globally optimal full multilevel solver or minimum
wall-clock completion time. The attached review's suggestion to retain
(2.5, 1) is superseded by the user's explicit Run 04 request.

## Checked derivation

For an SPD matrix, set \(D_{ii}=\sum_j|a_{ij}|\). Pairing symmetric
off-diagonal entries and applying \(2|x_ix_j|\le x_i^2+x_j^2\) gives
\(0\prec A\preceq D\). Thus
\(T=A^{1/2}D^{-1}A^{1/2}\) has spectrum in \((0,1]\), and one weighted
sweep in energy coordinates is \(S(w)=I-wT\).

The two-step polynomial is
\[
p_a(t)=(1-t)(1-at),\qquad
\eta(a)=\max_{0\le t\le1}t(1-t)^2(1-at)^2.
\]
The objective follows the weighted two-level smoothing bound in
[Lottes, equation (4)](https://arxiv.org/pdf/2202.08830v3).
The anchored restriction and calculation here are a specialization, not a
claim that the source prescribes this pair for BoomerAMG. Full-row ℓ₁
normalization is also explicitly used in
[D'Ambra et al., equation (7)](https://arxiv.org/html/2407.09848v3).

Writing \(f_a(t)=\sqrt t(1-t)(1-at)\),
\[
f'_a(t)=\frac{1-3(a+1)t+5at^2}{2\sqrt t}.
\]
At \(a_*=(3+\sqrt5)/2\), the stationary points
\(t_+=1-2/\sqrt5\) and \(t_-=(1+1/\sqrt5)/2\) have opposite equal
peaks, with squared magnitude \((10-2\sqrt5)/125\). Decreasing \(a\)
increases the positive value at the same \(t_+\); increasing \(a\)
increases the negative magnitude at the same \(t_-\). This proves the
unique global minimizer, rather than merely identifying a stationary value.

| High coefficient | Exact-extrema evaluation of η |
|---|---:|
| 2.50 | 0.0457988174103 |
| **2.60** | **0.0444568224570** |
| (3+√5)/2 ≈ 2.61803398875 | 0.0442229123600 |
| 2.65 | 0.0465429476041 |

All 41 points in {1, 1.05, …, 3} were evaluated at their analytic extrema:
2.60 is the minimizer. Nearest rounding alone would not prove this.
The weights' sum and product determine the polynomial; their difference
does not constitute an effective constant weight. The high step may amplify
some modes individually; the pair is the object of the smoothing analysis.

For an exact fixed coarse projection \(Q=Q^\top=Q^2\), the note's identity
also holds:
\[
\rho(E(1)E(a))=\|Qp_a(T)Q\|_2^2
\le \|T^{-1/2}Q\|_2^2\eta(a),\qquad E(w)=S(w)QS(w).
\]
Cyclic permutation of square factors preserves the characteristic polynomial,
even when a factor is singular. Compression to range(Q) is symmetric; its
squared norm gives the spectral radius. This does not assert equality with
the norm of the original, potentially nonnormal, two-cycle product.
The same radius for the reversed phase does not imply identical intermediate
residuals or stopping times. The full proof is in the accompanying TeX note.

## Match to the actual code

| Assumption or action | Production evidence | Conclusion |
|---|---|---|
| SPD fine operator | `amg_runtime.c` maps stencil 0 to positive k,c,a0 diffusion and zero a1,a2,a3 advection; all 600 jobs checked | Positive-coefficient Dirichlet 7-point diffusion is SPD |
| ℓ₁-Jacobi | `configure_smoother_profile` selects 18/18/9 | Down/up Jacobi; Gaussian elimination on the coarsest level |
| Full-row diagonal | `par_amg.c` defaults to relax_order=0; no job or step overrides it; `par_amg_setup.c` uses norm option 1 with NULL marker; `ams.c` sums absolute entries | Exactly the diagonal in the proof, not a C/F-restricted variant |
| Simultaneous sweep | `hypre_BoomerAMGRelax18WeightedL1Jacobi` routes natural-order sweeps through Relax7Jacobi, which forms the old-iterate residual before diagonal scaling | x ← x + w D⁻¹(b−Ax) |
| Action granularity | `solve_schedule_case` calls `step_rl` once per cycle; `amg_rl_step_solver` applies the weight then one complete solve iteration | One full V-cycle per action, with one pre- and one post-sweep |
| Weight scope | `HYPRE_BoomerAMGSetRelaxWt` updates the level weights | Same weight on all smoothing levels; direct coarse solve unaffected |
| Fixed exact Q | Production uses recursive multilevel coarse solves | Exact only in the ideal two-grid specialization |
| High-first phase | Schedule starts with 2.6 and resets on every primary solve | Explicit prescription; not proved preferable to reverse phase |
| Cost and stopping | Existing native timers, relative tolerance 10⁻⁶, 50-cycle native status rule, reference fallback | Unchanged in Run 04; all failed/recovery work charged |

In original coordinates the actual fine-level cycle has the form
\[
E_0(w)=S_0(w)[I-P_0B_1(w)P_0^\top A_0]S_0(w),
\quad S_0(w)=I-wD_0^{-1}A_0.
\]
The recursive coarse solver \(B_1(w)\) depends on the weight. It is generally
not \(A_1^{-1}\). A direct solve at the last level does not make every
intermediate coarse correction exact. Standard Galerkin SPD preservation
requires full-column-rank interpolation in exact arithmetic; the audit does
not certify all floating-point coarse matrices.

This is suitable as a supporting proposition for the paper's prescribed
baseline, alongside its existing nonstationary-operator argument. It is not
a replacement proof of RL optimality, feedback necessity, or finite-time
speedup. Nonsymmetric advection is outside the SPD theorem. Lottes's sharper
multilevel objective differs from η and has different optimum coefficients;
it must not be conflated with this constrained objective.

## Reproducible checks and Run 04 scope

`experiments/paper_final/verify_anchored_schedule.py` verifies the symbolic
peaks, all grid points, 500 random small SPD/projection identities and bounds,
row-ℓ₁ domination examples with mixed off-diagonal signs, all 600 frozen
input/setup jobs, and native 18/18/9 V-cycle behavior for each checkpoint.
Results are saved in `anchored_schedule_verification_20260927.json`.
The largest numerical identity error was about 1.2×10⁻¹⁵.

Run 04 means **Module 05**, following the linked task's Run 03: six unchanged
accepted checkpoints (including refreshed seed 4), the same 100 cases,
three repetitions, and all previously selected fixed weights. Only the
(2.5,1) comparator becomes (2.6,1); the (1,3) comparator remains. Each
method receives fresh timings in the original execution order. The primary
heatmaps now include the anchored comparator. No additional online branch,
retraining, new PDE family, reversed-phase tuning or unanchored pair was
requested or introduced from the attached review.

## Suggested paper text

> We include a prescribed period-two relaxation baseline with weights
> (2.6,1), motivated by normalized SPD ℓ₁-Jacobi polynomial smoothing.
> Anchoring one coefficient at unity and minimizing
> max_{t∈[0,1]} t(1−t)²(1−at)² gives a*=(3+√5)/2; analytic evaluation on
> our 0.05 grid selects a=2.6. An exact two-grid model links this surrogate
> to an upper bound on the two-cycle spectral radius. We transfer the
> coefficients to complete multilevel cycles and assess their finite-tolerance
> cost empirically.
