# Period-two relaxation: accepted mathematical basis for the paper

Recorded on 2026-09-28 from the user's review and the independently checked
scalar calculations and paired experiment in this chat. This note preserves
the mathematical basis for later manuscript work.

The prescribed **unanchored** baseline is **(2.85, 1.10)**, applied in
high–low order and reset at each primary solve. Under the September 28 module
renumbering, it is the prescribed periodic continuation for Module 05
checkpoint training and the planned Module 06 frozen-policy follow-up (formerly
discussed as Module 05 Run 05).
Its selection rule is the discrete weighted minimax criterion below. The
timing comparison provides supplementary implementation evidence. The
current (2.6, 1) material describes the earlier **anchored Run 04** problem;
its theorem remains correct for that restricted class. Historical run
sources and notes are retained unchanged for reproducibility.

Suggested method name: **Grid-constrained weighted-minimax period-two
relaxation.** Module 05 now prepares checkpoints using a shared 1000-problem
W1 prefix and four method-specific 4000-problem continuations; the full
unanchored Module 06 follow-up has not been launched. The separate two-pair
timing comparison is complete and retains its historical `05_policy` path.

## Normalization and assumptions

For a symmetric positive definite matrix A, define the full absolute row-sum
diagonal D by D_ii = sum_j |a_ij|. The inequality
2|x_i x_j| <= x_i² + x_j² gives 0 < A <= D in the Loewner order. Therefore

\[
T=A^{1/2}D^{-1}A^{1/2},\qquad 0\prec T=T^\top\preceq I.
\]

In energy coordinates, a weighted full-row ℓ₁-Jacobi sweep is
S(w) = I - wT. For two weights define

\[
p_{a,b}(t)=(1-at)(1-bt),\qquad
\eta(a,b)=\max_{0\le t\le1}t\,p_{a,b}(t)^2.
\]

This is the square of the weighted polynomial quantity in the classical
two-level smoothing bound discussed by
[Lottes, *Optimal Polynomial Smoothers for Multigrid V-cycles*, equations
(4), (7)–(8)](https://arxiv.org/pdf/2202.08830v3).
The current PDE family is SPD diffusion. The result does not extend as
stated to nonsymmetric advection.

## Continuous optimum and a short proof

Among real polynomials of degree at most two with p(0) = 1, the unique
minimizer is the scaled fourth-kind Chebyshev polynomial

\[
p_*(t)=1-4t+\frac{16}{5}t^2,
\qquad \max_{0\le t\le1}t p_*(t)^2=\frac1{25}.
\]

Its factorization gives, up to permutation,

\[
(a_*,b_*)=
\left(2+\frac2{\sqrt5},\;2-\frac2{\sqrt5}\right)
\approx(2.894427190999916,\;1.105572809000084).
\]

To see the bound, write x = sqrt(t). Then
x p_*(x²) = (16x⁵ - 20x³ + 5x)/5 = T_5(x)/5, where T_5 is the
first-kind Chebyshev polynomial. Hence the absolute weighted value is at
most 1/5. It attains alternating values +1/5, -1/5, +1/5 at

\[
t_1=\frac{3-\sqrt5}{8},\qquad
t_2=\frac{3+\sqrt5}{8},\qquad t_3=1.
\]

For uniqueness, suppose q(0) = 1 and q has no larger objective. Since
q - p_* = t(c_1 + c_2 t), the linear function c_1 + c_2 t must be
nonpositive at t_1 and t_3 and nonnegative at t_2. Its interior value is a
strict convex combination of the endpoint values. These inequalities force
all three values to be zero, so c_1 = c_2 = 0 and q = p_*.

Both exact coefficients lie in [1,3], so this pair also solves the
continuous coefficient problem on [1,3]². The polynomial is unique; the
ordered coefficient pair is unique only up to reversal.

## Discrete choice and the distinction from rounding

Use the same admissible weight grid as the learned controller:

\[
\mathcal W=\{1+j/20:j=0,\ldots,40\},\qquad
(a_{\mathcal W},b_{\mathcal W})\in
\operatorname*{arg\,min}_{a,b\in\mathcal W}\eta(a,b).
\]

There are 41×42/2 = 861 unordered pairs, including repeated coefficients.
For f(t) = sqrt(t)[1-(a+b)t+ab t²], interior stationary points obey

\[
1-3(a+b)t+5ab t^2=0,
\qquad
t_\pm=\frac{3(a+b)\pm\sqrt{9(a+b)^2-20ab}}{10ab}.
\]

For each grid pair, evaluate t p(t)² at 0, 1, and these roots when they lie
strictly inside (0,1). Other stationary points of t p(t)² arise from zeros
of p and have value zero. This yields the maximum without sampling a
spectral mesh.

Evaluation at 80-digit precision gives

\[
\operatorname*{arg\,min}_{a,b\in\mathcal W}\eta(a,b)
=\{(2.85,1.10),(1.10,2.85)\}.
\]

| Pair | Definition | η |
|---|---|---:|
| (2+2/√5, 2-2/√5) | Continuous optimum | 0.0400000000000000 |
| (2.90, 1.10) | Componentwise-rounded continuous optimum | 0.0414047573559777 |
| **(2.85, 1.10)** | **Discrete weighted minimax choice** | **0.0405233195838067** |
| (2.60, 1.00) | Historical anchored grid optimum, b = 1 | 0.0444568224569786 |

The continuous pair has the smallest objective. The discrete choice improves
on its rounded approximation. The grid calculation uses no PDE test times
and is a finite optimization of the mathematical objective. The complete
861-pair ranking and extrema are reproduced by the script linked below.
This is high precision numerical verification of the finite minimum, rather
than a claim that the decimal pair follows from a continuous closed form.

For comparison, fixing b = 1 instead gives the exact anchored optimum
a = (3+√5)/2 and η = (10-2√5)/125. Its grid minimizer is (2.6,1).
That restriction defines a different feasible class.

## Why 2.85 improves on rounding to 2.90

Rounding both exact coefficients to their nearest grid values gives

\[
p_{2.90,1.10}(t)=1-4t+3.19t^2=p_*(t)-0.01t^2.
\]

The downward perturbation makes the middle negative peak more negative.
Holding the lower coefficient at 1.10 and decreasing the upper one gives

\[
p_{2.85,1.10}(t)-p_{2.90,1.10}(t)
=0.05t(1-1.10t).
\]

This correction is positive for 0 < t < 1/1.10. It raises the negative
middle part and the first positive peak. Their combined effect reduces the
largest absolute weighted value:

| Pair | First positive peak of sqrt(t)p(t) | Negative middle peak | Value at t = 1 | Maximum magnitude |
|---|---:|---:|---:|---:|
| Exact continuous | 0.200000000 | -0.200000000 | 0.200000000 | 0.200000000 |
| (2.90, 1.10) | 0.199971841 | -0.203481590 | 0.190000000 | 0.203481590 |
| (2.85, 1.10) | 0.201304048 | -0.196111135 | 0.185000000 | 0.201304048 |

Each interior peak is evaluated at that pair's own stationary point.
Distance between coefficient pairs is a different objective. The effect is
compensation for joint rounding error, with no floating-point anomaly.
The reduction in η is approximately 2.13%; this is not a prediction of a
2.13% reduction in solve time.

## Exact two-grid connection and production limits

For a fixed exact coarse-complement projection Q = Qᵀ = Q² in energy
coordinates, define E(w) = S(w)QS(w) and P = p_{a,b}(T). Then

\[
\rho(E(b)E(a))=\|QPQ\|_2^2
\le \|T^{-1/2}Q\|_2^2\,\eta(a,b).
\]

Cyclic permutation of square factors gives the spectrum of (PQ)². In a
basis adapted to range(Q), the nonzero eigenvalues of PQ are those of the
symmetric compression QPQ, proving the equality. Inserting T^{-1/2}T^{1/2}
and using ||T^{1/2}p(T)||² <= η gives the inequality. Cyclic spectral
invariance does not require invertible factors. The displayed spectral
radius is not asserted to equal the norm of the potentially nonnormal
original two-cycle product.

The scalar objective is symmetric in a and b. It chooses an unordered
pair. Equal two-grid spectral radii for the two orders do not imply equal
intermediate errors, stopping times or runtime. **High first is an explicit
implementation convention**, held fixed in the comparison.

The earlier implementation audit remains applicable: Run 04 uses natural,
all-point relaxation order 0, full-row ℓ₁-Jacobi type 18 for both sweeps, a
type-9 direct coarsest solve, and one pre- and one post-sweep per action.
Each action executes one complete V-cycle and applies its weight across
smoothing levels. Phase is reset on each primary solve. The forthcoming
pair changes the weights while retaining those semantics.

The production fine-level cycle has the form

\[
E_0(w)=S_0(w)[I-P_0B_1(w)P_0^\top A_0]S_0(w),
\qquad S_0(w)=I-wD_0^{-1}A_0.
\]

The recursive coarse solver B_1(w) depends on the weight and is generally
inexact. A direct solve at the last level does not make every intermediate
coarse correction exact. The fixed exact Q model is an idealization.
Hierarchy-specific compression, initial residuals, finite stopping,
transients, phase and recovery can all change practical performance.

Lottes's sharper multilevel-bound objective,
sup_t t p(t)² / [1-p(t)²], is another optimization problem. The current
coefficient claim concerns η. Do not call this an optimal BoomerAMG runtime
schedule, a proof of RL optimality, or a guarantee for every hierarchy.

## What the paired timing comparison supports

The completed separate comparison used 100 previously used diffusion 60³
inputs, six frozen Joint checkpoint-selected hierarchy sets, four
repetitions per schedule, and 4,800 timed solves. It used one serial worker,
one MPI rank, one numerical library thread, adjacent matched pairs, and
balanced execution order reversed between rounds. All 24 warm-up solves
were excluded. Every timed solve converged without recovery.

| Schedule | Mean native continuation time (ms) | Including schedule dispatch (ms) | Mean primary cycles |
|---|---:|---:|---:|
| (2.85, 1.10) | 65.761815 | 65.771474 | 11.171666667 |
| (2.90, 1.10) | 66.168117917 | 66.177914395 | 11.226666667 |

Native continuation excludes common initial setup and includes any
policy-specific recovery cost. Using unrounded data, the reduction for
(2.85,1.10), relative to (2.90,1.10), is **0.614046%**. The approximately
0.6136% in the user's review comes from the displayed rounded means; both
support the same 0.61% summary. The cycle-count reduction is approximately
0.49%. Of the 600 distinct checkpoint/input pairs, 2.85 used fewer cycles
in 17, 2.90 used fewer in 20, and 563 tied; the sizes of those differences
give the reported mean reduction.

The observed runtime advantage held in all four rounds, both execution-order
groups and four of six checkpoint means. Its paired case-bootstrap 95%
interval is **[-0.209752%, 1.695355%]**, so it does not establish a decisive
expected-runtime separation. The 10,000 bootstrap resamples treat each of
the 100 shared PDE input IDs as a cluster across all six checkpoints.
Checkpoints and the four timing rounds are held fixed; the interval is not
a general hardware-noise or new-training-seed interval. The raw timings,
costs, coverage and confidence calculation were independently checked in
this repository; the external review accepted the supplied interval rather
than reconstructing it.

Accepted interpretation: **the discrete smoothing criterion selects
(2.85,1.10); the paired experiment is compatible with a small practical
advantage over the rounded pair but does not establish a decisive runtime
separation.** The theory predicts a smaller worst-case polynomial surrogate,
not the observed percentage runtime improvement. Selecting this baseline
does not require a statistically significant timing victory. Further
coefficient or phase searches are not part of the accepted selection rule.

## Manuscript wording and attribution

> We include a prescribed period-two relaxation policy motivated by weighted
> polynomial smoothing. The coefficients minimize
> max_{t in [0,1]} t[(1-at)(1-bt)]² over the same relaxation-weight grid used
> by the learned controller. Evaluating the analytic stationary points for
> all 861 unordered grid pairs gives {a,b} = {2.85,1.10}. We apply the pair
> in high–low order, resetting its phase at each primary solve. This is a
> grid-constrained minimizer of the normalized smoothing surrogate, not a
> claim of optimality for full multilevel completion time.

Cite Lottes for the underlying weighted minimax construction and the
continuous fourth-kind Chebyshev polynomial. Supply our finite-grid
calculation in the reproduction material. **Do not attribute the specific
discrete pair (2.85,1.10) to Lottes.** The continuous polynomial is existing
literature; our grid specialization and its application to the prescribed
full-cycle baseline should be described at that scope.

## Reproduction and source record

- [Scalar verification script](../../experiments/paper_final/verify_period_two_minimax.py):
  analytic extrema at 80-digit precision, exact rational perturbation
  identities, and every unordered grid pair. It reads no PDE runtime data.
- [Saved scalar verification](period_two_minimax_verification_20260928.json).
- Development paired experiment report (`20260928_pair285_vs290_joint_6seeds_100cases`, retained on the development branch),
  with raw.jsonl, protocol.json, summary.json and independent_audit.json in
  the same directory.
- [Development paired timing runner](https://github.com/EriseHe/RLTune-BoomerAMG/blob/online-bandit-rl/experiments/archive/paper_development/compare_05_periodic_pairs.py).
- [Historical anchored derivation and implementation audit](anchored_schedule_review_20260927.md).
- An external clarification supplied during the original derivation review.
- User's follow-up review in this chat, accepted 2026-09-28. The peak values
  and perturbation identities were independently rechecked while saving
  this note. Use this note as the current unanchored-baseline reference;
  preserve the older anchored note as Run 04 provenance.

From the repository root, reproduce the scalar check with the pinned
experiment environment:

```sh
python -m experiments.paper_final.verify_period_two_minimax
```
