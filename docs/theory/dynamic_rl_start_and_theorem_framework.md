# Dynamic RL activation and a defensible theoretical framework

Working theory note, 2026-09-12. This is a proposed analysis, not a claim that the current joint implementation satisfies every assumption below. No new PDE experiments were run for this note.

## 1. Experimental contract and evidence

The current main configuration uses scalar anisotropic diffusion, Canonical 8D setup context, LinUCB-v5 with structured candidate sets, and Recursive LSTDQ-v3. The corrected 60³ experiment has 5,000 problems in the total budget, with fixed-weight solve for the first 1,000 and joint learning thereafter. Setup continues learning after solve control begins. The solve configuration is weighted ℓ₁-Jacobi on the down/up sweeps, direct coarse solve (18/18/9), relative tolerance 10⁻⁶, and a 50-cycle cap. The action grid is 1, 1.05, ..., 3.

The corrected state encoder distinguishes the actual setup categories and maps the previous weight over its actual range. The solve representation has 35 state features and 9 action-basis features, hence 315 joint features. Its update uses the executed next action, discount 1, and trace parameter 0.8: it is a SARSA-type recursive LSTDQ(λ) control scheme.

In the latest corrected five-method run, Canonical + LSTDQ averaged 209.502 ms native time versus 372.856 ms for the baseline, a 43.81% reduction. Including the reported learner/controller overhead gives 221.270 ms, a 40.66% reduction. These are different cost definitions. The native reduction agrees with the corrected three-seed range, 43.55%–44.52%; that agreement does not isolate a causal effect of the encoder repair, since timing conditions and resulting learning paths differed across runs.

The previous prefix audit covered 11 experiment directories and 43 learning trajectories. These are descriptive trajectories, not 43 independent replications. In the latest Canonical staged branch, first-attempt failures in cases 1–100, 101–200, 201–400, 401–600, 601–800, and 801–1000 were respectively 76%, 39%, 5.5%, 1%, 1%, and 1.5%. All 133 prefix failures were solve nonconvergence. The full 5,000-problem run had 142 primary failures, all recovered.

The audit found occasional rebounds and materially more persistent failure in advection experiments. A permanent sharp change is therefore a useful model to analyze, not an established property of all trajectories.

Sources: [latest report](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/joint/paper_n60_canonical8d_vs_physics_linear_coarse9_staged1000_5k_tol1e6_encoderfix_terminal_20260912/screen_report.md), [prefix audit](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/diagnostics/paper_convergence_prefix_audit_20260912/audit.json), [corrected seed report](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/joint/paper_final_n60_canonical8d_lstdq_v3_seed_stability_5k_encoderfix_20260912/screen_report.md), [experiment configuration](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/joint/solve_control/configs/paper_n60_canonical8d_vs_physics_linear_coarse9_staged1000_5k_tol1e6_encoderfix.json).

## 2. What the activation problem actually estimates

For problem t, during the setup-only phase, define

\[
Y_t=\mathbf 1\{\text{the first primary attempt does not reach the prescribed tolerance within its budget}\}.
\]

Count failure before retry/fallback, once per problem. A recovered failure remains Y_t=1. For a general deployment, inability to construct the primary hierarchy also counts as primary unreliability; alternatively, it can be reported as a separate event while remaining in the gate's denominator.

Let \(\mathcal F_{t-1}\) contain the observed learning history before the next problem is revealed, and define

\[
p_t=\mathbb E[Y_t\mid\mathcal F_{t-1}].
\]

This is the conditional failure risk of the deployed setup-selection procedure paired with the fixed solve policy. It averages over the next problem and any relevant algorithmic randomness. It is not the failure probability for every individual PDE context or every candidate action. The probability model is over problem streams/randomness; conditioning on a completely fixed deterministic benchmark and all its random seeds would not yield an empirical Bernoulli sampling model.

The relevant scientific question is whether primary reliability has improved enough to enable solve-policy learning. It is not whether setup parameters, predictions, or the setup optimizer have converged.

There is no distribution-free implication from an arbitrarily long successful history to low next-step risk: two processes can have identical observations through time t and different next-step failure probabilities. In particular, one can change to certain failure immediately after a data-dependent activation time. Thus a stopping rule needs an explicit target and an explicit statement of what its error control means.

A fixed window is not inherently mathematically invalid. The difficulty with the current rule is its heuristic calibration and the gap between a historical rate and current/future risk. Repeatedly checking ordinary 95% binomial intervals does not supply simultaneous 95% coverage. CUSUM average-run-length control is also different from controlling the probability of any false activation.

## 3. Recommended criterion: a mixture sequential test

Choose, before inspecting the evaluation trajectories:

- \(p_{\rm bad}\in(0,1)\): an operationally unacceptable primary-failure rate;
- \(p_{\rm good}\in(0,p_{\rm bad})\): a design rate at which timely detection is desired;
- \(\delta\in(0,1)\): the allowed probability of a false activation under persistent unreliability.

The gap between the two rates is an indifference region, not a claim that risk makes a discontinuous jump. For illustration only, one might use \(p_{\rm bad}=0.05\), \(p_{\rm good}=0.01\), and \(\delta=0.05\). These numbers are not calibrated recommendations for this experiment.

Define the one-step evidence factor

\[
\ell_t=
\left(\frac{p_{\rm good}}{p_{\rm bad}}\right)^{Y_t}
\left(\frac{1-p_{\rm good}}{1-p_{\rm bad}}\right)^{1-Y_t}.
\tag{1}
\]

For each possible onset k, let

\[
L_{k,t}=
\begin{cases}
1,&t<k,\\
\prod_{i=k}^{t}\ell_i,&t\ge k.
\end{cases}
\]

Choose deterministic weights \(\omega_k>0\) with \(\sum_{k\ge1}\omega_k=1\). A simple horizon-free choice is

\[
\omega_k=\frac1{k(k+1)}.
\]

Combine all onsets, retaining the mass assigned to as-yet-unstarted tests:

\[
M_t=\sum_{k=1}^{t}\omega_kL_{k,t}+\sum_{k>t}\omega_k,
\qquad M_0=1.
\tag{2}
\]

The activation boundary and first RL-controlled problem are

\[
\tau=\inf\{t\ge1:M_t\ge1/\delta\},
\qquad \text{first RL problem}=\tau+1.
\tag{3}
\]

If the set is empty within the stream, solve remains fixed for that stream. Any imposed deadline that activates RL without crossing (3) is a separate protocol decision outside this certificate.

The statistic tests every problem and accounts for every possible onset. Early failures penalize tests that started early, but do not contaminate all subsequently started tests. There is no fixed observation window or arbitrary number of passing blocks.

For the displayed weights, implementation needs only a scalar accumulator:

\[
B_0=0,\qquad
B_t=\ell_t\left(B_{t-1}+\frac1{t(t+1)}\right),
\qquad
M_t=B_t+\frac1{t+1}.
\tag{4}
\]

Thus time and storage per problem are O(1). Compute in log space for numerical stability. This is an algebraic recursion, not an implemented change to the runner.

For a known maximum horizon T, \(\omega_k=1/T\), k=1,...,T, is an alternative with a simpler onset penalty \(\log T\). The horizon must be chosen prospectively; extending it after observing evidence requires a new valid allocation. Other predetermined proper weights preserve validity and change detection delay. The weights represent an explicit cost of searching over unknown onset times, not an unacknowledged window length.

### Proposition 1: control of false activation

Suppose \(p_t\ge p_{\rm bad}\) almost surely at every monitored time. Then

\[
\Pr(\tau<\infty)\le\delta.
\tag{5}
\]

More locally, if the inequality holds for t<ν, with deterministic ν, then

\[
\Pr(\tau<\nu)\le\delta.
\tag{6}
\]

No independence, identical distributions, or linear realizability is required.

**Proof.** Conditional on \(\mathcal F_{t-1}\),

\[
\mathbb E[\ell_t\mid\mathcal F_{t-1}]
=p_t\frac{p_{\rm good}}{p_{\rm bad}}
+(1-p_t)\frac{1-p_{\rm good}}{1-p_{\rm bad}}
\le1.
\]

The expression is decreasing in p_t and equals one at \(p_t=p_{\rm bad}\). Each \(L_{k,t}\) is consequently a nonnegative supermartingale starting at one under the null. Their proper mixture is also one. Ville's inequality gives (5). Stopping the argument at ν−1 gives (6). ∎

This is a direct specialization of the nonnegative-supermartingale approach to sequential inference; it is not presented as a new general statistical theorem. See [Howard et al., time-uniform Chernoff bounds](https://arxiv.org/abs/1808.03204) and [Howard et al., confidence sequences](https://arxiv.org/abs/1810.08240). For related change-detection constructions and the distinction between false-alarm probabilities and average run length, see [Shin, Ramdas, and Rinaldo, E-detectors](https://nejsds.nestat.org/journal/NEJSDS/article/59/text).

### Proposition 2: finite detection delay after durable improvement

Suppose additionally that, under continued setup-only operation, \(p_t\le p_{\rm good}\) almost surely for every t≥ν. Define

\[
I=D_{\rm KL}\bigl(\operatorname{Bern}(p_{\rm good})
\Vert\operatorname{Bern}(p_{\rm bad})\bigr)>0,
\]

\[
a=\log\frac{1-p_{\rm good}}{1-p_{\rm bad}},\qquad
b=\log\frac{p_{\rm good}}{p_{\rm bad}},\qquad
R=a-b>0.
\]

For any \(\eta\in(0,1)\), if an integer n satisfies

\[
nI-R\sqrt{\frac n2\log\frac1\eta}
\ge \log\frac1{\delta\omega_\nu},
\tag{7}
\]

then

\[
\Pr\{\tau>\nu+n-1\}\le\eta.
\tag{8}
\]

With the pre-change condition of Proposition 1,

\[
\Pr\{\nu\le\tau\le\nu+n-1\}\ge1-\delta-\eta.
\tag{9}
\]

**Proof.** The increment \(\log\ell_t=a+(b-a)Y_t\) lies in [b,a]. Its conditional expectation is at least I when \(p_t\le p_{\rm good}\). Conditional Hoeffding concentration therefore gives

\[
\Pr\!\left\{
\sum_{t=\nu}^{\nu+n-1}\log\ell_t
<nI-R\sqrt{\frac n2\log\frac1\eta}
\right\}\le\eta.
\]

On the complementary event, (7) implies
\(\omega_\nu L_{\nu,\nu+n-1}\ge1/\delta\), hence \(M_{\nu+n-1}\ge1/\delta\). Combine with (6). ∎

The leading evidence requirement is \(\log(1/(\delta\omega_\nu))/I\), with the explicit finite-probability correction in (7). This is a scale calculation, not an exact expected-delay formula or a minimax-optimality claim. With \(\omega_\nu=1/[\nu(\nu+1)]\), later onsets incur only a logarithmic additional evidence cost.

For the illustrative 5%/1% choice, a success contributes about 0.04124 log-evidence and a failure −1.60944. At 1% actual failure risk the expected increment is about 0.02474. Occasional failures are therefore allowed, but a sustained low-risk run is needed. A stringent certificate need not activate earlier than 1,000 on every trajectory.

### Precisely what is, and is not, certified

Without an assumption relating past and future risk, crossing (3) rejects the persistent-unreliability hypothesis. It does not itself prove \(p_{\tau+1}\le p_{\rm bad}\), and certainly does not prove that the newly activated RL controller has that risk.

Under a permanent two-regime model, (9) has the useful interpretation that activation occurs after the reliable regime begins, with controlled early-activation probability and finite detection delay. A gradual decline with an interval between the two thresholds can be handled by distinguishing the end of the certified bad regime from the start of the good regime. No delay guarantee is promised while risk remains inside the indifference region.

The actual setup policy continues exploring and can experience rebounds. Thus the paper should call this a sequential reliability-based activation rule and state the durable-improvement assumption for its readiness interpretation. It should not label the event “setup convergence” or an unconditional safety guarantee.

If a single good design rate is undesirable, mix (1) over several predetermined alternatives below \(p_{\rm bad}\), with proper weights. The same proof applies. This costs a small number of scalar recursions and an additional log-weight term in the delay bound. Choosing an alternative retrospectively without a mixture or multiplicity correction is not covered.

## 4. Stronger current-risk statements require a drift model

A complementary formulation makes the missing assumption especially visible. For every suffix [k,t], let n=t−k+1 and

\[
\widehat p_{k,t}=\frac1n\sum_{i=k}^tY_i,\qquad
\bar p_{k,t}=\frac1n\sum_{i=k}^tp_i.
\]

Allocate \(\delta_{k,n}=\delta/[k(k+1)n(n+1)]\). Conditional Hoeffding concentration and a union bound yield, simultaneously for all k and t, with probability at least 1−δ,

\[
\bar p_{k,t}\le
\widehat p_{k,t}+
\sqrt{\frac{\log(k(k+1)n(n+1)/\delta)}{2n}}.
\tag{10}
\]

If a known, valid drift envelope satisfies

\[
p_{t+1}\le p_i+d_{i,t+1}\quad\text{for every }i\le t,
\]

then a current-risk upper bound is

\[
U_t=\min\left\{1,\ \min_{1\le k\le t}
\left[
\widehat p_{k,t}
+\sqrt{\frac{\log(k(k+1)n(n+1)/\delta)}{2n}}
+\frac1n\sum_{i=k}^td_{i,t+1}
\right]\right\}.
\tag{11}
\]

The stopping time \(\tau_U=\inf\{t:U_t\le p_{\rm tol}\}\) satisfies

\[
\Pr\{\tau_U<\infty,\ p_{\tau_U+1}>p_{\rm tol}\}\le\delta,
\tag{12}
\]

for the continued fixed-solve process. Formula (10) alone only concerns averages of historical conditional means; it cannot be substituted directly into (12).

For nonincreasing risk one can set d=0. For a known one-step upward drift bound \(p_{j+1}-p_j\le v\), the correction is \(v(n+1)/2\). The minimization in (11) then selects a history length through an explicit tradeoff between sampling error and drift. Neither monotonicity nor a suitably small v follows from ordinary LinUCB updates or from a smoothed plot.

This version is stronger in its target but stronger in assumptions and computational cost. The Hoeffding bound is intentionally elementary and can be conservative for rare failures. A sharper Bernoulli/martingale bound can replace it after the target and drift assumption have been settled.

## 5. Setup regret and payback: what can be proved for which algorithm

Fix the downstream solve and recovery policy. Let \(C_t\) be the candidate set available before the current cost observation, let \(\phi_t(a)\in\mathbb R^d\), and assume

\[
\mu_t(a)=\phi_t(a)^\top\theta_\star,\qquad
Y_t^{\rm cost}=\mu_t(a_t)+\xi_t,
\]

with bounded features and parameter norm and conditionally sub-Gaussian, mean-zero noise. Use a new notation \(Y_t^{\rm cost}\) to distinguish cost from the binary gate observation.

For
\(V_t=\lambda I+\sum_{i=1}^t\phi_i(a_i)\phi_i(a_i)^\top\),
a valid simultaneous radius has the familiar form

\[
\beta_t=
\sigma\sqrt{2\log
\frac{\det(V_t)^{1/2}}{\det(\lambda I)^{1/2}\delta}}
+\sqrt{\lambda}\,\|\theta_\star\|_{\rm bound}.
\tag{13}
\]

An LCB policy using this radius and minimizing over \(C_t\) has candidate-set pseudo-regret \(\widetilde O(d\sqrt T)\), under the standard boundedness/scaling conditions. History-dependent candidate generation is allowed; the current noise cannot be used to define the candidate set. This is an application of existing linear-bandit theory, not a new general regret result. See [Abbasi-Yadkori, Pál, and Szepesvári (2011)](https://proceedings.neurips.cc/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf).

The implemented structured-512 path enables alpha decay, and v5 inherits v4's coefficient \(\alpha_t=\alpha_0/\sqrt{t+1}\), up to indexing/burn-in. It is not (13). See [construction](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/joint/solve_control/setup_aware_compare_common.py:933) and [coefficient](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/setup/learners/linucb/SharedLinUCB_AMG_v4.py:676).

A useful deterministic inequality exposes the exact extra requirement. On the confidence event, put \(u_t(a)=\|\phi_t(a)\|_{V_{t-1}^{-1}}\) and \(a_t^\star\in\arg\min_{a\in C_t}\mu_t(a)\). The implemented LCB choice satisfies

\[
\mu_t(a_t)-\mu_t(a_t^\star)
\le(\beta_{t-1}+\alpha_t)u_t(a_t)
+(\beta_{t-1}-\alpha_t)_+u_t(a_t^\star).
\tag{14}
\]

**Proof.** Use confidence bounds on the chosen and oracle actions, with
\(\widehat\mu_t(a_t)-\alpha_tu_t(a_t)
\le\widehat\mu_t(a_t^\star)-\alpha_tu_t(a_t^\star)\). ∎

The first sum in (14) admits the usual elliptical-potential control. The second is a coverage deficit and is not automatically sublinear. One possible sufficient condition is \(\lambda_{\min}(V_t|_{\mathcal S})\ge\kappa t\) eventually on the identifiable feature span \(\mathcal S\); it gives \(u_t(a_t^\star)=O(t^{-1/2})\). This is a substantial persistent-excitation assumption, not a consequence of sampling 512 candidate actions. Only executed actions enter V. Using the feature span matters because Canonical 8D contains a redundant common-scale feature.

Forced initial-guess rounds must be accounted for separately; a fixed number contributes only a bounded additive term when per-round regret is bounded.

Thus there are two honest routes: analyze a confidence-calibrated variant, or retain the existing algorithm and supply assumptions controlling the second term in (14). Do not silently identify the current decay rule with the textbook confidence policy.

The theorem also needs an exact cost contract. If the regression target includes recovery or controller time, the oracle must use the same quantity. Correlated suffix-cost labels from several recovery decisions do not automatically form one unbiased bandit observation per problem. Warmup is included in the total problem budget.

Full-action-space regret additionally contains
\(\sum_t[\min_{a\in C_t}\mu_t(a)-\min_{a\in\mathcal A}\mu_t(a)]\).
Candidate-set regret alone does not control it.

Once RL changes the downstream solve policy, setup costs become policy-dependent. Fixed-downstream realizability does not establish stationary realizability for the joint run. The gate does not remove this issue. A joint guarantee would need a drift/tracking term, a suitable changing-parameter model, or a separately justified restriction.

For payback, define the candidate oracle's cumulative advantage \(G_T\), the learner's candidate regret \(R_T\), and overhead \(H_T\), without counting the same cost twice. The expected/pseudo accounting identity is

\[
\operatorname{Savings}(T)=G_T-R_T-H_T.
\]

If \(G_T\ge gT-B_G\), \(H_T\le hT+B_H\), \(g>h\), and \(R_T=o(T)\), eventual positive savings follows. For a quantitative payback bound, solve
\((g-h)T>R_T+B_G+B_H\).
Positive average oracle advantage by itself is insufficient when overhead consumes it. An assertion about realized savings holding for every sufficiently large T additionally requires a uniform stochastic bound or an appropriate strong law. This is naturally a corollary.

## 6. Nonstationary relaxation: repair and strengthen the solver result

For a fixed nonsingular system A and fixed hierarchy, write error propagation as \(e^+=E_H(w)e\). Residual propagation is \(r^+=AE_H(w)A^{-1}r\); the spectral radius is unchanged, while finite-tolerance norm behavior need not be.

The proposed spectral-rate inequality does not imply that the periodic policy beats every fixed weight for every initial residual. For example, with equal cycle costs,

\[
E_L=\operatorname{diag}(0.9,0.1),\qquad
E_H=\operatorname{diag}(0.1,0.9),
\]

the periodic product is \(0.09I\), with better worst-case asymptotic rate than either fixed action. But on \(e_0=(0,1)^\top\), fixed L contracts by 0.1 per step and is asymptotically faster than the periodic policy's 0.3 per-step rate.

### Proposition 3: a correct worst-case asymptotic comparison

Let the action set be finite, as in the implemented 41-weight grid. Assume fixed positive per-cycle costs, fixed linear operators, and \(0<\rho(E_H(w))<1\) for the fixed-action comparators. For a period σ let \(P_\sigma=E_H(w_p)\cdots E_H(w_1)\), \(c_\sigma=\sum_i\tau_H(w_i)\), and \(0<\rho(P_\sigma)<1\).

Define fixed-policy time by the first iterate whose induced residual-propagation norm is at most ε, and define periodic time using period-end iterates. Then

\[
\lim_{\epsilon\downarrow0}
\frac{T_w^{\rm wc}(\epsilon)}{\log(1/\epsilon)}
=\frac{\tau_H(w)}{-\log\rho(E_H(w))},
\]

\[
\lim_{\epsilon\downarrow0}
\frac{T_\sigma^{\rm wc}(\epsilon)}{\log(1/\epsilon)}
=\frac{c_\sigma}{-\log\rho(P_\sigma)}.
\tag{15}
\]

Consequently a strict gap in the proposed spectral contraction per unit cost yields a uniform small-tolerance advantage over the finite set of fixed comparators, in this worst-case sense.

**Proof sketch.** For a stable matrix B, \(\rho(B)^n\le\|B^n\|\), while for each \(r>\rho(B)\) there is a constant C_r with \(\|B^n\|\le C_rr^n\). These give matching leading hitting-time bounds as ε→0 and r↓ρ(B). Similarity by fixed A changes the norm constants but not the leading rate. A finite comparator set permits one common sufficiently small tolerance. Stopping inside a period can only improve the periodic method's time. ∎

A continuum of comparators requires a corresponding uniform argument. Unstable fixed comparators cannot meet arbitrarily strict worst-case tolerances. Nilpotent/zero-radius operators require a separate finite-termination statement. Actual finite-tolerance performance also depends on transient amplification, starting residuals, failure caps, and timing variability.

### Constructive witness inside an idealized two-grid model

There is a more informative result than merely assuming that a superior product exists. Consider

\[
A=\begin{pmatrix}2/3&-1/3\\-1/3&2/3\end{pmatrix},
\qquad D_{\ell_1}=I,
\qquad
P=\frac12\begin{pmatrix}\sqrt3-1\\\sqrt3+1\end{pmatrix}.
\]

Use one weighted ℓ₁-Jacobi pre-sweep and one identical post-sweep, \(S(w)=I-wA\), with exact Galerkin coarse correction
\(C=I-P(P^\top AP)^{-1}P^\top A\).
The two-grid error operator is \(E(w)=S(w)CS(w)\).

In A-energy coordinates and the eigenbasis of A, C becomes the projection onto \(u=(1,1)^\top/\sqrt2\), so

\[
\widetilde E(w)=v(w)v(w)^\top,\qquad
v(w)=\frac1{\sqrt2}(1-w/3,\ 1-w)^\top.
\]

Hence, over all \(w\in[1,3]\),

\[
\min_w\rho(E(w))
=\min_w\frac{(1-w/3)^2+(1-w)^2}{2}
=\frac15,
\quad w_{\rm best}=\frac65.
\tag{16}
\]

For \(w_L=1\) and \(w_H=2.9=29/10\), both on the experimental action grid,

\[
v(w_H)^\top v(w_L)=\frac1{90},\qquad
\rho(E(w_H)E(w_L))=\frac1{8100}.
\tag{17}
\]

With equal per-cycle costs, the periodic effective spectral factor is therefore \(1/90<1/5\), strictly better than every fixed weight. Meanwhile \(\rho(E(2.9))=65/36>1\): the high-weight cycle alone is unstable. The strict gap is robust to sufficiently small perturbations of this finite-dimensional model. The limiting pair 1 and 3 even has a zero two-cycle product, but the nonzero-radius example (17) avoids relying on exact annihilation.

Each fixed operator in this example is self-adjoint in energy coordinates, giving the uniform bound \(\|E(w)^n\|_A\ge5^{-n}\). If \(F=E(2.9)E(1)\), its rank-one form gives \(F^n=\rho(F)^{n-1}F\). These two identities directly supply the eventual norm-based time advantage, including uniformity over the continuum of fixed weights.

This is an existence theorem for a precisely specified two-grid model sharing weighted ℓ₁-Jacobi and exact coarse correction. It does not assert that the observed 60³ BoomerAMG hierarchies realize these matrices or that a particular high–low pair is optimal there.

Nonstationary relaxation has a substantial existing numerical-analysis literature. The manuscript should distinguish this solver mechanism from any novelty of learning a hierarchy- and residual-dependent feedback policy. Relevant primary sources include [Adsuara et al., scheduled relaxation and Richardson](https://arxiv.org/abs/1607.03712) and [Lottes, optimal polynomial smoothers for multigrid V-cycles](https://arxiv.org/abs/2202.08830). Whole AMG-cycle operators are not automatically polynomials in a single matrix.

## 7. What the LSTDQ statements can establish

### Representability

Writing \(Q_\theta(s,w)=b(w)^\top\Theta x(s)\) shows state–action interactions. A pairwise preference reversal is expressible when the relevant action-basis rows and history features distinguish the two cases. Merely listing last-weight and residual features is not a proof that arbitrary policies or true value functions are representable.

For the actual normalized compact-RBF basis, a concrete endpoint construction is available. Let \(b_1,b_9\) denote the basis components centered at weights 1 and 3, and set \(g(w)=b_9(w)-b_1(w)\). On the current grid, g has its unique minimum at 1 and maximum at 3. The endpoint basis components have disjoint support; normalization makes each attain its unique maximum at its own endpoint. With the encoded previous weight \(z=(w_{\rm last}-2)\), the class contains

\[
Q(s,w)=c+z\,g(w).
\]

Minimizing Q chooses 3 after previous weight 1 and chooses 1 after previous weight 3. The constant c can make all values nonnegative. This proves expressibility of a nonconstant alternating policy over the full grid, not that training finds it or that it is optimal.

The same action-scoring class could also be used by a history-augmented contextual bandit. Sequential credit assignment comes from action-dependent solver dynamics and the multi-step cost target, not from representability alone.

### Conditional preservation of a fixed-policy advantage

Use a finite horizon K, including an absorbing terminal state and a consistently defined failure/recovery terminal cost. Let \(Q_h^{\pi^\star}\) be cost-to-go for a reference policy, and suppose uniformly over relevant histories/states and actions that

\[
|\widehat Q_h-Q_h^{\pi^\star}|\le\varepsilon_Q.
\]

If \(\widehat\pi_h\) minimizes \(\widehat Q_h\), then

\[
J(\widehat\pi)\le J(\pi^\star)+2K\varepsilon_Q.
\tag{18}
\]

Therefore \(2K\varepsilon_Q<\Delta_{\rm fix}\) preserves strict superiority to the best fixed comparator under the same initial-state distribution and cost definition.

**Proof.** Approximate greediness gives
\(Q_h^{\pi^\star}(s,\widehat\pi_h(s))-V_h^{\pi^\star}(s)\le2\varepsilon_Q\).
The finite-horizon performance-difference identity sums these advantages along trajectories of \(\widehat\pi\). ∎

The reference need not be globally optimal. This is a generic approximate-policy-improvement proposition, not yet an LSTDQ estimation theorem. See [Lagoudakis and Parr, LSPI](https://jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf) for the broader policy-evaluation/improvement framework.

The actual policy uses an uncertainty bonus, clipping, and epsilon exploration. If its conditional mean selection slack in estimated Q is at most \(\zeta_h\), (18) becomes

\[
J(\widehat\pi_{\rm deployed})
\le J(\pi^\star)+2K\varepsilon_Q+
\mathbb E\sum_{h=0}^{K-1}\zeta_h.
\tag{19}
\]

Any controller cost omitted from Q must also be added when comparing wall time. The real cycle cap can provide K; an informal effective horizon cannot replace a termination assumption.

The 35-dimensional encoder has not been shown to be Markov-sufficient. Use full solver states or histories for the theorem, then state the approximation assumption for their encoded representation. Equal residual norms do not identify equal residual directions.

For a prospective LSTDQ-specific bridge, the deterministic identity

\[
\widehat\theta_N-\theta_\star
=A_N^{-1}\left[
\sum_{i=1}^Nz_i(c_i-v_i^\top\theta_\star)
-\lambda_{\rm ridge}\theta_\star
\right],
\quad v_i=\psi_i-\gamma\psi_{i+1},
\tag{20}
\]

separates Bellman approximation/bias, moment concentration, and conditioning of \(A_N\). Under an appropriate fixed-policy moment model, bounded moments, and a nondegenerate normalized A_N, this provides a route to parameter and uniform Q bounds. The evolving joint behavior policy does not automatically satisfy those fixed-policy assumptions.

### Recursive equals batch

For the same committed transitions, executed next actions, terminal indicators, trace resets, discount, trace parameter, and ridge,

\[
z_i=\gamma\lambda_{\rm trace}z_{i-1}+\psi_i,\quad
A_N=\lambda_{\rm ridge}I+\sum_i z_i(\psi_i-\gamma\psi_{i+1})^\top,\quad
b_N=\sum_i z_ic_i.
\]

If every required prefix A_i is nonsingular and the inverse updates are valid, Sherman–Morrison gives \(\theta_N^{\rm rec}=A_N^{-1}b_N\) in exact arithmetic. This is equivalence to a batch accumulation of the same historical estimating equations, not to LSPI recomputing every target under a newly chosen single policy.

Positive ridge alone does not guarantee nonsingularity for this nonsymmetric estimating matrix. The implementation's pseudoinverse safeguards and floating-point behavior need separate qualification when the nonsingularity assumption fails. See [recursive mean update](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/solve/controllers/recursive_lstdq/v1.py:103).

### Optional covariance analysis

Episode clustering addresses dependence within solves, but does not itself establish confidence coverage. V3 accumulates episode scores evaluated at different historical post-fit parameters. Consistency would require an appropriate parameter limit, negligible historical-parameter evaluation error, a nonsingular limiting estimating matrix, moment/Lindeberg conditions, and either cross-episode martingale orthogonality or a covariance estimator accounting for cross-episode dependence. Fixed-policy asymptotic intervals would still not be anytime-valid exploration bounds. This can remain outside the principal theorem claims.

## 8. Suggested paper scope and genuinely necessary validation

A coherent three-part theory is:

1. Candidate-set cost learning, with payback as a corollary and an explicit distinction between the confidence-calibrated and implemented decay rules.
2. Sequential reliability activation, with false-activation control and detection delay under a stated durable-improvement model.
3. Constructive nonstationary solver advantage, followed by conditional preservation under sufficiently accurate action-value estimates.

Representability and recursive equivalence are supporting propositions. These are conditional modules; they do not constitute an end-to-end guarantee for the adaptive joint learner merely by being placed in sequence.

The immediate work can remain theoretical. Existing histories suffice to compute the proposed activation statistic and audit what it would have observed, without new PDE solves. Such a replay would describe the gate only; it would not reconstruct the changed trajectory caused by earlier RL activation. No gate replay or calibration was performed for this note.

Two additional empirical checks become necessary only if the corresponding stronger claims enter the paper:

- “The deployed dynamic-start method improves on fixed-1000”: this requires prospective runs with the gate fixed before evaluation, since activation changes both learners' future data.
- “The learned feedback policy outperforms the best fixed or predetermined periodic relaxation on these AMG systems”: this requires matched systems, hierarchies, initial residuals, stopping rules, and timing definitions. The current default-weight comparison does not establish the positive \(\Delta_{\rm fix}\) premise. A fixed-periodic comparator is needed for a claim about feedback's advantage over scheduling.

Neither a proof of the conditional propositions nor the constructive two-grid witness requires another broad experiment sweep. The critical next theoretical decisions are the operational risk target, the meaning of the regime assumption, the exact cost contract, and which algorithm the regret statement describes.
