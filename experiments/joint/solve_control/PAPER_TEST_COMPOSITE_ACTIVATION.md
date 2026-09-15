# 60³ composite reliability activation: experiment design

Status: implementation and launch validation completed on 2026-09-15.
The continuous gate and shared online prefix are implemented in the existing
joint runner. The pilot config is `configs/paper_test_n60_composite_activation_seed1.json`.
Existing PAPER_FINAL and historical activation configurations retain their
original meaning. Launch metadata and checks are saved beside the pilot results.

## 1. Question and scope

Keep the unacceptable conditional first-attempt failure risk at **5%**.
Test whether a gate that accommodates unknown low failure rates and unknown
onset times is a useful alternative to starting RL on problem 1001.

There are two separate outcomes:

1. When does the reliability evidence justify triggering under the specified
   statistical model, and how much computation does monitoring require?
2. What is the complete 5000-problem cost of the resulting joint learner,
   relative to the prescribed-start learner?

The false-activation theorem concerns the first question. Runtime superiority
is an experimental outcome, not a consequence of that theorem.

The first test uses one scalar anisotropic diffusion stream on a 60³ grid,
one previously specified new replicate, and **two methods**:

| ID | Setup | Solve control |
|---|---|---|
| `fixed_start` | Online LinUCB | Reference solve on 1–1000, LSTDQ on 1001–5000 |
| `composite_start` | Same initial LinUCB | Reference solve until crossing; LSTDQ from the following problem |

Proposed experiment name: `paper_test_n60_composite_activation_seed1`.
The two methods each account for all 5000 problems. No additional free training
prefix is provided. This diagnostic does not add Default, setup-only LinUCB,
legacy 8D, mean-augmented contexts, or an independent legacy-gate branch.
Its primary comparison is directly against `fixed_start`, not an old run's
Default. It does not complete PAPER_FINAL Modules 1 and 2 by itself.

## 2. Fixed inputs and learner settings

Reuse the problem, stream, seed tuple, setup and solve sections from
`configs/PAPER_FINAL/PAPER_FINAL_diffusion_n60_seed1.json`:

- 5000 matrices/RHS inputs; scalar anisotropic diffusion; grid 60³.
- Spatially constant directional diffusion coefficients sampled on [1, 1000],
  varying across problems; zero advection.
- Shared problem context `(1, log(cx)/log(1000), log(cy)/log(1000), log(cz)/log(1000))`.
  Both learners omit mean; the intercept is included once.
- Existing resolution-20 setup space and structured-512 AOT candidates.
- 18/18/9 smoothing/coarse profile, tolerance 1e-6, 50-cycle attempt cap,
  and the current bounded-recovery protocol.
- Corrected state encoding, recursive LSTDQ-v3, weights `1.00:0.05:3.00`,
  nine RBF centers, and unchanged ridge, beta, trace and epsilon settings.
- Setup continues learning after RL activation. Each controller starts with
  its normal untrained state and exploration schedule when enabled.

This uses **new replicate 1**, not historical C or D selected by performance.
Its first generator seed is 56700120, shuffle seed 56748120, bandit seed
56760120, controller seed 56766120, method-order seed 56772120, and both
method seed offsets are zero. The complete generator seed tuple remains in
the source config. Its input-stream SHA-256 is
`e79ce1e088712599abd69a0b6b48cfbea54fbcf6c684b057353bd64af38ec823`.

Run native solves sequentially with the existing single-thread settings.
After branching, randomize the two methods' order on each problem using the
prespecified method-order RNG. Record actual execution order.

## 3. The new gate

Fix before observing this run:

\[
p_0=0.05,\qquad \delta=0.05,\qquad N=5000,\qquad H=N-1=4999.
\]

Here H is the last useful monitoring time: a crossing after problem H can
still enable RL on problem N. This is a budget boundary, not a minimum
prefix, observation window, or forced activation deadline.

Before activation, observe exactly once per problem

\[
Y_t=\mathbf 1\{\texttt{first\_primary\_status}\ne\texttt{success}\}.
\]

Use the same first-attempt definition as the old gate, including construction
failure or primary nonconvergence before reselection/default recovery. A
successful recovery does not erase the first failure. Distinguish these
failure types in reporting. The monitored procedure is learned setup paired
with reference solve, not hierarchy construction alone.

For each q in (0, p0), define

\[
L_i(q)=\left(\frac q{p_0}\right)^{Y_i}
       \left(\frac{1-q}{1-p_0}\right)^{1-Y_i}.
\]

Use a prospectively fixed **continuous uniform mixing distribution**
\(\pi(dq)=dq/p_0\) on (0, p0). For each onset k, mix cumulative products:

\[
E_{k,t}=\frac1{p_0}\int_0^{p_0}\prod_{i=k}^tL_i(q)\,dq,
\qquad k\le t.
\]

Uniform mixing gives positive mass near every strictly low failure rate;
it is a simple specified design, not a claim that this prior is optimal.
Do not replace this expression by a product of fixed single-step averages:
that would collapse to the point alternative q = p0/2.

Allocate onset weights 1/H and include dormant tests:

\[
M_t=\frac1H\sum_{k=1}^tE_{k,t}+\frac{H-t}{H},
\quad 0\le t\le H,\qquad M_0=1.
\]

\[
\tau=\inf\{1\le t\le H:M_t\ge20\}.
\]

Enable RL on problem tau+1. If there is no crossing by H, keep reference
solve through problem N and record `first_rl_problem = null`. Once triggered,
stop feeding observations into the gate; RL-controlled outcomes are not
reference-policy evidence. Do not retune parameters or force activation at
problem 1001, or at any other point, after inspecting the trajectory.

### Guarantee and detection interpretation

Define the conditional risks on the hypothetical process that continues
reference solve and setup learning, with the executed process agreeing with
it until activation. Under p_i >= p0 almost surely for every monitored i,

\[
\mathbb E[L_i(q)\mid\mathcal F_{i-1}]
=1+\frac{(p_i-p_0)(q-p_0)}{p_0(1-p_0)}\le1.
\]

Products, fixed alternative mixtures and normalized onset mixtures therefore
form nonnegative supermartingales, and Ville's inequality gives
\(\Pr(\tau\le H)\le\delta\). If the unacceptable-risk condition holds up
to a specified change time nu, it similarly controls triggering before nu.
Independence is not required for this false-activation result.

This rejects persistent unacceptable reference risk. It is not a confidence
certificate for an arbitrarily evolving instantaneous risk or for the new RL
policy. Interpreting it as entry into a durable reliable regime requires a
corresponding persistence assumption. Detection-delay results also require a
strict separation below p0, despite the algorithm not requiring a chosen
post-change rate as input.

For orientation, under a stationary post-change Bernoulli rate r, the
best matching point alternative has mean log increment D_KL(r || 0.05).
A continuous mixture also pays an adaptation penalty. The following are
algebraic information rates, **not measured delays or delay bounds**:

| r | D_KL(r || 0.05) | Old q=0.01 point-alternative mean log increment |
|---|---:|---:|
| 1% | 0.02473615 | 0.02473615 |
| 2% | 0.01214296 | 0.00822934 |
| 4% | 0.00112671 | -0.02478428 |

Thus 4% can support the composite alternative, but its information rate is
about 22 times smaller than at 1%. Nonactivation within the finite budget
remains a legitimate result. Composite mixing need not be faster than the
correctly chosen point alternative at 1%.

## 4. Identical prefix and controlled branching

Do not launch two independent setup learners from problem 1 and rely on equal
seeds to keep them equal. Timing feedback caused divergence even before RL
activation in the earlier C/D study.

1. Execute one shared setup/reference-solve path. Update the composite gate
   after each completed problem. Instantiate the two controllers with equal
   initial states and RNGs, without training them on this prefix.
2. Let h be the earlier of problem 1000 and a composite crossing on the shared
   path. After completing h, save the shared mutable setup state and the
   execution counters, then initialize both branches from that state.
3. If h < 1000, enable RL only in the composite branch on h+1. The fixed
   branch continues reference solve through problem 1000.
4. If no crossing has occurred by problem 1000, fork there. The fixed branch
   starts RL on 1001. The composite branch continues its own reference-solve
   setup learning and gate updates until crossing or H.
5. A crossing at exactly 1000 enables both controllers on 1001. Afterwards,
   both branches have independent mutable states and their own measured
   feedback. Do not resynchronize or force equal setup actions.

Copy or restore all decision-relevant state: regression and failure-model
arrays, observation/update counts, candidate statistics, RNG state, AOT
cursor, and the previous-update timing estimate used by the setup wrapper.
Keep independent schedule cursors over matching candidate data. Record and
check equality at the fork. Restore prefix histories after checkpoint loading:
structured candidate selection uses the last history entry as an anchor, so
that entry is also decision state, not just provenance.

The existing `clone_for_independent_updates` rejects attached AOT schedules.
Use the existing `save_mutable_state` / `load_mutable_state` path, which already
supports candidate statistics, RNG and AOT cursor, in separately constructed
branches. Do not circumvent that restriction with a generic deep copy or
share a mutable candidate cursor. Reuse the existing recovery and per-method
execution functions after the fork.

This design removes the pre-intervention learning-path difference. Subsequent
setup divergence is part of the effect of adopting the complete activation
protocol. A single wall-clock realization still does not establish an
expected causal effect over training randomness or machine conditions.

## 5. Cost accounting and reporting

Each method's full-stream cost includes the same shared prefix's measured
setup, solve and bandit cost, counted **once per method**. Physically solving
the prefix once is an experimental coupling, not free method training.
Mark shared rows explicitly rather than presenting them as two independent
timing measurements.

Charge gate computation only to `composite_start`, including its monitoring
cost during the shared prefix. Preserve the existing gate timing convention:
monitoring is additional controller overhead in the reported cost, and is
not retrospectively inserted into already-issued setup feedback. Shared
setup labels are the same observed reference-solve feedback.

Use the existing recorded-online metric:

\[
C_t=C_t^{setup}+C_t^{solve}+C_t^{controller}+C_t^{bandit}.
\]

Include unsuccessful attempts, reselection and recovery exactly once.
Activation runtime is a reported subset of controller overhead, and recovery
runtime a subset of the relevant setup/solve totals. Neither is added twice.
Report one-time candidate preparation, checkpoint/fork work, other reporting
I/O and whole-process elapsed time separately; do not label the per-problem
metric as the cost of the entire application.

Primary outcomes:

- First RL problem, or nonactivation; prefix/fork index h.
- Cumulative difference `C_composite - C_fixed` over all 5000 problems, and
  `100 * (1 - C_composite/C_fixed)`. Positive reduction means composite saves time.
- Full-stream setup, solve, controller, bandit, gate and recovery components.
- First-attempt failure types and unrecovered failures.

Secondary outcomes:

- Cumulative cost-difference curve and log(M_t) up to the actual crossing.
- Last-1000 cost breakdown and cycle counts.
- Descriptive differences before either policy differs, while only one
  controller is enabled, and after both are enabled, where those intervals exist.
- Relaxation trajectories from the actual controller logs.

Use paired problem identities and the actual first-attempt outcome. Do not
treat the 5000 adaptively generated costs as independent training replicates,
or select the reported time window after seeing which method wins.

The old point-alternative gates with heavy-tail or uniform onset weights can
be replayed on the **observed reference-solve prefix** as secondary evidence
diagnostics. Record the available prefix length and mark noncrossings as
censored. Such replay is not a closed-loop runtime result for the old gate;
do not continue it on RL outcomes or feed its decisions into this run.

## 6. Implementation checks before native execution

Extend `joint_rl_activation.py` with an explicit new mode while preserving the
legacy mode. Extend the common online execution path only as needed for the
opt-in shared prefix; ordinary PAPER_FINAL and old configurations must retain
their execution semantics. Reuse current config parsing, stream validation,
recovery accounting and reporting rather than adding a second solver loop.

For f failures and s successes on a suffix, the continuous integral is

\[
E(f,s)=\frac{B_{p_0}(f+1,s+1)}{p_0^{f+1}(1-p_0)^s},
\]

where B is the unregularized incomplete beta integral. Evaluate this with
appropriate scaling/logarithms, not naive division of underflowed quantities.
An especially useful check is E(f,0)=1/(f+1); the numerator can underflow in
ordinary floating point even though this ratio is moderate. The all-success
identity is E(0,s)=[1-(1-p0)^(s+1)]/[p0*(s+1)*(1-p0)^s].

Required validation includes short-sequence integral/onset checks, all-failure
and all-success limits, next-problem activation, no forced activation, and
the last useful monitoring time. A finite positive quadrature rule defines
a discrete mixture; do not silently describe it as the continuous prior.

Use mocked native outcomes to cover crossing before, at, and after 1000, plus
nonactivation. Verify fork-state equality, independent subsequent mutation,
controller counters/RNGs, AOT cursor alignment, shared-row accounting, and
retained first-failure information. Check old gate/config regressions and
the existing PAPER_FINAL validation. Measure gate overhead independently
before the native run. Synthetic Bernoulli checks illustrate behavior but
do not replace the supermartingale proof or validate an AMG persistence model.

## 7. Budget and interpretation

Nominal budget: **two methods × 5000 = 10,000 problem tasks**.
Executing the shared prefix once uses `10000 - h` distinct problem executions
(roughly 9000–10000), before any retry/recovery work. Gate diagnostics require
no additional AMG solves. There is one new training/problem seed, so this is
an exploratory comparison, not a claim of seed-robust superiority.

Report an earlier, later, coincident, or absent trigger as observed. A useful
rule need not beat problem 1001 on every stream; runtime and reliability
interpretations remain separate. Any extension to diffusion–advection, another
seed, or other grid sizes is a separately specified follow-up budget.

The relevant statistical distinction between normalized e-process mixtures
and ARL-controlled e-detectors is described in
[Shin, Ramdas and Rinaldo, Remark 2.7](https://nejsds.nestat.org/journal/NEJSDS/article/59/text).
See also [Waudby-Smith and Ramdas](https://arxiv.org/abs/2010.09686) for betting
and mixture constructions. The validity calculation above is specific to
the conditional first-failure-risk null used here.
