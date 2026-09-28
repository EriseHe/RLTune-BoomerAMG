# SISC paper completion plan: online policy comparison, matched evaluation, and theory

**Current planning revision: September 28, 2026.** This documentation-only update implements the user's requested module numbering and prospective experiment design. It does not launch a run, modify the learner/solver, change any completed result, or authorize overwriting existing outputs.

**Module 04 stays unchanged.** The newly proposed complete online solve-policy comparison is **Module 05**. The previously named Module 05 frozen-policy study becomes **Module 06**. Theory integration becomes **Module 07**.

The full earlier plan, including the detailed T1–T6 mathematical review, September 15–17 execution history, numerical-fix record and original task checklist, is preserved unchanged in the [pinned historical version](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md). Its blob is `398449afbe5f65b87132371c8c34759aa1808365`. This current file supersedes its operational schedule and module numbers, not its recorded historical evidence or qualified mathematical statements.

The [experiment index](../../experiments/paper_final/README.md) has the same current numbering. Existing scripts, imports, filenames, result directories and archived manifests keep their historical names. Logical renumbering is not a filesystem migration.

## 1. Contribution and division of evidence

The paper studies online BoomerAMG setup–solve learning for cumulative recorded solver cost on related linear-system streams. Keep three different comparisons separate:

- **Module 04:** the completed Default / setup-only LinUCB / LinUCB–LSTDQ full-system comparison and its component/payback analysis.
- **Module 05:** a new full online comparison in which prescribed and learned solve policies each obtain their own independently adapting setup learner. This asks whether learned solve control is worthwhile relative to a strong inexpensive periodic alternative after both are allowed to find compatible hierarchies.
- **Module 06:** frozen solve policies evaluated on identical hierarchies from explicitly named setup sources. This measures conditional numerical/controller effects and cross-source transfer.

An independently trained setup learner is not guaranteed to be the optimum for its solve policy. A setup-only hierarchy source has been trained under weight one; a Joint source has been trained under an evolving controller after its initial weight-one prefix. Neither is a universally neutral hierarchy distribution, but either supplies a valid conditional matched-hierarchy comparison. Crossed sources reveal compatibility; independent online branches answer the complementary complete-system question.

Do not require a proof that RL is indispensable. Report what additional within-solve variation, feedback and online adaptation accomplish relative to the tested alternatives. Good periodic schedules may capture much of the benefit; this is an informative possible result, not a reason to alter the outcome criteria.

## 2. Current logical module map

| Module | Priority | Purpose | Status / scope |
|---|---|---|---|
| 01 | P0 | Numerical correctness, inverse recovery, timing and stopping tests | Preserve existing implementation checks and records |
| 02 | P0/P1 | Development fixed/schedule and numerical diagnostics | Preserve existing records |
| 03 | P1 | Activation sensitivity and shared-prefix diagnostics | Preserve existing records; fixed 1000 remains the current main rule |
| **04** | P1 | Existing complete online experiment and component analysis | **Leave completed configurations, data and figures unchanged** |
| **05** | P1 | Independently coadapted online solve-policy comparison | New proposed 5000-problem development experiment, then a locked final scope |
| **06** | P1 | Frozen matched-hierarchy policy evaluation | Former logical Module 05 / physical `05_policy`; existing archives stay valid historical records |
| **07** | P1 | Theory integration and deterministic proof checks | Former `06_theory`; work in parallel with experiments |
| 08 | P2 | Stronger conventional numerical baselines and problem breadth | Former `07_baselines`; conditional on the paper's intended scope |
| 09 | P2 | Retrained observation/cadence ablations | Former `08_feedback`; needed only for stronger feedback claims |
| 10 | P3 | Forgetting, solve-policy bandits, SquareCB and other new learners | Former `09_algorithms`; not prerequisites and not part of Module 05 |
| 11 | P1 | Final tables, figures, manuscript and reproduction checks | Former `10_paper` |

P1 identifies importance to the intended claim, not an instruction to delay all paper writing until every extension is complete. No new dynamic activation, failure-target, cap, feature or learning-algorithm redesign is included in this revision.

## 3. Module 05: independent setup adaptation for each solve policy

### 3.1 Question and recommended arms

**Question:** given the same sequence of systems and the same number of online problem opportunities, does the existing Joint learner recover the extra cost of learning its solve policy relative to a prescribed periodic policy, when each uses its own setup learning trajectory?

| Arm | Setup learning | Primary solve policy | First use |
|---|---|---|---|
| `setup_w1` | Own fresh LinUCB | Constant weight one | Problem 1 |
| `setup_fixed` | Own fresh LinUCB | One development-selected constant `w_dev` | Problem 1 |
| `setup_periodic` | Own fresh LinUCB | `(2.85, 1.10)` periodically, high first | Problem 1 |
| `setup_rl` | Own fresh LinUCB and LSTDQ | Weight one for problems 1–1000; existing online LSTDQ for 1001–5000 | RL starts at problem 1001 |

The central comparison is `setup_periodic` versus `setup_rl`. `setup_w1` is a useful control. `setup_fixed` is optional until a development coefficient and its selection rule are frozen; it need not block the first periodic comparison.

An unchanged nonlearning Default may additionally be executed as a contemporaneous reference. It is not a fifth setup learner. Existing Module 04 times are historical results, not contemporaneous timing observations in the new experiment.

**This does not introduce another solve-bandit.** Constants and schedules are prescribed. Every adaptive setup branch uses the existing LinUCB machinery, and only `setup_rl` learns a solve policy.

### 3.2 Common conditions and independent mutable state

All arms receive identical matrices/RHS within each replicate, the same setup action space, initial setup priors, feature definitions, hyperparameters, candidate-generation rules and admissible relaxation grid. Each branch owns separate mutable regression state, selected/elite history, candidate cursors and random-number state. Use corresponding initial setup seeds and candidate base pools; independent state does not require selecting deliberately different initial random draws.

Each branch updates only from the realized cost of its own solve-and-recovery procedure. Local candidate sets and hierarchy choices may diverge as histories diverge. That is the intended coadaptation, not an error to eliminate by forcing shared actions.

Preserve the current Module 04 numerical and learning protocol: 18/18/9 smoother profile, tolerance 1e-6, cap 50, existing LSTDQ feature/action definitions and exploration settings, and `rollback_unrecovered`. First-attempt failures recovered by the prescribed fallback remain learned; finally unrecovered episodes follow the existing rollback rule. Every physically incurred cost remains in the evaluation regardless of whether its learning update is retained.

Do not change the fallback's policy merely because the primary method is periodic. Reset the periodic phase at each primary solve and at any new primary attempt as specified by the existing attempt protocol. A chosen weight controls the same complete-cycle action as in the RL path; this is not replacing the smoother by a different internal polynomial implementation.

### 3.3 Activation and resource accounting

Known constants and schedules can be used immediately, so their main deployment comparison starts at problem 1. Artificially delaying them until problem 1001 would withhold an available advantage. The Joint learner's weight-one prefix and later exploration/computation are part of the method being evaluated.

Count setup, native solve/residual monitoring, all primary and recovery work, setup-bandit updates and selection, controller inference/update/lifecycle costs, and any additional recurring baseline dispatch cost consistently. Record native-only and broader enclosing wall costs separately. Do not add recovery a second time or sum overlapping suffix learning labels as if they were separately executed physical work. Distinguish the actual measured cost from a learning target that may have a narrower scope.

One-time initialization and candidate preparation retain the same primary-metric treatment as Module 04 and are recorded separately for deployment-cost interpretations. Fixed-weight development calibration is likewise recorded separately; do not charge it only to one frozen-policy baseline while ignoring other historical development, and do not imply a tuned constant was obtained without prior information.

### 3.4 Which fixed weight?

A global fixed weight is one coefficient across a specified input/hierarchy distribution. It is not an absolute optimum independent of hierarchy construction.

One allowed rule is

`w_dev = argmin_w sum_{i in development} C(x_i, H_default(x_i), w)`.

This yields a **development-selected constant on the Default hierarchy distribution**. It is not the only valid calibration method, nor a guarantee that the coefficient is best after the new LinUCB branch adapts its hierarchies.

Another allowed rule uses a predefined, balanced development panel of frozen setup-only and Joint hierarchies (optionally with Default). This can reduce dependence on one source without training a separate setup learner for every candidate weight. All candidates must be evaluated on the same panel and the aggregation rule specified in advance. Success/recovery are handled consistently; do not pick a fast failed solve as the best successful policy.

Freeze `w_dev` before the new 5000-problem stream. Do not call the approximate 1.6 observed on earlier Joint hierarchies the optimum on Default without checking that claim. Do not use the final Module 06 test-hindsight choice to set the new online baseline.

The stronger optimization `min_w E[C_5000(B_w,w)]`, where `B_w` is trained throughout under that weight, is a different problem. Estimating it would require separate online setup-training runs for candidate constants and independent development selection. This expensive nested search is not required to include a credible tuned-constant baseline.

### 3.5 Theory-prescribed periodic pair

Use `(2.85,1.10)` as the current unanchored grid-based prescription. It minimizes, up to permutation,

`eta(a,b) = max_{0<=lambda<=1} lambda[(1-a lambda)(1-b lambda)]^2`

over `W={1,1.05,...,3}`. The continuous weighted degree-two minimizer has coefficients `2 +/- 2/sqrt(5)`. Componentwise rounding gives `(2.90,1.10)`; it is not the discrete minimizer. The grid calculation evaluates analytic extrema for all 861 unordered pairs and does not use PDE timing outcomes.

This is a normalized SPD smoothing-surrogate result, not an optimum of finite-budget multilevel wall-clock cost and not an advection convergence theorem. High-first phase is separately prescribed; the symmetric polynomial objective selects an unordered pair. Keep the existing exact algebra/grid verification with the baseline definition. The earlier anchored `(2.6,1)` has a different feasible-class restriction and may remain a Module 06 numerical comparison, not another mandatory online arm.

### 3.6 No per-instance oracle in the online comparison

A complete 41-weight evaluation on 5000 supplied hierarchies already costs 205000 solve trials per replicate before timing repetitions and recovery. This can be computed with sufficient resources, but it is not a practical equal-budget online comparator.

There is a deeper distinction: if each online input selects its best weight after observing all alternatives, its feedback creates another setup-training history with extra counterfactual information. A scan on one branch's recorded hierarchies does not optimize all possible histories. Keep per-instance best-observed fixed weights in Module 06's smaller matched-hierarchy diagnostic; do not feed those oracle outcomes to the main setup learners.

### 3.7 Pilot, final scope and interpretation

Recommended first development experiment: one prespecified 60-cubed diffusion stream, 5000 problems, with `setup_periodic` and `setup_rl`; add `setup_w1` when affordable. Add `setup_fixed` once its development calibration is settled. Four arms require 20000 method–problem executions per replicate, plus recovery, initialization and development work. A nonlearning Default adds 5000. These are counts, not predicted durations.

First run small functional checks for phase reset, weight application, stopping, recovery, independent learning states and timing. They do not estimate full-stream payback. Preserve the full 5000-problem horizon for the development comparison; an early or favorable final window alone is insufficient. Final scope, replicates and any advection extension are to be frozen after this diagnostic, not silently expanded into all old grids. No execution has been initiated by this plan edit.

Primary outputs:

- Total recorded cost for every full 5000-problem branch and paired cumulative `C_periodic-C_RL`.
- Native/overhead/setup/solve/recovery components, with first-attempt and final failures and actual final residuals.
- First 1000, active 4000 and last 1000 as secondary windows, plus final snapshots.
- Per-training-replicate results and descriptive uncertainty; input-level observations and timing repeats are not independent trained agents.

Use serial native solves in randomized within-problem method order, fixed thread limits and recorded workload conditions. Do not assume operating-system scheduling pins a job to a named core. Keep measurement checks and source/configuration fingerprints; a numerical failure is distinct from a protocol or timing failure.

Do not predetermine the winner. If Joint has deficit `D` relative to periodic after problem 1000, its final balance is `-D + sum_{t=1001}^{5000}(C_periodic,t-C_RL,t)`. Learning from scratch creates a cost but does not prove that later gains cannot repay it. Conversely, late-window superiority does not itself establish payback. A periodic win is a valid result about this finite budget, not a reason to discard or modify the comparison.

An optional shared-prefix experiment can answer a separate question about switching policies from the same learned setup state at problem 1000. It is not required for this first pilot and must not replace the main from-problem-one deployment comparison.

## 4. Module 06: frozen matched-hierarchy and source interaction

This is the previously named Module 05. Preserve its completed data and filenames; future outputs should identify both the new logical module number and the source experiment.

At minimum, use the setup-only and Joint frozen selectors for cross-source evaluation. After Module 05, its periodic-adapted selector is a useful additional source, and the tuned-constant source is optional. A setup source is a row of the controlled comparison, not a universally neutral baseline.

For each fresh test input and source, generate its hierarchy/configuration once, then evaluate **every** compared solve policy under the same matrix, RHS, hierarchy, initial iterate, cap, tolerance and recovery. Policies include weight one, a development-selected fixed weight, appropriately labelled global/per-instance scan-selected fixed references, prescribed periodic relaxation, and frozen LSTDQ. No solve outcome may update the source selector. Fixed references must be recalculated for each source; the old best weight on Joint-selected hierarchies need not remain best elsewhere.

Reset all mutable solve state. If rebuilding instead of restoring, verify structural/numerical consistency; a hash of only setup parameters is not a hash of the native hierarchy arrays. Randomize policy order and use prespecified repeated timings. Controller parameters/statistics remain frozen and random exploration is disabled as specified. A frozen controller may still change its action using residual or measured-cycle-time inputs.

Within a source, compare solve-plus-controller continuation cost, retaining policy-induced recovery setup. Common initial setup can be displayed separately. When comparing total cost across hierarchy sources, include their different construction costs.

The fixed scan yields a global fixed reference and a per-input fixed reference from the same cost matrix. These are best-observed grid diagnostics, not exact continuous optima, statistically exact expected-cost minima or deployable online learners. Independent retiming of selected choices is useful but does not retime every unselected alternative. Never compare methods on different success subsets without an explicit success-conditioned interpretation.

Report source-specific contrasts `Delta_g = C(H_g,periodic)-C(H_g,RL)` and their differences. If the relative policy ranking changes by source, that shows compatibility/transfer limitations; it does not invalidate the conditional row comparison. Different solve policies evaluated only on their own different hierarchies do not isolate direct control effects—that is the complete-system experiment in Module 05.

No new solve-bandit, unrestricted dynamic oracle, or proof that feedback is necessary is required. A retrained observation ablation remains conditional on asserting a feedback-specific contribution.

## 5. Module 07: theory integration, not another large PDE run

The [pinned detailed mathematical review](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md#2-数学核验与建议采用的证明) remains the source for the full T1–T6 discussion. Preserve its distinctions while integrating:

1. General rank-one spectral-moment analysis: fixed-cycle norms and mixed-cycle overlaps are different quantities. Keep admissibility, `q_*>0`, and norm assumptions explicit.
2. Three-dimensional Galerkin example: the cancelling pair `(1,2)` involves coarse correction; seven fixed cycles versus two uses a common initial Euclidean residual and equal-cost assumptions. Align the mathematical stopping priority with the implementation, or take a cap strictly greater than seven.
3. Separate two-dimensional nine-versus-two residual-greedy example: it addresses the difference between immediate progress and completion cost, not learning convergence.
4. Complete retained-episode LSTDQ trace-energy identity: prove episode-boundary well-posedness, not all-prefix invertibility, a uniform condition-number bound, statistical Q accuracy or calibrated optimism. Effective-span issues remain relevant.
5. Conditional setup/policy/joint comparisons and payback: preserve calibrated/fixed-downstream hypotheses; distinguish a modeled expected balance, realized finite-run crossings and future guarantees. Do not transfer the benchmark regret theorem to the decaying-alpha adaptive implementation.
6. Polynomial schedule note: distinguish anchored and unanchored weighted minimax problems, the admissible action grid, exact continuous coefficients and discrete selection. The production full-cycle application is a motivated baseline transfer, not universal multilevel runtime optimality.

Keep the constant-weight barriers and preset schedules consistent: an informed open-loop schedule can attain the toy mechanism. A successful learned policy need not demonstrate that RL is indispensable. Module 06's frozen evaluation is not a theorem or measurement of its historical learning payback; Module 04/05 supply the online cost evidence.

Do not start new forgetting, dynamic-start or failure-penalty algorithms during this integration. Relevant prior literature belongs where it supports the exact statement, not as an automatic demand for another implemented method.

## 6. Paper assembly and finite-scope completion

Begin paper editing now, while the narrowly scoped Module 05 diagnostic is prepared. The September 30 submission target does not support an open-ended search over caps, policies, grids and activation rules. Fix a feasible pilot and final scope rather than promise unmeasured runtimes.

Organize experimental evidence as: existing full-method Module 04; new independently coadapted alternatives in Module 05; source-controlled policy comparisons in Module 06. Keep all original outputs separate and numerically traceable. A familiar module number is not an excuse to overwrite a completed dataset.

Before finalizing:

- Confirm method definitions, setup/solve targets, failure handling, feature dimensions and timer scopes agree between paper and evaluated code.
- Keep actual failed-attempt/recovery costs in the primary tables and report unresolved outcomes; do not call incomplete advection coverage all-success completion time.
- Distinguish native work, controller-inclusive online time, broader method wall time and one-time preparation.
- Use the appropriate comparator denominator, per-replicate aggregation and uncertainty description.
- State each hierarchy source and each constant/schedule selection rule; preserve development/calibration and final evaluation roles.
- Treat source/configuration hashes, original paths and model identifiers as reproducibility metadata, not an additional algorithmic task.
- Integrate core proofs compactly and verify the final page limit, citations, figures and exact numerical claims against the submission instructions.

## 7. Existing implementation and historical references

- [Experiment index and current module mapping](../../experiments/paper_final/README.md)
- [Module 04 protocol and historical results entry](../../experiments/paper_final/04_online/README.md)
- [Original complete September 15–17 plan, unchanged](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md)
- [General online runner](../../experiments/joint/solve_control/run_paper_final.py)
- [Online component analysis](../../experiments/joint/solve_control/analyze_paper_final.py)
- [Shared-prefix execution and state restore](../../experiments/joint/solve_control/joint_4k_execution.py)
- [LSTDQ updates](../../solve/controllers/recursive_lstdq/v1.py) and [v3 statistics](../../solve/controllers/recursive_lstdq/v3.py)
- [Episode and recovery handling](../../solve/controllers/sarsa/online_td_lambda.py)
- [State encoder](../../solve/controllers/common/state_encoder.py)
- [Setup cost updates](../../setup/learners/linucb/setup_reselection.py)
- [Existing frozen cross-source diagnostic](../../experiments/diagnostics/solve_control/diagnose_context_activation_crossed.py)
- [Earlier theory review](theory_review_and_revision_plan_20260915.md)

This document records an updated design and numbering. Proposed Module 05 runners/configurations and new final results are not implied to exist merely because their design is specified here.
