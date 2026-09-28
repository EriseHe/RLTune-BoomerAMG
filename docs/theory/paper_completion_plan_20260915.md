# SISC paper completion plan: checkpoint preparation, frozen evaluation, and theory

**Purpose clarified September 28, 2026:** Module 05 is a checkpoint-training stage for Module 06. It is **not primarily a new standalone cumulative-cost or payback experiment**. Its purpose is to give the prescribed solve policies their own comparably trained setup selectors, instead of evaluating every alternative only on hierarchies learned in conjunction with RL.

This clarification supersedes the earlier September 28 recommendation to activate periodic/fixed policies from problem 1 and make their online training-time ranking the primary Module 05 result. That recommendation answered a different deployment question. The current user-selected protocol is a common 1000-problem weight-one setup prefix followed by 4000 problems of branch-specific adaptation, then freezing and evaluation.

**Module 04 remains unchanged.** No source code, completed configuration, result, checkpoint or historical path is modified by this documentation revision. No experiment is launched. The earlier full plan remains available in [revision 53f48f8](https://github.com/EriseHe/RLTune-BoomerAMG/blob/53f48f8ab16cc91f38aa7863c964147e132ca521/docs/theory/paper_completion_plan_20260915.md). The detailed original T1–T6 proof review and September 15–17 execution history remain in [revision 007bd195](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md).

## 1. Current division of work

| Module | Role | Primary deliverable |
|---|---|---|
| 01–03 | Numerical checks and earlier development/activation diagnostics | Preserve existing records |
| **04** | Existing complete online experiment and cumulative-cost evidence | Leave completed data and protocol unchanged |
| **05** | **Matched-budget setup/controller checkpoint preparation** | Four frozen setup selectors and the corresponding trained RL controller |
| **06** | **Fresh-input evaluation of the frozen trained methods, with optional crossed-policy diagnostics** | Post-training runtime, cycles, accuracy/recovery, and hierarchy–controller compatibility |
| **07** | Theory integration and deterministic proof checks | Consolidated numerical mechanisms, estimator identities, and qualified performance statements |
| 08 | Stronger conventional baselines/problem breadth | Conditional on manuscript scope |
| 09 | Retrained feedback/cadence ablations | Only for stronger mechanism claims |
| 10 | Optional new learning algorithms | Not required for Modules 05–06 |
| 11 | Paper assembly and reproducibility checks | Final manuscript, tables and archived protocol |

Former Module 05 / physical `05_policy` is now logical Module 06. Former `06_theory` is logical Module 07; later logical numbers shift accordingly. Existing scripts, imports, result directories and archive names are not renamed.

## 2. Module 05: matched training to prepare frozen setup policies

### 2.1 Why this stage exists

The final Joint setup selector was trained under weight one for the first 1000 systems, then under an evolving LSTDQ controller for the next 4000. Evaluating other solve policies only on that selector's hierarchies is a valid conditional substitution test, but does not give those alternatives their own trained setup–solve pairing.

Module 05 supplies that missing pairing. Fairness here means a common starting setup history and an equal number of subsequent problem opportunities for adaptation to each designated solve method. It does not mean that all models converge to an optimum, have identical retained update counts, or receive equal numbers of seconds/cycles. Those are outcomes, not matching constraints.

### 2.2 Training protocol

For each prescribed training replicate, execute **one actual LinUCB + weight-one prefix on problems 1–1000**. Clone the complete setup-learning state at the boundary: regression/inverse state, retained observations/counts, selected and elite history, candidate cursor/generator state, random-number state, and protocol counters. Four independently timed prefixes with the same seed are not a substitute for a genuinely shared state when measured times enter learning.

Retain the shared observations in every branch. Do not reset the model only for prescribed alternatives. The continuation is:

| Branch | Problems 1–1000 | Problems 1001–5000 | Final artifacts |
|---|---|---|---|
| W1 | Common setup learning under weight one | Continue setup learning under weight one | Frozen setup selector B_W1 |
| Fixed | Same cloned prefix | Setup learning under one prescribed development-selected w_dev | Frozen setup selector B_Fixed; recorded w_dev |
| Periodic | Same cloned prefix | Setup learning under (2.85,1.10), high first, reset per primary solve | Frozen setup selector B_Periodic; schedule specification |
| RL | Same cloned prefix | Setup learning plus existing online LSTDQ cycle decisions | Frozen setup selector B_RL and final solve controller pi_RL |

Fixed and Periodic start their designated solve policies at **problem 1001**, not problem 1. All four setup models have 5000 logical problem opportunities and share the same first 1000. Fixed, Periodic and RL each have 4000 continuation-specific adaptation problems. W1 is the no-switch control. Only the RL branch trains a solve learner; no solve-bandit or new learning algorithm is introduced.

One shared prefix and four continuations require **17000 physical method–problem executions per training replicate**, before recovery and initialization, while each logical branch has a 5000-problem history. This is a count, not a runtime estimate. Module 05 is not a requirement to rerun or modify Module 04. Use an existing prefix only when its complete state and compatibility are actually available; do not infer equality from seeds or reconstruct a full learner from theta alone.

Every branch receives the same continuation inputs but owns separate mutable state and observes only its own realized costs. Hierarchies and adaptive local candidate sets may diverge after problem 1000. This is the intended source of solve-compatible setup selectors.

Keep the current Module 04 numerical and learning protocol: cap 50, tolerance 1e-6, 18/18/9 profile, setup action space and features, LSTDQ action/features/exploration, and rollback-unrecovered convention. Preserve fallback behavior; changing the primary solve schedule does not change the fallback automatically. All actually executed costs, including failed attempts, remain logged even when learning updates are rolled back.

### 2.3 Selecting prescribed policies before training

Freeze w_dev using separate development inputs before training its setup branch. Tuning on Default hierarchies is permissible but not uniquely neutral and does not establish an optimum on later learned hierarchies. A prespecified mixed development panel is another permissible choice. Call it development-selected, not the global optimum of the coadapted system. Do not use Module 06 test hindsight to choose this coefficient.

Use (2.85,1.10) as the specified periodic pair. It minimizes, up to permutation, `max_{0<=lambda<=1} lambda[(1-a lambda)(1-b lambda)]^2` over `W={1,1.05,...,3}`. That discrete SPD smoothing-surrogate rationale does not imply optimal multilevel runtime or nonsymmetric-advection convergence. High-first phase is separately prescribed. Other already tested schedules can remain Module 06 diagnostics; do not add more training arms automatically.

A per-instance fixed-weight hindsight oracle is not a training branch. Scanning 41 weights on 5000 supplied hierarchies already requires 205000 solve trials before repeats and recovery; feeding their minima into setup training creates a different information-rich procedure. Retain the fixed-weight oracle as a smaller Module 06 diagnostic.

### 2.4 What to retain and how to judge completion

Save full trajectories, terminal outcomes, cost components, learning-state audits, final setup/controller checkpoints, and exact source/config/input identifiers. Run functional checks for clone independence, policy switching at problem 1001, schedule reset, recovery, and accounting before the prescribed training runs.

**The primary product is the final checkpoint set, not a winning training-time curve.** Periodic being faster during training does not disqualify the RL checkpoint or this design. Do not require an online victory or payback threshold before proceeding to Module 06. Training costs remain available for diagnostics and a transparent description of the budget; they need not become an additional standalone primary paper experiment.

Use the prescribed final training endpoint and retain every valid replicate. Test outcomes must not select replacement checkpoints. Final scope/seeds are fixed before training. Initial engineering checks and any development runs remain separate. No launch or final seed list is implied by this document.

## 3. Module 06: evaluate the trained frozen methods

### 3.1 Primary comparison: each solve method gets its own adapted setup selector

On each common fresh test input x, evaluate these four frozen pipelines:

- B_W1(x) followed by fixed weight one;
- B_Fixed(x) followed by fixed w_dev;
- B_Periodic(x) followed by prescribed (2.85,1.10);
- B_RL(x) followed by frozen pi_RL.

All setup regressions, statistics and inference conventions are frozen. Fix the same intended frozen-selector rule across models before evaluation; do not silently switch some models from LCB to mean-greedy. No test cost updates any setup model or solve controller. Prescribe candidate generation, source selection and model snapshots before evaluating outcomes; cache each input/source hierarchy choice for paired timing repetitions and crossed evaluations. Frozen RL can still respond to the evolving residual and measured cycle-time inputs, but it performs no parameter/statistic updates; random exploration is disabled according to the frozen-evaluation protocol.

The primary full-pipeline metric includes setup-selector inference, hierarchy construction, native solve/residual monitoring, solve-policy execution, and recovery. There is no online training-update cost during frozen evaluation, though necessary execution/bookkeeping is charged. Retain separate native and controller-inclusive breakdowns and successful outcomes/failures. Historical training expense is described in Module 05; do not claim Module 06 by itself establishes cumulative online payback. Module 04 remains the paper's existing evidence for that distinct claim.

This comparison removes the asymmetry of forcing every prescribed baseline onto Joint-selected hierarchies. It compares **post-training complete methods**, not relaxation policies on one fixed hierarchy. Different trained sources can legitimately select different hierarchies, and their construction costs cannot be omitted from this full-pipeline comparison.

The common weight-one prefix is a deliberately specified, matched training history. It does not guarantee that 4000 further problems suffice for every learner or that the resulting models are globally optimal. It also does not evaluate the fastest possible training protocol for every prescribed policy; no such claim is needed for this checkpoint comparison.

### 3.2 Optional crossed controls to explain compatibility

When attributing a difference specifically to solve control, test policies on the same hierarchy. The compact Periodic/RL crossed comparison is:

| Hierarchy source | Periodic solve | Frozen RL solve |
|---|---|---|
| B_Periodic | Own pairing | Transferred RL pairing |
| B_RL | Transferred periodic pairing | Own pairing |

Generate each source hierarchy once per input and restart the identical initial state for its policy comparisons. Common initial setup cancels in within-row solve contrasts, while policy-induced recovery setup remains charged. Add W1/fixed sources and fixed-weight scans only where they address the intended claim; a full four-by-four evaluation is not automatically required.

The two own-pairing cells answer post-training pipeline performance. The within-row differences answer conditional controller performance. A changed ranking between sources indicates compatibility, not that either conditional comparison is invalid. No hierarchy source is universally neutral.

### 3.3 Fixed-reference and timing details

Per-case/global fixed references must be recomputed on their specified source hierarchies. Best-observed values from a finite, noisy scan are not exact continuous optima. Do not transplant a weight selected on B_RL and label it optimal on B_Periodic. Keep development-selected weights distinct from test-hindsight diagnostics.

Use the same matrices/RHS, tolerance, cap and recovery convention; reset all mutable native solve state. If rebuilding, verify numerical/structural consistency rather than calling a hash of parameters a full hierarchy-array hash. Randomize serial policy execution and use prescribed timing repetitions without training on those repetitions. Report uncertainty at appropriate problem and independently trained checkpoint levels; repetitions are not additional independent training seeds. No fastest-repeat selection or silent dropping of failed cases.

No dynamic oracle, new solve-bandit, or proof of RL necessity is required. A retrained observation ablation remains conditional on a specific residual-feedback claim, not on completing the current evaluation.

## 4. Module 07: concise theory integration

Use the [original detailed mathematical review](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md) for complete T1–T6 arguments. Preserve:

1. Spectral-moment identities distinguishing fixed-cycle contraction from mixed-cycle interactions, with explicit admissibility and SPD/norm assumptions.
2. The three-dimensional exact-Galerkin construction and its seven-versus-two comparison, with the same initial residual and equal-cycle-cost qualifications; take the implementation cap strictly greater than seven if cap exhaustion has priority.
3. The separate two-dimensional residual-greedy counterexample, which concerns completion objectives, not learning convergence.
4. Complete retained-episode LSTDQ coercivity and recursive/batch identities, not all-prefix invertibility, accurate Q-values or calibrated confidence.
5. Conditional performance/payback accounting, not a proved regret theorem for the actual evolving joint learner.
6. Explicit polynomial-schedule optimization classes: anchored (2.6,1), continuous fourth-kind pair, and grid-constrained (2.85,1.10). Their smoothing objectives are not full multilevel runtime objectives.

Proceed with manuscript editing in parallel. Do not reopen cap, failure-target, dynamic-start, feature or learner design solely to make a training curve favorable.

## 5. Reproduction and paper checklist

- Module 04 is unchanged; Modules 05 and 06 have distinct training and evaluation roles.
- Clone the actual common setup prefix; all four branches inherit its records and complete state.
- Freeze endpoint models before accessing fresh Module 06 outcomes.
- Preserve all incurred training costs in logs, but make frozen test cost the new comparison's principal result.
- Include hierarchy construction when comparing own-pairing pipelines; use matched hierarchies for direct solve contrasts.
- Keep prescribed-policy selection, training budget and final evaluation inputs distinct.
- Match solver tolerances, caps, failure/recovery, features, cost scopes and model-selection conventions.
- Keep old files/archives and numbering traceable; no destructive renaming is needed.
- Report supported finite-domain results without asserting necessary RL, neutral hierarchies or globally optimal learned setups.

## 6. Existing entry points

- [Experiment index](../../experiments/paper_final/README.md)
- [Module 04 protocol](../../experiments/paper_final/04_online/README.md)
- [Shared-prefix/state restore machinery](../../experiments/joint/solve_control/joint_4k_execution.py)
- [Existing frozen cross-source diagnostic](../../experiments/diagnostics/solve_control/diagnose_context_activation_crossed.py)
- [Setup feedback](../../setup/learners/linucb/setup_reselection.py)
- [LSTDQ episode/recovery handling](../../solve/controllers/sarsa/online_td_lambda.py)
- [Original theory review](theory_review_and_revision_plan_20260915.md)

This is a documentation correction. New training/evaluation runners and completed results are not implied to exist merely because their intended protocol is specified.
