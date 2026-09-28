# Paper final work

## Current plan — updated September 28, 2026

This is a documentation-only revision requested by the user. **Leave the completed Module 04 experiments, configurations, results and algorithm unchanged.** No experiment is launched and no learner or solver is changed by this revision.

### Current logical module numbering

| Module | Purpose | Implementation / artifact policy |
|---|---|---|
| 01–03 | Numerical checks, development diagnostics, activation sensitivity | Existing records and paths remain unchanged |
| **04** | Existing Default / LinUCB / LinUCB–LSTDQ full online comparison | Preserve unchanged; do not append new policies to, overwrite, or silently pool with its completed runs |
| **05** | **New 5000-problem comparison of independently coadapted setup learners with prescribed or learned solve policies** | Development pilot and final protocol are to be prepared separately; see design below |
| **06** | **Frozen matched-hierarchy policy evaluation and cross-source transfer** | This is the work previously called Module 05 / `05_policy`; existing filenames and archives remain valid historical identifiers |
| **07** | Theory integration and deterministic proof checks | Former logical Module 06 / `06_theory`; can proceed in parallel |
| 08 | Stronger numerical baselines and problem breadth | Former logical Module 07 / `07_baselines`; conditional on scope |
| 09 | Retrained feedback ablations | Former logical Module 08 / `08_feedback`; needed only for stronger feedback claims |
| 10 | Optional algorithm alternatives | Former logical Module 09 / `09_algorithms`; no new solve-bandit is required by Module 05 |
| 11 | Final paper assembly and reproducibility checks | Former logical Module 10 / `10_paper` |

These are **logical planning numbers**, not an instruction to rename existing scripts, imports, result directories, archived manifests or citation paths. Historical dated sections below retain their original numbering. The current mapping above takes precedence for new planning. The [original completion plan](../../docs/theory/paper_completion_plan_20260915.md) carries the same amendment.

### Module 05: complete online policy alternatives, not a frozen hierarchy test

**Question:** over 5000 problems, does online cycle-level LSTDQ justify its learning and computation costs against a strong prescribed periodic policy when each is allowed to develop its own compatible setup choices?

Recommended deployable arms:

| Arm | Mutable setup state | Primary solve policy |
|---|---|---|
| `setup_w1` | Own fresh LinUCB | Constant weight 1 from problem 1 |
| `setup_fixed` (optional until development coefficient is fixed) | Own fresh LinUCB | One development-selected constant `w_dev` from problem 1 |
| `setup_periodic` | Own fresh LinUCB | `(2.85, 1.10)` in high–low order, reset at each primary solve, from problem 1 |
| `setup_rl` | Own fresh LinUCB and LSTDQ | Preserve the current method: weight 1 on problems 1–1000, online LSTDQ on 1001–5000 |

The essential first pilot is `setup_periodic` versus `setup_rl`; `setup_w1` is a useful control. `setup_fixed` is a useful additional numerical baseline, not a prerequisite to checking the periodic comparison. An unchanged nonlearning Default may be executed in the same session as a timing/reference control; it is not another setup learner.

**No new solve-bandit is being proposed or implemented.** Every arm uses the existing setup LinUCB. Only `setup_rl` learns a solve controller. Other solve policies are prescribed before their online run.

Proposed initial scope: one prespecified development stream of 5000 diffusion 60-cubed problems, then a decision on replicated final runs and whether to include the advection family. A small smoke test checks wiring, phase/reset behavior, stopping/recovery and cost accounting; it does not decide 5000-problem payback. All pilot outcomes are informative, including a periodic win. The scope and final seed list must be fixed before the final suite; this document does not authorize a launch.

**Common protocol:** use the current Module 04 solver, features, setup action space, initial priors, exploration settings, cap 50, tolerance 1e-6, 18/18/9 profile, and rollback-unrecovered learning protocol. Keep all physical attempted/recovery work in the reported costs. All arms receive the same matrices/RHS and comparable initial setup states and candidate-generation rules, but own separate mutable regressions, histories, cursors and RNG states. Each learner observes only its own realized costs. Adaptive local candidate sets and chosen hierarchies may legitimately diverge. Do not force a shared hierarchy in this full-method comparison.

Use serial native solves with randomized within-problem method order and record the execution conditions. Do not use old Module 04 timestamps as contemporaneous observations for the new comparison. The new RL arm starts fresh; it is not a pretrained Module 04 checkpoint. Prescribed constants and schedules start on problem 1 because they need no solve-policy training. Preserve the existing recovery policy rather than automatically applying the primary periodic schedule to fallback.

**Fixed-weight calibration:** "global fixed" means one coefficient across a specified evaluation distribution, not a coefficient universally optimal for every hierarchy. Default-only development tuning is allowed but must be named as such; it is not the only valid source and need not be optimal after setup coadaptation. A balanced, predefined development panel of frozen setup-only and Joint hierarchies is an alternative without training a new setup bandit for each candidate weight. Select `w_dev` from development data before the online run and label it development-selected, not a proven optimum of the coadapted online system. Do not borrow a test-hindsight optimum from Module 06. Record calibration/search cost separately from the online curve. An actual end-to-end optimal constant over coadapted learners would require comparing independently trained setup learners for the candidate constants; that expensive search is not required for a credible baseline.

**Periodic coefficient rationale:** `(2.85,1.10)` is the discrete minimizer, up to permutation, of `max_{0<=lambda<=1} lambda[(1-a lambda)(1-b lambda)]^2` over the 41-point `[1,3]` grid. It is not the componentwise rounding `(2.90,1.10)` of the continuous fourth-kind pair, and it is not a theorem about optimal full multilevel runtime. The coefficient rule is fixed before the run; high-first phase is a separate prescribed convention. The SPD rationale is not an advection convergence guarantee.

**No online per-instance oracle arm.** A 41-weight scan on 5000 supplied hierarchies already needs 205000 complete solve trials per replicate before repetition/recovery. More importantly, using those counterfactual outcomes during setup training defines a different, information-rich procedure. Keep the existing per-case fixed-weight scan and its timing/selection qualifications in Module 06 on a smaller fixed test set.

**Primary outputs:** total recorded online cost over all 5000 problems; cumulative `C_periodic-C_RL`; all component totals, recovery costs, first-attempt/final failures and final residuals; first 1000, post-activation 4000 and last 1000 as secondary windows; final setup/controller snapshots. Report native-only and broader wall timers separately. Do not add recovery or overlapping suffix labels twice. The winner is not known in advance: learning from scratch imposes a cost but does not prove the periodic method must win. A better late policy is insufficient unless its gains repay the earlier deficit within the stated horizon.

### Module 06: crossed frozen evaluation, formerly Module 05

Preserve the completed frozen-policy datasets. New tests can use setup snapshots from `setup_w1`, `setup_periodic`, `setup_rl`, and optionally `setup_fixed` after Module 05, with the original Module 04 sources retained where useful.

On each source's hierarchy, evaluate **every** compared solve policy on the same input, hierarchy/configuration and initial state. Include weight 1, development-selected fixed weight, appropriately labelled scan-selected fixed references, the prescribed periodic policy, and frozen RL. Recompute fixed references for each hierarchy source; do not assume a weight selected on one source is optimal on another. Common primary setup cancels in within-source contrasts, but policy-induced recovery setup remains charged. Compare full setup-plus-solve cost when comparing across hierarchy sources.

A hierarchy source trained with weight 1 is not universally neutral, and a Joint-trained source is not automatically an invalid test environment. Each row measures a conditional policy effect. Comparing those rows reveals compatibility and transfer; different policies on different rows are not a direct controller ablation. Independent coadapted Module 05 arms answer the complementary complete-method question.

Freeze all evaluation models and source selection rules; no test outcome changes future source choices. A frozen feedback controller can still choose different cycle actions. Save all policies and source choices before evaluation, and use repeated paired timings without training on repetitions. Per-case fixed oracles remain labelled finite-grid diagnostics, not practical online algorithms or unrestricted optimal control.

Theory integration proceeds in Module 07; do not delay paper editing for a new learner or a proof that RL is indispensable.

---

## Historical progress note — September 17

The dated status and old physical stage names below are retained as history, not as the current completion state or current logical numbering.

Stages **01, 02 and both 03 development runs are complete**. The second 03 seed tested only starts after 250/500/750. Stage 04's September 17 diffusion replicate is complete at 40³, 60³ and 80³. The user-authorized [September 18 suite](04_online/20260918/README.md) uses the earlier diagnosis seed across both families, adds controller lifecycle timing and per-method wall timing, and sets the advection cap to 100. Its source, protocol and outputs are kept separate from the September 17 results. The most useful reading order is:

| Order | Read | What it answers |
|---|---|---|
| 1 | [03 activation report](../../results/paper_final/03_activation/REPORT.md) | Index of the completed s1 run and the fresh-seed 250/500/750 follow-up; each run links its own report and figures |
| 1a | [03 timing and seed comparison](../../results/paper_final/03_activation/advection_80_s2/analysis/REPORT.md) | Branch/order audit, frozen native replay, why after 750 changed, and links to the subsequent timing/stopping fixes |
| 2 | [02 diagnostics report](../../results/paper_final/02_diagnostics/REPORT.md) | How frozen RL compares with tuned fixed weights and short schedules on matched setups; includes all three diagnostic figures |
| 3 | [01 numerical fixes](01_numerics/README.md) | Inverse recovery, stopping semantics, timing coverage and their verification |
| 4 | [Paper completion plan](../../docs/theory/paper_completion_plan_20260915.md) | Remaining priorities; read §2 for the proposed proofs, §3 for historical status and §5 for the original experiment design |
| 5 | [Detailed theory review](../../docs/theory/theory_review_and_revision_plan_20260915.md) | Two-grid calculations, observation aliasing, conditional policy quality, candidate-set accounting and payback; the newer completion plan adds the 3D example and episode coercivity |
| 6 | [04 formal protocol](04_online/README.md) | The original Default / LinUCB / LinUCB–LSTDQ comparison, contexts, timing, seeds and reporting |

### Results in brief — historical development evidence

- **03:** all five RL activation paths reduced recorded cumulative cost versus
  setup-only by 14.09%–22.53% on this development seed. Starting after 1000 gave
  20.49%. Starting after 750 was best over the full stream; after 1500 was best
  in the last 1000 inputs. Each path still had 28–30 unrecovered inputs, so these
  are recorded costs including failed attempts, not all-success completion
  times. This compares complete joint learning paths, whose setups can diverge.
- **03 follow-up:** after 750 was best among 250/500/750 for both the full
  stream and the last 1000 inputs. Its cross-run cost increased 8.69%; the
  last-window gap is primarily more cycles and recovery, with mean native
  cycle time only 0.30% higher. The run has 44–45 unrecovered inputs per path.
- **Measurement:** wiring/prefix/order audits passed, but the historical RL timer
  omitted external residual monitoring, and full-solve versus RL success flags
  differed in the recorded runs when tolerance was reached on cycle 50.
  The [stopping correction](../../results/paper_final/01_numerics/stopping_alignment/REPORT.md)
  now follows BoomerAMG's native status, with 60 tests and two historical
  boundary cases verified. The [timing correction](../../results/paper_final/01_numerics/timing/REPORT.md)
  now includes initial and per-step residual monitoring, with 68 relevant tests
  passed. Existing development records remain unchanged.
- **02:** on the transferred diffusion-pilot setup for advection, frozen RL-LCB
  cost 6.87% less than the selected fixed weight and 3.50% more than the selected
  periodic schedule. Reference-setup fixed/schedule baselines were also faster
  than RL. Preserve these baselines in the formal policy evaluation; the current
  screen does not establish a general feedback advantage over schedules.
- **Encoding:** on the six held-out advection inputs, the previous-cycle time
  feature saturated after the first decision. This identifies a limitation of
  that input feature, without establishing that it caused the performance gap.
- **Numerics:** the inverse recovery fix and completed-run audits passed.
  Algorithmic learning/convergence guarantees remain separate mathematical
  questions.

### Method names

| Meaning | Display name |
|---|---|
| Setup bandit with the diffusion context | **LinUCB (4D)** |
| Setup bandit with the diffusion–advection context | **LinUCB (7D)** |
| Joint setup and solve method | **LinUCB–LSTDQ**, with the context dimension when needed |

Dimensions include the intercept. The implementation uses LinUCB v4 and
recursive LSTDQ v3; context dimension is a setting of the method. In a plot
where all paths share one context, show it once in the settings and use short
curve labels such as `Setup only` and `RL after 1,000`. The feature definition
belongs in the protocol, rather than in every method name.

### Original next-step sequence — superseded by the current plan above

1. Freeze the formal protocol using the corrected timing and stopping code,
   including independent input streams and the
   treatment/reporting of unrecovered failures. Development reuse of prepared
   formal inputs must be resolved or explicitly disclosed before launch.
2. Run **04** for the complete online comparison; reuse its logs for component
   analysis. Then use the final checkpoints for **05**, the independent matched
   setup evaluation against fixed weights and schedules.
3. Integrate the theory in **06**: the general moment identity, the 3D
   coarse-correction example, the 2D myopic-residual example, episode-boundary
   LSTDQ coercivity, and the conditional comparison/accounting results.
4. Keep algorithm alternatives and dynamic deployment ideas conditional on
   a separate decision; they are not prerequisites automatically added by 02.

### Reference material — read only for the relevant detail

- Protocol details: [02](02_diagnostics/README.md) and [03](03_activation/README.md).
- Figure indexes: [02](../../results/paper_final/02_diagnostics/figures/README.md)
  and [03](../../results/paper_final/03_activation/advection_80_s1/figures/README.md).
- Full per-window numbers: [03 screening tables](../../results/paper_final/03_activation/advection_80_s1/screen_report.md)
  and [5000-input summary](../../results/paper_final/03_activation/advection_80_s1/summary_5000.md).
- Checkpoint provenance: [historical compatibility audit](../../results/paper_final/02_diagnostics/checkpoint_audit/REPORT.md).
  Its then-pending request for a current advection checkpoint has now been
  satisfied by 03. Archived checkpoints are not the current advection model.
- Earlier mechanism work: [context/activation diagnosis](../../docs/theory/context_activation_mechanism_diagnosis_20260913.md).
- Optional research notes: [cost-based deployment](../../docs/theory/cost_based_rl_activation_framework.md),
  [reliability-trigger theory](../../docs/theory/dynamic_rl_start_and_theorem_framework.md),
  and [activation/performance limits](../../docs/theory/activation_and_amg_performance_limits.md).
  These are background directions, not additional required experiments. Some
  archived context labels count only coefficients; current display dimensions
  consistently include the intercept.

The completed `REPORT.md` files above give the historical conclusions at their
dates. `STATUS.md`, `FOLLOWUP.md`, preflight reports and captured source READMEs
are execution history or reproduction material.

## Historical stage directory

The paths below are retained for existing code, links and artifacts; use the
current logical module numbers above for new planning. No path has been renamed.

| Original order | Directory | Purpose | Historical state |
|---|---|---|---|
| 01 | [numerics](01_numerics/README.md) | Inverse recovery, checkpoints, rollback, stopping and timing | Original 65 tests; subsequent stopping/timing checks passed |
| 02 | [diagnostics](02_diagnostics/README.md) | Matched setup fixed/schedule checks; harder advection functional checks | Complete: 1176 diffusion + 336 advection comparisons; 58 abandoned training rows retained |
| 03 | [activation](03_activation/README.md) | Advection activation-time diagnostics | s1 and s2 complete: six and three paths × 5000; original full s2/s3 suite remains unexecuted |
| 04 | [online](04_online/README.md) | Main three-method comparison; reuse logs for cost breakdown | Diffusion s1 launched September 17, 04:53 EDT; 40³ → 60³ → 80³ |
| 05 | `05_policy` | Final frozen-policy evaluation on independent inputs | Now logical Module 06 |
| 06 | `06_theory` | Integrate mathematical results and deterministic proof checks | Now logical Module 07 |
| 07 | `07_baselines` | Stronger native/polynomial baselines and problem breadth | Now logical Module 08 |
| 08 | `08_feedback` | Matched feedback ablations, if needed for the claims | Now logical Module 09 |
| 09 | `09_algorithms` | Forgetting / policy-bandit alternatives | Now logical Module 10; conditional |
| 10 | `10_paper` | Regenerate final figures/tables and reproduction checks | Now logical Module 11 |

The general joint runner and native helpers remain in their existing modules.
These entry points reuse them:

```bash
python -m unittest experiments.paper_final.test_01_numerics
python -m experiments.paper_final.run_02_diagnostics       # inspect protocol only
python -m experiments.paper_final.run_02_diagnostics --run
python -m experiments.paper_final.run_03_activation        # validation only
python -m experiments.paper_final.run_04_online --validate-only
python -m experiments.paper_final.analyze_04_online
```

Use the project's `rl` Python environment. Stage 03 and the remaining stage 02
advection diagnostic completed on September 16. Existing outputs retain their
original protocol and numbering. The current planning amendment does not launch
or authorize another run and does not alter the completed Module 04 data.

See also [the paper completion plan](../../docs/theory/paper_completion_plan_20260915.md).
