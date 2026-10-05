# Historical paper-final progress — September 29, 2026

This preserves the previous stage index and progress narrative. It is an execution-history record; use [the current reproduction index](../reproduction.md) for accepted experiments and required artifacts. Relative links have been adjusted for this archive location.

# Paper final work

**Current manuscript scope, September 29:** The paper retains Module 04 and
the corrected six-checkpoint [matched-hierarchy Run 05](../../experiments/paper_final/05_policy/RUN05.md).
The separate experiment based on the new Module 05 checkpoints has been
removed from the paper at the user's request. Its completed records below
are historical archives, not evidence to reinsert into the current manuscript.

**Module plan updated September 28, 2026:** Module 04 retains its accepted
protocol and results. New [Module 05](../../experiments/archive/paper_development/05_online_policies/README.md) prepares
solve-specific frozen checkpoints using one shared 1000-problem W1 prefix
and five independent 4000-problem continuations.
The former Module 05 frozen-policy study is now [Module 06](../../experiments/archive/paper_development/06_policy/README.md);
the remaining planned stages shift to 07–11. At the user's request the initial
four-branch run was discarded. The revised Module 05 includes both
Default-development-selected w=1.40 and historical learned-hierarchy
reference w=1.60. It completed all 21,000 method–problem executions in
54.81 minutes, with zero unrecovered failures and verified frozen checkpoints;
see its [training report](../../results/paper_final/05_online_policies/20260928_shared_prefix/training/TRAINING_REPORT.md).
Module 06 completed the five frozen methods on 100 fresh diffusion 60³ inputs,
three repetitions, plus the periodic/RL 2×2 cross: 2,100 trials in 5.00 minutes,
zero failures or recoveries, all freeze/data audits passed. For this one trained
checkpoint set, periodic had the lowest complete-method mean (119.988 ms),
followed by RL (124.671 ms) and fixed 1.60 (127.213 ms). See the
[fresh frozen-method report](../../results/paper_final/06_policy/20260928_frozen_five_methods/REPORT.md).
The original Run 04 figure system now renders this new data: eight updated
main figures and four timing-repeat supplements. The main heatmaps use W1,
Fixed 1.60, Periodic (2.85,1.10), and frozen RL. See the
[12-figure PDF](../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/all_figures.pdf)
and [figure pack](../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/figure_pack.zip).

Module 05 now also has [two training cost figures](../../results/paper_final/05_online_policies/20260928_shared_prefix/training/analysis/paper_figures/module05_runtime_comparisons.pdf)
using Module 04's stacked-bar layout, for all 5000 and the final 1000 problems.
The [cost comparison](../../results/paper_final/05_online_policies/20260928_shared_prefix/training/analysis/paper_figures/REPORT.md)
shows that Periodic already wins the late-training window despite RL's small
cumulative lead. Download the current modules together in the
[Module 05/06 archive](../../results/paper_final/releases/module05_module06_20260928.zip).

**Module 06 corrected matched-hierarchy Run 05 (historical Module 05 numbering)**
uses **(2.85,1.10)** as the only periodic baseline, removing (1,3) and (2.6,1).
All five roles share the same Joint-trained hierarchy within each of six
checkpoints and 100 inputs; three new repetitions produced 8460 distinct
successful solves, with no recovery. The unfinished attempt was discarded.
RL's native reduction against Periodic is 0.63% ± 0.55 percentage points;
including controller cost gives -3.65% ± 0.63. See the
[Run 05 report, current atlas and verified archive](../../experiments/paper_final/05_policy/RUN05.md).
The older [Run 04 completion record](../../experiments/archive/paper_development/05_policy/COMPLETED.md) and archives
remain historical. The prescribed pair follows the
[discrete weighted-minimax rule](../theory/period_two_weighted_minimax_20260928.md).

## Earlier overview — September 17

The following overview records that date's progress. Current module numbers
and the current checkpoint-preparation task are defined above and in the stage directory.

Stages **01, 02 and both 03 development runs are complete**. The second 03 seed tested only starts after 250/500/750. Stage 04's September 17 diffusion replicate is complete at 40³, 60³ and 80³. The user-authorized [September 18 suite](../../experiments/archive/paper_development/04_online/20260918/README.md) uses the earlier diagnosis seed across both families, adds controller lifecycle timing and per-method wall timing, and sets the advection cap to 100. Its source, protocol and outputs are kept separate from the September 17 results. The most useful reading order is:

| Order | Read | What it answers |
|---|---|---|
| 1 | [03 activation report](../../results/paper_final/03_activation/REPORT.md) | Index of the completed s1 run and the fresh-seed 250/500/750 follow-up; each run links its own report and figures |
| 1a | [03 timing and seed comparison](../../results/paper_final/03_activation/advection_80_s2/analysis/REPORT.md) | Branch/order audit, frozen native replay, why after 750 changed, and links to the subsequent timing/stopping fixes |
| 2 | [02 diagnostics report](../../results/paper_final/02_diagnostics/REPORT.md) | How frozen RL compares with tuned fixed weights and short schedules on matched setups; includes all three diagnostic figures |
| 3 | [01 numerical fixes](../../experiments/archive/paper_development/01_numerics/README.md) | Inverse recovery, stopping semantics, timing coverage and their verification |
| 4 | [Paper completion plan](../theory/paper_completion_plan_20260915.md) | Remaining priorities; read §2 for the proposed proofs, §3 for status and §5 for formal experiment design |
| 5 | [Detailed theory review](../theory/theory_review_and_revision_plan_20260915.md) | Two-grid calculations, observation aliasing, conditional policy quality, candidate-set accounting and payback; the newer completion plan adds the 3D example and episode coercivity |
| 6 | [04 formal protocol](../../experiments/archive/paper_development/04_online/README.md) | The planned Default / LinUCB / LinUCB–LSTDQ comparison, contexts, timing, seeds and reporting |

### Results in brief

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

### What comes next

1. Prepare **05** with Default-only development calibration and a common
   1000-problem W1 prefix followed by five 4000-problem continuations. Save
   each setup selector and the LSTDQ controller at problem 5000; training
   victory/payback is not an acceptance requirement.
2. Keep **04** as accepted. In **06**, evaluate each frozen setup–solve
   pairing on fresh inputs with full setup/controller/recovery cost. A small
   periodic/RL crossed-hierarchy comparison addresses compatibility.
3. Integrate the theory in **07**: the general moment identity, the 3D
   coarse-correction example, the 2D myopic-residual example, episode-boundary
   LSTDQ coercivity, the weighted-minimax prescription, and the conditional
   comparison/accounting results.
4. Keep algorithm alternatives and dynamic deployment ideas conditional on
   a separate decision; they are not prerequisites automatically added by 02.

### Reference material — read only for the relevant detail

- Protocol details: [02](../../experiments/archive/paper_development/02_diagnostics/README.md) and [03](../../experiments/archive/paper_development/03_activation/README.md).
- Figure indexes: [02](../../results/paper_final/02_diagnostics/figures/README.md)
  and [03](../../results/paper_final/03_activation/advection_80_s1/figures/README.md).
- Full per-window numbers: [03 screening tables](../../results/paper_final/03_activation/advection_80_s1/screen_report.md)
  and [5000-input summary](../../results/paper_final/03_activation/advection_80_s1/summary_5000.md).
- Checkpoint provenance: [historical compatibility audit](../../results/paper_final/02_diagnostics/checkpoint_audit/REPORT.md).
  Its then-pending request for a current advection checkpoint has now been
  satisfied by 03. Archived checkpoints are not the current advection model.
- Earlier mechanism work: [context/activation diagnosis](../theory/context_activation_mechanism_diagnosis_20260913.md).
- Optional research notes: [cost-based deployment](../theory/cost_based_rl_activation_framework.md),
  [reliability-trigger theory](../theory/dynamic_rl_start_and_theorem_framework.md),
  and [activation/performance limits](../theory/activation_and_amg_performance_limits.md).
  These are background directions, not additional required experiments. Some
  archived context labels count only coefficients; current display dimensions
  consistently include the intercept.

The completed `REPORT.md` files above give the current conclusions. `STATUS.md`,
`FOLLOWUP.md`, preflight reports and captured source READMEs are execution history
or reproduction material; they do not need to be read to understand the results.

## Stage directory

Use short names within each numbered stage. Code/protocols live here; outputs
live under `results/paper_final/<stage>/`. Older evidence retains its original
location and provenance.

| Order | Directory | Purpose | State |
|---|---|---|---|
| 01 | [numerics](../../experiments/archive/paper_development/01_numerics/README.md) | Inverse recovery, checkpoints, rollback, stopping and timing | Original 65 tests; subsequent stopping/timing checks passed |
| 02 | [diagnostics](../../experiments/archive/paper_development/02_diagnostics/README.md) | Matched setup fixed/schedule checks; harder advection functional checks | Complete: 1176 diffusion + 336 advection comparisons; 58 abandoned training rows retained |
| 03 | [activation](../../experiments/archive/paper_development/03_activation/README.md) | Advection activation-time diagnostics | s1 and s2 complete: six and three paths × 5000; original full s2/s3 suite remains unexecuted |
| 04 | [online](../../experiments/archive/paper_development/04_online/README.md) | Existing main three-method comparison and cost breakdown | Accepted scope and results retained |
| 05 | [checkpoint preparation](../../experiments/archive/paper_development/05_online_policies/README.md) | Shared 1000 W1 + five independent 4000 continuations; five setup selectors and LSTDQ checkpoint | Complete: 21,000 executions, zero unrecovered failures, frozen artifacts verified; original four-branch run discarded |
| 06 | [policy](../../experiments/archive/paper_development/06_policy/README.md) | Five frozen complete methods and periodic/RL 2×2 cross on 100 fresh inputs × three repetitions | Complete: 2,100 trials, zero failures/recoveries, audits passed; periodic lowest full cost on this checkpoint set; [historical accepted data](../../experiments/archive/paper_development/05_policy/COMPLETED.md) preserved |
| 07 | `07_theory` | Integrate mathematical results and deterministic proof checks | Paper editing stage |
| 08 | `08_baselines` | Stronger native/polynomial baselines and problem breadth | Planned |
| 09 | `09_feedback` | Matched feedback ablations, if needed for the claims | Conditional |
| 10 | `10_algorithms` | Forgetting / policy-bandit alternatives | Conditional, not part of the current run |
| 11 | `11_paper` | Regenerate final figures/tables and reproduction checks | Final assembly |

Historical `05_policy` directories, `run_05_policy*` entry points, and
`module05` archives now provide Module 06 evidence. Their original names and
captured manifests remain the reproduction identifiers. New Module 05 work
uses `05_online_policies` and does not overwrite those records.

The general joint runner and native helpers remain in their existing modules.
These entry points reuse them:

```bash
python -m unittest experiments.archive.paper_development.test_01_numerics
python -m experiments.archive.paper_development.run_02_diagnostics       # inspect protocol only
python -m experiments.archive.paper_development.run_02_diagnostics --run
python -m experiments.archive.paper_development.run_03_activation        # validation only
python -m experiments.paper_final.run_04_online --validate-only
python -m experiments.paper_final.analyze_04_online
python -m experiments.archive.paper_development.plot_05_policy          # completed logs only; no solver runs
```

Use the project's `rl` Python environment. Stage 03 and the remaining stage 02
advection diagnostic completed on September 16. The latest authorized stage 04
protocol is linked above. Existing 02 outputs are retained; its original entry point refuses
to overwrite them. Both completed stages remain development diagnostics.

See also [the paper completion plan](../theory/paper_completion_plan_20260915.md).
