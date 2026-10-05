# Module 06: evaluate frozen complete methods

**Paper scope update, September 29:** The new-checkpoint own-pairing/crossed
study documented below is excluded from the manuscript at the user's request.
The retained frozen experiment is the six-checkpoint, Joint-trained
[historical Module 05 Run 05](../05_policy/RUN05.md), with the prescribed
(2.85,1.10) pair. The completed data below remain historical records.

Module 05 prepares comparable solve-specific checkpoints from one shared
1000-problem W1 prefix and five independent 4000-problem continuations.
Module 04 retains its existing online cumulative-performance evidence.

The [combined Module 05/06 archive](../../../results/paper_final/releases/module05_module06_20260928.zip)
contains both current modules' data, checkpoints and figures. The
[training-window comparison](../../../results/paper_final/05_online_policies/20260928_shared_prefix/training/analysis/paper_figures/REPORT.md)
reconciles RL's small cumulative training advantage with Periodic's lower
final-thousand and frozen-test costs. The
[controller-transfer diagnostic](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/controller_transfer_diagnostic/REPORT.md)
separately explains the crossed-hierarchy action shift.

## Primary fresh-input comparison

| Frozen setup selector | Solve policy |
|---|---|
| W1-adapted | Fixed w=1 |
| Fixed-1.40-adapted | Default-development-selected fixed w_dev=1.40 |
| Fixed-1.60-adapted | Historical setup-only/Joint hierarchy fixed choice w=1.60 |
| Periodic-adapted | (2.85,1.10), high first, reset per primary attempt |
| RL-adapted | Frozen LSTDQ, no exploration or parameter updates |

Use common fresh matrices/RHSs and initial iterates with the accepted
smoother, tolerance, cap and recovery protocol. Test observations update
neither learner. Freeze the candidate-selection/evaluation rule before test
execution and audit unchanged setup/controller learning statistics.

Different setup selectors may choose different hierarchies. The primary
cost therefore includes setup selection and construction, solve execution,
controller computation, and recovery. This evaluates the complete frozen
methods after prescribed training. Equal training opportunities do not
assert equally optimal final models.

## Optional mechanism comparison

| Hierarchy source | Periodic solve | Frozen RL solve |
|---|---|---|
| Periodic-adapted setup selector | Own pairing | Crossed pairing |
| RL-adapted setup selector | Crossed pairing | Own pairing |

Within a row, match the hierarchy and solve state. These comparisons separate
solve behavior on that source from compatibility of the trained pair. An
exhaustive five-by-five study is not required. Per-instance fixed-weight
oracles remain labelled diagnostics on smaller matched-hierarchy samples.

## Completed execution: September 28, 2026

All **2,100 trials completed in 4.997 minutes**, finishing at 09:55 New York
time. Every solve reached tolerance without fallback. Seven contract tests,
28 separate native preflight trials, exact frozen-state checks and an
independent reconstruction of all reported means/intervals passed. Module
04, the Module 05 checkpoints and the solver/learner implementation remain
unchanged.

Mean full-method costs were W1 **160.118 ms**, fixed 1.40 **132.523 ms**,
fixed 1.60 **127.213 ms**, periodic **119.988 ms**, and frozen LSTDQ
**124.671 ms**. Periodic was fastest for this checkpoint set. Relative to
periodic, RL cost 3.903% more (paired input-bootstrap 95% interval
[2.844%, 5.004%]); RL cost 1.998% less than fixed 1.60
([0.970%, 2.996%]). These are fresh frozen-method results, not a training
payback claim or evidence over multiple retraining seeds.

The matched-hierarchy cross also favored periodic after controller costs:
61.998 versus 83.387 ms on periodic-adapted hierarchies, and 60.880 versus
62.731 ms on RL-adapted hierarchies (periodic versus RL; common initial
selection/setup excluded). On RL-adapted hierarchies the native-only RL
advantage was just 0.321%, with interval [-1.465%, 1.965%]; charging controller
work changed the comparison to 3.041% higher RL cost. The larger gap on
periodic-adapted hierarchies supports a setup/controller compatibility effect.

- [Full results and paired intervals](../../../results/paper_final/06_policy/20260928_frozen_five_methods/REPORT.md)
- [Independent raw-data verification](../../../results/paper_final/06_policy/20260928_frozen_five_methods/independent_verification.json)
- [Completion record](../../../results/paper_final/06_policy/20260928_frozen_five_methods/complete.json)

The user authorized this evaluation after Module 05 completed. The locked
protocol uses **100 fresh diffusion 60³ inputs and three timing repetitions**:
1,500 trials for the five complete methods, plus 600 trials for the two
crossed pairings. This evaluates one trained checkpoint set. Seven contract
tests cover frozen selection/state, paired repetitions, policy dispatch,
recovery accounting, and the successful-RL reporting schema. The native
preflight uses two separate inputs and is excluded from the 2,100 trials.

The fresh inputs do not overlap the 5,312 unique training, calibration,
functional-check and historical-evaluation inputs checked in the input audit.
The complete training decision histories are restored. A new common candidate
schedule and fixed per-input tie seeds make each source's setup choice
independent of timing order. Every trial measures selection again, cancels
the pending observation, and restores the frozen cursor/RNG. Learning
statistics and decision history are audited unchanged; deterministic caches
may warm. The original Module 05 files are retained and hash-checked.

The existing frozen-evaluation recovery rule is explicit: one selected-hierarchy
primary attempt, followed if necessary by Default-setup/W1 recovery. There is
no outcome-dependent setup reselection. Use tol=1e-6, cap=50 and the accepted
l1-Jacobi/direct-coarse smoother. Execute serially, randomizing case order
each repetition and method order within each case.

The primary metric includes setup selection, native setup/solve, controller
work and all recovery. Within-source crossed comparisons exclude only common
initial selection/setup. Average all three repetitions per input, then weight
the 100 inputs equally. Paired bootstrap intervals are conditional on this
checkpoint set, not uncertainty over retraining seeds.

- [Locked protocol](../../../results/paper_final/06_policy/20260928_frozen_five_methods/protocol.json)
- [Execution status](../../../results/paper_final/06_policy/20260928_frozen_five_methods/status.json)
- [Runner](../run_06_frozen_methods.py): prepare, run, analyze.
- [Contract tests](../test_06_frozen_methods.py)

A preflight-only reporting repair normalized missing zero-valued recovery
fields on successful RL solves. Its provenance is recorded; no test trials
had run and no protocol, input, setup choice, checkpoint or solver was changed.

Frozen means model parameters, learning state, history and RNG remain unchanged;
it does not prescribe an identical RL action sequence. The accepted encoder
includes the last measured cycle time. RL action traces varied across timings
on 39/100 own-source and 34/100 periodic-source inputs, while prescribed-policy
traces were identical. All repetitions, including this observation-dependent
variation, remain in the averages.

## Current-data figure set

At the user's request, the original Run 04 plotting system is reused for the
current Module 06 data. All eight main figure roles are updated, with four
supplementary action/residual heatmaps for timing repetitions 2 and 3. The
main heatmaps show exactly **W1, Fixed 1.60, Periodic (2.85,1.10), and frozen RL**.
The fixed-1.40 results remain in the experiment data and report but are outside
the requested four-method figure set.

- [All 12 figures PDF](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/all_figures.pdf)
- [Eight main figures PDF](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/main_figure_atlas.pdf)
- [Gallery](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/index.html)
- [Complete PNG/PDF/SVG figure pack](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/figure_pack.zip)
- [Captions](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/CAPTIONS.md) and [verification](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/paper_figures/VERIFICATION.json)

The eight main roles are policy savings, action heatmaps, residual heatmaps,
the RL trajectory atlas, periodic/RL comparison, direct RL advantage, the
period-two objective, and timing repeatability. Original style, exporter,
trajectory rendering, common baseline-difficulty ordering and gallery helpers
are reused unchanged through [the current-data adapter](../plot_06_policy.py).

The original six-training-seed atlas is adapted to the actual available
dimensions: two hierarchy sources × three timing repetitions of one frozen
controller. No timing repetition is called a training seed. The objective
figure uses the current free period-two criterion, not the former anchored
(2.6,1) criterion. All own-pairing figures identify distinct adapted setups;
the crossed figure matches the hierarchy within each row.

W1 batch logs omit per-cycle residuals. A separate
[trace collector](../collect_06_w1_traces.py) performed 100 same-input,
same-setup constant-W1 diagnostic solves. All cycle counts and final residuals
match the 300 original W1 timing trials (rtol=1e-10, atol=1e-14). Its deterministic
residual traces supplement the W1 panels, with disclosure in every trajectory
caption. Other traces come directly from the recorded repetition. Original
formal timings, raw records, checkpoints and learner states are unchanged.
The [replay audit](../../../results/paper_final/06_policy/20260928_frozen_five_methods/analysis/w1_trace_audit.json)
and figure verification retain this distinction. All 1,800 represented
method–repetition–input trajectory records and all 12 previews were checked.

## Corrected six-checkpoint matched-hierarchy comparison — September 29

[Historical Module 05 Run 05](../05_policy/RUN05.md) is now complete with the
prescribed (2.85,1.10) pair and five policy roles. It retains the original six
Joint-trained checkpoint pairs, hierarchy tuples, fixed-grid choices and
100 inputs, then retimes every role in three repetitions. The discarded
unfinished attempt contributes no data. All 8460 distinct solves succeeded
without recovery. Native RL reductions are 11.72% against stream-wide fixed,
9.58% against per-instance fixed, and 0.63% against Periodic; with controller
cost, RL costs 3.65% more than Periodic. This complements the separate fresh
own-pairing and crossed study above, without retraining either experiment.

## Historical evidence

- [Historical protocol and run index](../05_policy/README.md).
- [Accepted Run 04 completion record](../05_policy/COMPLETED.md): six frozen
  checkpoints, 100 common diffusion 60³ inputs, three timing repetitions,
  with the historical anchored schedule (2.6,1).
- [Paired (2.85,1.10) versus (2.90,1.10) comparison](../../../results/paper_final/05_policy/20260928_pair285_vs290_joint_6seeds_100cases/REPORT.md):
  4800 solves, four repetitions and balanced serial order.
- [Current prescribed-pair mathematical basis](../../../docs/theory/period_two_weighted_minimax_20260928.md).

Historical `05_policy` directories, `run_05_policy*` scripts, run numbers,
captured source/manifests and `module05` release names remain the reproduction
identifiers. They are linked here as Module 06 evidence. The new five-method
evaluation above uses the unanchored pair on fresh inputs and the new Module
05 checkpoints; it is distinct from the historical six-checkpoint study.

The historical matched-hierarchy results answer their original conditional
question. They are preserved as evidence and are not relabelled as the new
five-method fresh-input evaluation. Existing archive paths and manifests
retain their original reproduction identities.
