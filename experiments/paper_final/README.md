# Paper final work

## Current plan — purpose clarified September 28, 2026

**Module 05 prepares trained frozen setup selectors for Module 06. It is not primarily another standalone online-performance/payback experiment.** This corrects the preceding plan, which mistakenly foregrounded training-time competition and activated prescribed policies from problem 1.

**Module 04 remains unchanged.** The documentation edits do not launch experiments, change learner/solver code, overwrite results or rename historical paths. See the [current completion plan](../../docs/theory/paper_completion_plan_20260915.md) for the full protocol and qualifications.

## Logical module map

| Module | Purpose |
|---|---|
| 01–03 | Existing numerical, development and activation diagnostics |
| **04** | Existing complete online comparison and cost/payback analysis — unchanged |
| **05** | **Matched-budget training to produce frozen setup/controller checkpoints** |
| **06** | **Fresh-input evaluation of those frozen trained methods; optional crossed controls** |
| **07** | Theory integration and deterministic proof checks |
| 08 | Additional conventional baselines/problem breadth when warranted |
| 09 | Retrained feedback/cadence ablations only for stronger claims |
| 10 | Optional new learning algorithms, not prerequisites |
| 11 | Final paper assembly and reproducibility |

The former logical Module 05 / physical `05_policy` is Module 06; former `06_theory` is Module 07. Existing scripts, imports, dataset filenames and result directories keep their historical names.

## Module 05: checkpoint preparation

For each prescribed replicate, execute one shared 1000-problem setup LinUCB prefix with fixed relaxation weight one. Clone the **complete** setup state, including statistics, histories, candidate cursors and RNG, then continue on the same next 4000 problem inputs:

| Branch | Problems 1–1000 | Problems 1001–5000 | Freeze at completion |
|---|---|---|---|
| W1 | Common W1 setup-learning prefix | Setup learning under weight one | B_W1 |
| Fixed | Same cloned prefix | Setup learning under prescribed w_dev | B_Fixed and w_dev |
| Periodic | Same cloned prefix | Setup learning under (2.85,1.10), high first | B_Periodic and schedule |
| RL | Same cloned prefix | Setup learning with online cycle-level LSTDQ | B_RL and pi_RL |

All setup selectors receive 5000 logical problem opportunities; designated Fixed, Periodic and RL continuations each receive 4000 adaptation opportunities. They may generate different hierarchies after branching and update only from their own costs. Retain the inherited W1 observations in every branch. Do not run four separately timed prefixes and assume seeds guarantee identical learned states.

The known fixed and periodic policies intentionally switch at problem **1001** in this checkpoint-training design. This matches the user's intended preparation budget; it is not a claim that delayed activation is their best from-scratch deployment protocol.

Preserve the current Module 04 algorithm, setup space, features, cap 50, tolerance 1e-6, 18/18/9 profile, RL settings and rollback-unrecovered protocol. Reset schedule phase per primary solve. Keep fallback unchanged. No new solve-bandit is required. Select w_dev on separate development inputs before training; retain the grid-minimax rationale and limitations for (2.85,1.10). Per-instance fixed oracles remain frozen-evaluation diagnostics, not training branches.

**Primary artifacts:** all final setup snapshots, the trained RL controller, full state/config/input provenance, numerical audits, and cost/outcome logs. One common prefix plus four continuations requires 17000 physical method–problem executions before recovery. Training-time savings are available for diagnostics, but **RL beating Periodic during these runs is not a completion or acceptance criterion**. The scientific comparison of interest is Module 06 on fresh inputs.

## Module 06: frozen post-training comparison

The primary practical comparison pairs each solve method with its own comparably trained setup selector:

- B_W1 + weight one;
- B_Fixed + w_dev;
- B_Periodic + (2.85,1.10);
- B_RL + frozen pi_RL.

Evaluate on common fresh inputs after freezing setup and solve models/statistics. No test cost changes a learner, a selected fixed weight or a schedule. Prespecify the frozen selector's scoring/exploration and candidate convention consistently across branches. Cache input/source hierarchy choices for repetitions and any crossed controls. Frozen RL uses its state feedback but does not update parameters or uncertainty statistics; disable random exploration as prescribed.

For this own-pairing comparison, charge **setup inference and construction plus solve/controller execution and recovery**. Different sources may choose different hierarchies, so setup cost does not cancel. Report native and inclusive components and final success/failure separately. Training expense is described by Module 05; frozen evaluation alone does not establish online payback, which remains a distinct Module 04 result.

A compact optional crossed comparison helps attribute any gap:

| Hierarchy source | Periodic solve | Frozen RL solve |
|---|---|---|
| B_Periodic | Own pairing | Transfer |
| B_RL | Transfer | Own pairing |

Within a row, hold matrix/RHS, hierarchy and initial solve state fixed; only solve policy changes. Charge policy-induced recovery even when the common primary setup is omitted from a within-row continuation contrast. A four-source exhaustive cross is not automatically required. Fixed-weight scans can be retained as source-specific numerical diagnostics; do not call a weight selected on one source optimal on another.

Matching training history removes the asymmetry of giving only RL its own adapted setup. It does not guarantee optimality or that all final models are equally well learned. The own-pairing result is a frozen whole-method comparison; the crossed result measures conditional controller effects. Neither requires proving RL indispensable.

## Existing material and historical records

The full previous index, development summaries, original stage-directory table and links are preserved at [revision 53f48f8](https://github.com/EriseHe/RLTune-BoomerAMG/blob/53f48f8ab16cc91f38aa7863c964147e132ca521/experiments/paper_final/README.md). Its earlier activation recommendation is superseded by the matched training purpose above; its historical evidence is not altered.

- [Current detailed completion plan](../../docs/theory/paper_completion_plan_20260915.md)
- [Original detailed mathematics and execution history](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md)
- [01 numerical checks](01_numerics/README.md)
- [02 diagnostics](02_diagnostics/README.md)
- [03 activation](03_activation/README.md)
- [04 existing online protocol](04_online/README.md)
- [General online runner](../joint/solve_control/run_paper_final.py)
- [Shared-prefix execution](../joint/solve_control/joint_4k_execution.py)
- [Existing frozen cross-source diagnostic](../diagnostics/solve_control/diagnose_context_activation_crossed.py)
- [Theory review](../../docs/theory/theory_review_and_revision_plan_20260915.md)

Start paper editing and Module 07 in parallel. This plan records intended work only; it does not imply that new runners, calibrated constants, seed manifests, or completed checkpoints already exist.
