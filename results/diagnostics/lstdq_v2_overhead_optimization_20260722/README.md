# LSTDQ v2 overhead optimization audit

Date: 2026-07-22

## Scope

This pass changes only the implementation of the already selected LSTDQ v2
controller. It does not change its action set, state features, Bellman mean
update, coverage matrix, rolling-MAD window, residual floor, beta, epsilon
schedule, lower-bound tie break, recovery target, checkpoint contents, or
update order.

The retained optimizations are:

1. Evaluate the 288-dimensional coverage quadratic through the exact
   9-dimensional action-basis contraction instead of materializing all 41
   joint feature rows for a dense 41-by-288 product.
2. Use in-place BLAS rank-one updates for the same Sherman--Morrison equations
   when SciPy is available. The production `rl` environment contains SciPy.
   A row-major NumPy fallback is retained and benchmarked when SciPy is absent.
3. Store the same last 2048 post-fit TD residuals in a numeric circular window
   and compute the same finite-sample MAD using partial ordering. This removes
   the per-cycle deque-to-array conversion but does not approximate or thin the
   residual sample.
4. Reuse selected joint-feature and difference workspaces. The values written
   to the LSTDQ and coverage equations are unchanged.

## Learning-equivalence gate

A final 10,000-transition replay used the production dimensions (32 state
features, 9 action-basis functions, 41 actions), beta 4, the production epsilon
schedule, and a reference controller retaining the pre-optimization scoring,
MAD, joint-feature, and matrix-update code.

- selected-action mismatches: **0 / 10,000**
- greedy-action mismatches: **0 / 10,000**
- maximum absolute score difference: `2.0713e-15`
- maximum absolute Q difference: `1.5925e-15`
- maximum absolute uncertainty difference: `4.1547e-16`
- final parameter maximum absolute difference: `1.8908e-15`
- final LSTD inverse maximum absolute difference: `3.3307e-16`
- final coverage inverse maximum absolute difference: `2.2204e-16`
- maximum residual-scale difference: `1.4763e-15`
- inverse rebuilds: `0` in both implementations

The differences are floating-point last-bit effects from equivalent BLAS and
NumPy operation order; they are more than two orders of magnitude below the
existing `1e-12` decision tie tolerance.

Automated coverage was added for the factorized quadratic, full learning
trajectory, rolling-window wraparound, checkpoint, and rollback. The final
validation ran 39 SolvePhase tests plus 48 controller/runner/recovery tests
(87 total), all passing. Targeted tests also pass in a no-SciPy runtime, where
the implementation automatically uses the old row-major NumPy path.

## Timing

### Deterministic production-dimension benchmark

Seven repetitions of 800 select-and-update transitions in the production
`/opt/anaconda3/envs/rl` environment gave:

| implementation | median microseconds / transition |
|---|---:|
| conservative old-path reference | 464.52 |
| optimized LSTDQ v2 | 208.86 |

This is a **55.04% reduction** in the controller hot path. The old-path
reference already shares the new selected-feature workspace, so this estimate
is conservative with respect to the complete pre-pass implementation.

### Real 60-cubed HYPRE smoke

An alternating-order replay of 12 fixed-setup cases from the disjoint
calibration trace produced 571 cycles for each implementation:

| metric | old path | optimized | change |
|---|---:|---:|---:|
| controller ms / case | 28.857 | 24.615 | -14.70% |
| decision us / cycle | 224.276 | 152.328 | -32.08% |
| update us / cycle | 382.178 | 364.968 | -4.50% |
| primary failures | 0 | 0 | unchanged |
| unrecovered failures | 0 | 0 | unchanged |
| native runtime ms / case | 750.536 | 746.876 | no observed regression |

The native timing difference is treated as ordinary short-run timing noise;
this smoke is a systems-safety check, not a new performance claim.

## Interpretation

The canonical 4K result remains the immutable algorithm-screening result. It
was not rerun or retroactively rewritten. This pass establishes exact algorithm
equivalence and lower controller cost; a future formal end-to-end run can
measure how much of the hot-path reduction survives at 4K scale under the
paper's locked protocol.
