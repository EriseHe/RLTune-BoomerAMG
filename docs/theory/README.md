# Period-two smoothing references

The official matched-hierarchy Run 05 prescribes `(2.85, 1.10)`, high weight first,
with the phase reset at each primary solve. The pair minimizes the weighted
polynomial smoothing objective over all pairs from `{1, 1.05, ..., 3}` for the
normalized SPD surrogate. Its multilevel runtime is evaluated experimentally.

| Preserved reference | Purpose |
|---|---|
| [Weighted-minimax derivation](period_two_weighted_minimax_20260928.md) | Normalization, continuous optimum, discrete pair selection and the limits of transfer to BoomerAMG |
| [Finite minimax verification](period_two_minimax_verification_20260928.json) | High-precision evaluation of analytic extrema for all 861 unordered grid pairs |
| [Native smoother audit](anchored_schedule_verification_20260927.json) | Recorded full-row l1-Jacobi, action granularity and 18/18/9 profile checks used by the prescribed schedule |

These dated records are preserved as evidence. Some introductory workflow names
refer to earlier development plans; the current studies and commands are defined
in the [reproduction guide](../reproduction.md) and
[official Run 05 record](../../experiments/paper_final/05_policy/RUN05.md).
The native audit includes an earlier anchored coefficient calculation; that pair
is outside the current Run 05 method roster.

Regenerate the deterministic minimax check into a new file:

```sh
python -m experiments.paper_final.verify_period_two_minimax --output results/theory/run05_minimax_verification.json
```

The verifier requires mpmath, included in the computational dependencies. It
checks finite grid values using high precision rather than interval arithmetic.
It does not establish minimum multilevel completion time, prove a phase-order
preference, or extend the SPD smoothing result to nonsymmetric advection.

Exact copies of the three supporting records are included in the
[Run 05 frozen-input bundle](../../experiments/paper_final/reproduction/matched_policy/README.md).
Regeneration does not overwrite their accepted hashes.
