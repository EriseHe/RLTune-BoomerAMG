# Compatibility Imports

This directory preserves historical `SolvePhase.algorithms.*` import paths.
It contains no controller implementations. Canonical code is grouped by
controller family under `solve/controllers/`.

Existing callers can continue to use `SolvePhase.algorithms.sarsa`,
`SolvePhase.algorithms.lcb`, and the family-specific compatibility modules.
New code should use `solve.controllers` and `solve.registry`.
