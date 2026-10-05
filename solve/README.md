# Solve phase

The SISC studies use recursive LSTDQ V3 to select relaxation weights after each
AMG cycle. `registry.py` accepts `default`, `fixed`, and `recursive_lstdq_v3`.

- `controllers/recursive_lstdq/`: the V3 episode-cluster controller and the V1
  numerical base it inherits.
- `controllers/common/`: state and action encoders, construction inputs,
  controller/checkpoint pairing, and linear numerical primitives.
- `core/episode.py`: the cycle loop, completion-cost targets, recovery accounting,
  and transactional commit or rollback of controller learning.
- `core/outcomes.py`: failure classification shared with the native adapter.
- `tests/`: native status and timing, numerical updates, recovery transactions,
  controller construction, and checkpoint round trips.

A `ControllerBundle` pairs a controller with its exact state encoder. It runs a
case, reports a summary, and preserves checkpoint metadata. Setup selection,
stream order, fallback choice, and reporting belong to the experiment layer.

Setup and solve use the same native interfaces in `hypre/interfaces/` and the
same [PDE streams](../problems/README.md). Build the interfaces with
`make -C hypre`; see [reproduction](../docs/reproduction.md) for prerequisites.
Run solve tests with `python -m unittest discover -s solve/tests -p 'test_*.py'`.
