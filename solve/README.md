# Solve phase

The SISC studies use recursive LSTDQ to select relaxation weights after each
AMG cycle. `registry.py` accepts `default`, `fixed`, and `recursive_lstdq`.

- `controllers/recursive_lstdq/`: the single episode-cluster controller,
  numerical configuration, and factory.
- `controllers/common/`: state and action encoders, construction inputs,
  controller/checkpoint pairing, and linear numerical primitives.
- `core/episode.py`: the cycle loop, completion-cost targets, recovery accounting,
  and transactional commit or rollback of controller learning.
- `core/outcomes.py`: failure classification shared with the native adapter.
- `tests/`: native status and timing, numerical updates, recovery transactions,
  controller construction, and checkpoint round trips.

## LSTDQ implementation

[RecursiveLstdqController](controllers/recursive_lstdq/controller.py) contains
the recursive mean update, inverse maintenance, episode-cluster covariance,
and checkpoint operations in one implementation.
`controllers/recursive_lstdq/common.py` factorizes the shared state-action scoring.

`controllers/recursive_lstdq/config.py` defines the numerical specs; its factory
constructs the controller with the encoder and action configuration supplied by
`solve.registry`. `controllers/common/` contains the encoders, controller bundle,
TD configuration, and linear primitives. The TD configuration retains its
historical class name because checkpoint configuration fields remain unchanged.

The controller commits one cluster covariance update per successfully committed episode.
The episode runner owns recovery and rollback; the controller owns its numerical
state and NPZ checkpoint. The saved checkpoint fields and schema remain unchanged.

## Shared components and checks

A `ControllerBundle` pairs a controller with its exact state encoder. It runs a
case, reports a summary, and preserves checkpoint metadata. Setup selection,
stream order, fallback choice, and reporting belong to the experiment layer.

Setup and solve use the same native interfaces in `hypre/interfaces/` and the
same [PDE streams](../docs/repository_layout.md#pde-problems). Build the interfaces
with `make -C hypre`; see [reproduction](../docs/reproduction.md) for prerequisites.
Run solve tests with `python -m unittest discover -s solve/tests -p 'test_*.py'`.
