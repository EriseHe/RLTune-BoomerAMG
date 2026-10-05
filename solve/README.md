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

## LSTDQ implementation

`controllers/recursive_lstdq/v3.py` implements the episode-cluster controller.
`v1.py` remains as its required mean-update, inverse, checkpoint, and
behavior-policy base; it is not separately selectable as an experiment
algorithm. `controllers/recursive_lstdq/common.py` factorizes the shared
state-action scoring.

`controllers/recursive_lstdq/config.py` defines the numerical specs; its factory
constructs V3 with the encoder and action configuration supplied by
`solve.registry`. `controllers/common/` contains the encoders, controller bundle,
TD configuration, and linear primitives. The TD configuration retains its
historical class name because checkpoint configuration fields remain unchanged.

V3 commits one cluster covariance update per successfully committed episode.
The episode runner owns recovery and rollback; the controller owns its numerical
state and NPZ checkpoint. Canonical V1/V3 class paths and checkpoint schemas
remain stable.

## Shared components and checks

A `ControllerBundle` pairs a controller with its exact state encoder. It runs a
case, reports a summary, and preserves checkpoint metadata. Setup selection,
stream order, fallback choice, and reporting belong to the experiment layer.

Setup and solve use the same native interfaces in `hypre/interfaces/` and the
same [PDE streams](../docs/repository_layout.md#pde-problems). Build the interfaces
with `make -C hypre`; see [reproduction](../docs/reproduction.md) for prerequisites.
Run solve tests with `python -m unittest discover -s solve/tests -p 'test_*.py'`.
