# Setup phase

The SISC studies select BoomerAMG hierarchy parameters with shared context-action
LinUCB. `registry.py` accepts `default` and `linucb`; it constructs the learner
through the LinUCB factory. The native solver defaults remain in `space.py`.

- `space.py`: named categorical configuration spaces, parameter grids, and the
  setup observation encoder used by the solve controller, and stable action
  enumeration.
- `learners/linucb/`: learning, checkpoint persistence, and same-instance setup
  reselection.
- `learners/common/`: mixed-type action features, compact action catalogs,
  candidate schedules, and shared constructor inputs.
- `tests/`: numerical, candidate-schedule, failure-feedback, and registry checks.

## LinUCB implementation

[SharedLinUCB](learners/linucb/learner.py) shares a linear context-action model
across configurations and preserves its original numerical updates, RNG state,
and NPZ checkpoint format. `learners/linucb/setup_reselection.py` handles setup
failure and same-instance reselection; the experiment layer owns the recovery
policy and supplies the observed completion cost.

`learners/common/` supplies the mixed-type parameter specification, generic
action features, compact catalogs, cached factorized features, and ahead-of-time
candidate schedules. `learners/common/config.py` holds shared construction
inputs; `learners/linucb/config.py` and `learners/linucb/factory.py` own the
LinUCB-specific construction.

An experiment runner uses `setup.registry.make_setup_learner_spec` and
`setup.registry.build_online_setup_learner`. Canonical class paths remain under
`setup.learners.linucb`; official checkpoints use NPZ with `allow_pickle=False`
and do not require historical pickle module aliases.

## Shared components and checks

The setup and solve phases share
[PDE streams](../docs/repository_layout.md#pde-problems) and
[HYPRE bindings](../hypre/bindings/README.md). The joint experiment runner
evaluates the selected setup and its recovery attempts.

From the repository root, install the environment and native interfaces using
[the reproduction guide](../docs/reproduction.md). Run the setup tests with
`python -m unittest discover -s setup/tests -p 'test_*.py'`.
