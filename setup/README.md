# Setup phase

The SISC studies select BoomerAMG hierarchy parameters with shared context-action
LinUCB V4. `registry.py` accepts `default` and `linucb`; it constructs the learner
through the LinUCB factory. The native solver defaults remain in `space.py`.

- `space.py`: named categorical configuration spaces, parameter grids, and the
  setup observation encoder used by the solve controller, and stable action
  enumeration.
- `learners/linucb/`: V4 learning, checkpoint persistence, and same-instance setup
  reselection.
- `learners/common/`: mixed-type action features, compact action catalogs,
  candidate schedules, and shared constructor inputs.
- `tests/`: numerical, candidate-schedule, failure-feedback, and registry checks.

The setup and solve phases share [PDE streams](../problems/README.md) and
[HYPRE bindings](../hypre/bindings/README.md). The selected setup and its recovery
attempts are evaluated by the joint experiment runner, which supplies the
completion cost used by LinUCB.

From the repository root, install the environment and native interfaces using
[the reproduction guide](../docs/reproduction.md). Run the setup tests with
`python -m unittest discover -s setup/tests -p 'test_*.py'`.
