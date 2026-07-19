# Setup Learners

Setup algorithms are grouped by bandit family:

- `linucb/`: independent and shared LinUCB variants, including the active v4
- `bayesian/`: Bayesian linear bandit variants
- `thompson/`: linear, bootstrap, and random-feature Thompson sampling
- `tsallis/`: Tsallis-INF
- `common/`: action features and candidate selection shared across families

Experiment code should import public learners from `SetupPhase.learners` (or
`learners` when `SetupPhase` is already on `sys.path`) instead of importing a
variant file directly.
