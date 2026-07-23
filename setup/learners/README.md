# Setup Learners

Setup algorithms are grouped by bandit family:

- `linucb/`: independent and shared LinUCB variants, including the active v4
- `bayesian/`: Bayesian linear bandit variants
- `thompson/`: linear, bootstrap, and random-feature Thompson sampling
- `tsallis/`: Tsallis-INF
- `common/`: action features and candidate selection shared across families

New experiment code should select and build active learners through
`setup.registry`. Direct learner imports remain supported for
implementation-level tests.

For active families, `config.py` owns the algorithm-specific typed spec and
`factory.py` translates the shared typed factory request into the concrete
learner constructor. Shared action/context configuration remains in
`common/config.py`; its historical algorithm-spec imports are compatibility
reexports only.
