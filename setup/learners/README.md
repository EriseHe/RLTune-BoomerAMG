# Setup Learners

Setup algorithms are grouped by bandit family:

- `linucb/`: independent and shared LinUCB variants, including the retained
  v4, paper-final diffusion-advection v5, and internal quadratic-context v6
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

`SharedLinUCB_AMG_v5` reuses the v4 update and selection algorithm without
modifying v4. It freezes the paper experiment contract to the 8-D
diffusion-advection context and the named Tune-7 `recommended` setup space.

`SharedLinUCB_AMG_v5_RBF` is an experimental comparison variant. It keeps
that v5 contract and update rule, but adds five local Gaussian basis features
for each of `strong_threshold`, `max_row_sum`, and `trunc_factor`. The frozen
paper v5 action encoding is not changed.

`SharedLinUCB_AMG_v6` is an internal experiment that also leaves v5 unchanged.
It builds six non-redundant physical PDE coordinates (three diffusion
scale/contrast coordinates and three directional cell Péclet coordinates),
lifts them to a complete 28-D quadratic basis, and crosses that basis with the
same recommended Tune-7 action features used by v5.
