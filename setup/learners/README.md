# Setup learner

`linucb/SharedLinUCB_AMG_v4.py` is the setup learner used by both official SISC
studies. It shares a linear context-action model across configurations and
preserves its original numerical updates, RNG state, and NPZ checkpoint format.
`linucb/setup_reselection.py` handles setup failure and same-instance reselection
without moving experiment policy into the learner.

`common/` supplies the mixed-type parameter specification, generic action
features, compact catalogs, cached factorized features, and ahead-of-time
candidate schedules. `common/config.py` holds the shared construction inputs;
`linucb/config.py` and `linucb/factory.py` own the V4-specific construction.

Use `setup.registry.make_setup_learner_spec` and
`setup.registry.build_online_setup_learner` from an experiment runner. Canonical
class paths remain under `setup.learners.linucb`; official checkpoints use NPZ
with `allow_pickle=False` and do not require historical pickle module aliases.
