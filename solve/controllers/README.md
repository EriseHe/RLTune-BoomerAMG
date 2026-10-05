# Solve controller

`recursive_lstdq/v3.py` implements the episode-cluster recursive LSTDQ controller
used by the SISC studies. `v1.py` remains as its required mean-update, inverse,
checkpoint, and behavior-policy base. It is not a separately selectable
experiment algorithm. `common.py` factorizes the shared state-action scoring.

`recursive_lstdq/config.py` defines the numerical specs; its factory constructs
V3 with the encoder and action configuration supplied by `solve.registry`.
`common/` contains the shared encoders, controller bundle, TD configuration, and
linear primitives. The retained TD configuration keeps its historical class
name because checkpoint configuration fields remain unchanged.

V3 commits one cluster covariance update per successfully committed episode.
The episode runner owns recovery and rollback, while the controller owns its
numerical state and NPZ checkpoint. Canonical V1/V3 class paths and checkpoint
schemas remain stable.
