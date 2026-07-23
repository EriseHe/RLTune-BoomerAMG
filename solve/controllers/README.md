# Solve Controllers

This is the canonical home of solve-phase controller code. Each controller
family owns its configuration spec, implementation, and checkpoint logic:

- `bootstrap/`: bootstrap SARSA-LCB
- `common/`: shared controller configuration, types, and linear primitives
- `lsvi/`: stagewise and hierarchical LSVI-LCB
- `model_based/`: structured cycle-cost/progress control
- `ppo/`: PPO controller policy definitions
- `rblspi/`: recursive Bayesian LSTDQ / RBLSPI
- `recursive_lstdq/`: recursive LSTDQ-LCB v1 and v2
- `recursive_mc/`: recursive Monte Carlo LCB
- `sarsa/`: online SARSA(lambda), state encoding, and behavior policies
- `registry.py`: the typed construction boundary used by experiment runners

`SolvePhase/` is a compatibility namespace only. New code should import from
`solve.controllers`, and construct controllers through `solve.registry`.
