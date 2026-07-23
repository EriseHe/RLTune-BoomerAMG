# Solve Controllers

This is the canonical home of solve-phase controller code. Each
registry-managed online controller family owns its typed `config.py`,
construction `factory.py`, implementation, protocol metadata, and checkpoint
logic:

- `bootstrap/`: bootstrap SARSA-LCB
- `common/`: shared state/action encoding, controller configuration, types, and
  linear primitives
- `lsvi/`: stagewise and hierarchical LSVI-LCB
- `model_based/`: structured cycle-cost/progress control
- `ppo/`: frozen setup-aware PPO config, checkpoint runner, and factory
- `rblspi/`: recursive Bayesian LSTDQ / RBLSPI
- `recursive_lstdq/`: recursive LSTDQ-LCB v1 and v2
- `recursive_mc/`: recursive Monte Carlo LCB
- `sarsa/`: online SARSA(lambda) and behavior policies
- `registry.py`: compatibility re-export of the public solve registry

`SolvePhase/` is a compatibility namespace only. New code should import from
`solve.controllers`, and construct controllers through `solve.registry`. The
registry dispatches typed requests to family factories; it does not call
concrete controller constructors.
