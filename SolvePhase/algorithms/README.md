# Solve Algorithms

Solve controllers are grouped by algorithm family:

- `ppo/`: PPO policy definitions used by SB3 training
- `sarsa/`: online SARSA(lambda), state encoding, and behavior policies
- `lcb/`: shared-action Recursive MC-LCB, Recursive LSTDQ-LCB, bootstrap
  SARSA-LCB, and stagewise LSVI-LCB controllers

`SolvePhase/core/` contains solver environments and algorithm-independent
outcome handling. Joint runners import controllers through these packages.
