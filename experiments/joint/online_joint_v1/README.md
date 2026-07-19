# Joint Online Setup Bandit + Solve RL

This branch starts both learners from scratch. Shared LinUCB v4 chooses Tune7
categorical setup parameters for each new problem and updates from that
problem's final joint runtime. PPO changes the relaxation weight at each AMG
cycle and updates after each rollout.

The default solve action is discrete: `w` in `{1.2, 1.4, 1.5, 1.6}` at every
cycle. This keeps cold-start exploration inside a convergent range while still
allowing a state-dependent schedule. Set `ACTION_MODE=continuous_residual` to
run the retained Exp44-style residual action instead.

The default PPO policy is an MLP because the observation is Markov and MLP
inference is material on small M2 test problems. Set `MODEL_TYPE=lstm` to use
the retained recurrent policy.

PPO uses an undiscounted episodic objective (`gamma=1`) and charges a small
per-cycle policy cost matching the measured inference overhead. The reward
therefore targets end-to-end solve time rather than native solver time alone.
It also adds a signed log-residual potential difference for immediate credit;
with `gamma=1` this term telescopes over a converged episode and preserves the
total-time objective.

The cold-start profile uses a conservative one-parameter-at-a-time Tune7
neighborhood and four default-setup episodes before exploration. Every choice
and update is still online. Set `SETUP_ACTION_SPACE=full_cartesian` to remove
this guard after the learner has enough feedback.

Successful solves that consume more than 60% of the cycle budget receive a
continuous robustness penalty in the setup-bandit loss. This prevents a setup
that only narrowly converges on training cases from dominating held-out use.
Final exploit also applies an online safety shield: only observed arms with no
failure and no solve above 70% of the cycle budget are eligible. Exploration
data is retained; the shield only controls deployment evaluation.

Run a native smoke test first:

```bash
conda activate rl
./experiments/joint/online_joint_v1/run_online_joint.sh smoke
```

Run the bounded M2 signal profile:

```bash
./experiments/joint/online_joint_v1/run_online_joint.sh signal
```

The signal profile uses a `30^3` problem, a 13-arm conservative Tune7
neighborhood, 8,192 PPO steps, and 48 held-out cases. It finishes well within
one hour on the target M2 MacBook Air. Override any setting from the shell when
needed, for example `TOTAL_TIMESTEPS=16384 EVAL_CASES=64 ... signal`.

Each run writes `config.json`, the PPO model, the online bandit state,
`result.json`, and `console.log` under
`results/joint/online_joint_v1/run_logs/`. That directory is ignored by Git.
`result.json` reports the evolving learner's first/last windows and a
held-out exploit-only replay against default setup/default solve, final
bandit/default solve, and final bandit/fixed `w=1.4` and `w=1.6`. Speed
percentages are only reported when both methods converge on every held-out
case. The report also includes paired case-level improvement, win rate, and a
bootstrap 95% interval. Bandit selection and PPO inference time are included
in candidate end-to-end runtimes.
