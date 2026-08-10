# Composable Joint Setup/Solve Experiments

`run_joint_experiment.py` is the current entry point for paired online
BoomerAMG experiments. It reads one JSON file, validates the problem stream and
method roster, and delegates every branch to the shared execution and recovery
engine.

Complete the environment and HYPRE build in the repository
[`README.md`](../../../README.md) before running an experiment.

## Run from a JSON configuration

Validate a configuration without launching HYPRE:

```bash
python -u experiments/joint/solve_control/run_joint_experiment.py \
  --config experiments/joint/solve_control/configs/n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6.json \
  --validate-only
```

Run the same configuration into a fresh directory:

```bash
python -u experiments/joint/solve_control/run_joint_experiment.py \
  --config experiments/joint/solve_control/configs/n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6.json \
  --output-dir results/joint/hybrid_reproduction
```

The runner refuses to overwrite a non-empty output directory.

## Configuration contract

The high-level JSON fixes:

- the PDE, grid, deterministic shared stream, and all seeds;
- the setup action space, candidate schedule, and setup context;
- the solve tolerance, cycle limit, actions, controller, and solve context;
- the independent setup-plus-solve branches and solve activation case;
- reporting and plot-generation settings.

Each method owns independent mutable learner state. Methods see the same PDE
instances, and their execution order is randomized within each paired case.
Setup and solve contexts are configured separately, so aligned and hybrid
context experiments use the same runner.

## Reproduce a frozen paper run

Every completed run receives an `experiment_config.json` and `reproduce.sh`.
For example:

```bash
OUTPUT_DIR=results/joint/hybrid_reproduction \
  ./results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/reproduce.sh
```

Reference bundles:

- [primary 60^3 scalar-diffusion comparison](../../../results/joint/paper_n60_scalar_diffusion_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6_20260731/README.md)
- [latest 60^3 diffusion-advection hybrid comparison](../../../results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/README.md)

## Outputs and plotting

Full local runs write trajectories, checkpoints, mutable final states, resolved
protocol metadata, compact summaries, and a reproduction script. Raw generated
artifacts are ignored by Git; curated paper bundles retain only the frozen
config, stream manifest, compact reports, and primary figures.

Numerical execution and plotting are separate. Regenerate plots after a run:

```bash
python -u experiments/joint/solve_control/generate_joint_experiment_plots.py \
  --result-dir results/joint/hybrid_reproduction \
  --rolling-window 100
```

## Recovery protocol

A learned branch may make up to three measured setup attempts. A setup failure,
solve failure, non-finite result, or max-cycle nonconvergence then triggers one
measured default setup/default solve fallback. All consumed work is included in
runtime and learner feedback; pending updates are rolled back only if fallback
also fails.

## Main modules

- `run_joint_experiment.py`: config parsing, validation, and top-level run.
- `composable_joint_4k.py`: method construction and setup/solve composition.
- `joint_4k_execution.py`: paired execution, updates, recovery, and artifacts.
- `generate_joint_experiment_plots.py`: plot-only entry point.
