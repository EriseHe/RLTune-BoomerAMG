# RLTune-BoomerAMG

Research code for online BoomerAMG setup tuning and per-cycle solve control.

## Build from a fresh checkout

The Conda environment contains the Python dependencies. CMake, Make, a C/C++
toolchain, and MPI compiler wrappers (`mpicc` and `mpicxx`) must already be
available on the system.

```bash
conda env create -f environment.yml
conda activate rl
make -C hypre
```

The environment is named `rl` by `environment.yml`, but the experiment scripts
only require that the intended Python environment is active. The HYPRE build is
kept under `hypre/build/` and `hypre/install/`.

## Reproduce a paper experiment

The primary five-branch scalar-diffusion comparison is documented in
[`results/joint/paper_n60_scalar_diffusion_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6_20260731/README.md`](results/joint/paper_n60_scalar_diffusion_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6_20260731/README.md).

The latest diffusion-advection hybrid comparison is documented in
[`results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/README.md`](results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/README.md).

Validate the latest frozen configuration without running HYPRE:

```bash
python -u experiments/joint/solve_control/run_joint_experiment.py \
  --config results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/experiment_config.json \
  --validate-only
```

Run it into a new output directory:

```bash
OUTPUT_DIR=results/joint/hybrid_reproduction \
  ./results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/reproduce.sh
```

The exact stream, seeds, setup/solve contexts, activation boundary, tolerance,
and method roster come from the frozen JSON. Full trajectories and checkpoints
are generated locally; Git tracks only the compact paper evidence bundle.

## Current experiment path

- `experiments/joint/solve_control/run_joint_experiment.py`: JSON experiment
  entry point.
- `experiments/joint/solve_control/configs/`: reusable experiment configs.
- `experiments/joint/solve_control/generate_joint_experiment_plots.py`:
  plot-only entry point.
- `results/joint/paper_*/`: frozen configs, compact summaries, figures, and
  reproduction scripts.

See [`experiments/joint/solve_control/README.md`](experiments/joint/solve_control/README.md)
for runner details.

## Failure protocol

Each learned branch permits up to three setup attempts, followed by one
measured default setup/default solve fallback. Failed or nonconverged primary
work and fallback work are included in runtime and learning feedback. If the
fallback also fails, pending learner updates are rolled back.

## Tests

```bash
python -m unittest discover -s setup/tests -p 'test_*.py' -v
python -m unittest discover -s problems/tests -p 'test_*.py' -v
python -m unittest discover -s solve/tests -p 'test_*.py' -v
python -m unittest discover -s experiments/diagnostics/solve_control -p 'test_*.py' -v
```

Repository ownership and dependency rules are in
[`docs/repository_layout.md`](docs/repository_layout.md).
