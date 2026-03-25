## Repository Layout

- `SolvePhase/hypre/`: vendored HYPRE source tree
- `SetupPhase/learning-setup/`: setup-phase tuning harnesses and native solver wrapper
- `SetupPhase/learning-setup/solver/`: Python wrapper plus the native runtime libraries used by the setup-phase scripts

## First Pull / Reproducible Local Environment

The repo is intended to be usable from a fresh clone without carrying around local CMake build trees or experiment output folders.

What should be in Git:

- source code
- the vendored HYPRE source tree
- the setup-phase Python code
- the runtime libraries currently used by the setup-phase solver wrapper in `SetupPhase/learning-setup/solver/`

What should stay out of Git:

- local `build/` directories
- generated CMake files
- experiment plots / CSVs / JSON summaries
- local object files, import libraries, and other one-off native build artifacts

At the moment, the setup-phase Python wrapper loads a compiled shared library from:

- `SetupPhase/learning-setup/solver/libamg_setup_solver.dll` on Windows
- `SetupPhase/learning-setup/solver/libamg_setup_solver.dylib` on macOS

and expects the HYPRE runtime library alongside it. If you rebuild those native pieces locally, place the rebuilt library next to `boomeramg.py` or point `AMG_SETUP_SOLVER_LIB` at it.

## Dependencies (Solve Phase)

The following Python packages are required:

- numpy  
- gymnasium  
- stable-baselines3  
- torch  
- matplotlib  
- tensorboard  

## Running the Solve Phase

1. Navigate to `SolvePhase/Hypre/src/test/`.
2. recompile if needed
3. Run the training script:

```bash
python train_ppo.py
```

![Current solve phase progress](./Resource/current_solvephase_progress.png)

*Figure: Current solve phase progress.*
