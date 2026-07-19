# Problems

Shared PDE definitions and deterministic instance streams used by both learning
phases and by joint experiments.

- `amg.py`: 27-point Laplacian and diffusion-convection matrix definitions.
- `cases.py`: reusable matrix, RHS, grid, and evaluation case specifications.
- `scalar_anisotropic_diffusion.py`: scalar anisotropic diffusion stream.
- `streams.py`: seeded multi-instance stream generation.

This package defines problems only. It must not import setup learners, solve
controllers, or experiment runners.
