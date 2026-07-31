# Problems

Shared PDE definitions and deterministic instance streams used by both learning
phases and by joint experiments.

- `amg.py`: 27-point Laplacian and diffusion-convection matrix definitions.
- `cases.py`: reusable matrix, RHS, grid, and evaluation case specifications.
- `scalar_anisotropic_diffusion.py`: scalar anisotropic diffusion stream.
- `scalar_anisotropic_diffusion_advection.py`: the same scalar anisotropic
  diffusion family with independently sampled advection components per case.
- `streams.py`: seeded multi-instance stream generation.

This package defines problems only. It must not import setup learners, solve
controllers, or experiment runners.
