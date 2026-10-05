# Scalar PDE problems

This package defines the two PDE streams used by the SISC studies. It does not
import learners, controllers, or experiment runners.

- `scalar_anisotropic_diffusion.py`: scalar anisotropic diffusion with zero
  advection coefficients.
- `scalar_anisotropic_diffusion_advection.py`: the same diffusion family with
  independently sampled advection components.
- `amg.py`: the shared stencil-0 native coefficient mapping, coefficient sampler,
  and normalized PDE contexts.
- `streams.py`: deterministic instance streams generated from independent child
  seeds in their original order.
- `registry.py`: the two problem identities and the shared setup context views.

The underlying stream retains its eight-field coefficient representation.
Official diffusion experiments select `diffusion3d`; diffusion-advection
experiments select `canonical_no_c_mean`. Both retain an intercept and remove
redundant mean features from the learner-visible view. The original default
V4 context projection remains available for existing configuration defaults.
