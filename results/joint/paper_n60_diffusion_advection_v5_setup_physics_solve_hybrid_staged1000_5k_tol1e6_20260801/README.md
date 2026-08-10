# n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6

Matched 60^3/5K scalar anisotropic diffusion-advection hybrid-context experiment. It compares default setup/solve, the canonical LinUCB v5 setup plus canonical Recursive LSTDQ v3, the physics-linear setup plus physics-linear Recursive LSTDQ v3, and the new hybrid canonical LinUCB v5 setup plus physics-linear Recursive LSTDQ v3. All online solve controllers activate after case 1000.

- frozen config: [`experiment_config.json`](experiment_config.json)
- source config: [`n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6.json`](../../../experiments/joint/solve_control/configs/n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6.json)
- stream SHA-256: `99855ecccfe660366d651a5b6856230a68f77d4ae876e6e3b365c4cae407606f`
- instances: `5000` (`0` warmup + `5000` online)
- problem/grid: `scalar_anisotropic_diffusion_advection` / `[60, 60, 60]`
- methods: `default_setup_default_solve, linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6, linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6, linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6`
- execution: one shared stream, independent mutable learner state per branch, randomized per-instance method order
- timing/recovery: active joint-online bounded-recovery protocol

Follow the repository [setup instructions](../../../README.md), then `cd` to
this directory and run `OUTPUT_DIR=/new/path ./reproduce.sh`.
`stream_manifest.json` records the exact input stream. Plot generation runs
after the numerical runner and can be repeated separately.
