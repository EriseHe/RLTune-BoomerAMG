# n60_scalar_diffusion_context_ablation_seed2_staged1000_5k_tol1e6

New-seed 60^3/5K scalar anisotropic diffusion context ablation extending the prior paper comparison. The original five branches are preserved, and two setup-only branches are added with default solve: canonical-no-c-mean [1,c_x,c_y,c_z,a_x,a_y,a_z] and means-only [1,c_mean,a_mean]. For this scalar-diffusion stream, a_x=a_y=a_z=0, so a_mean is identically zero. All learned setup branches share the recommended Tune-7 space, structured-512 candidate schedule, and seed offset. Recursive LSTDQ v3 remains present only for the original v5 and physics-linear branches, activates after case 1000, and uses relative tolerance 1e-6 with a 50-cycle limit.

- frozen config: [`experiment_config.json`](experiment_config.json)
- source config: [`n60_scalar_diffusion_context_ablation_seed2_staged1000_5k_tol1e6.json`](../../../experiments/joint/solve_control/configs/n60_scalar_diffusion_context_ablation_seed2_staged1000_5k_tol1e6.json)
- stream SHA-256: `d924852d187f2890fb3b8a98e0bd5bf2790d701caa01110808de1d97b421927e`
- instances: `5000` (`0` warmup + `5000` online)
- problem/grid: `scalar_anisotropic_diffusion` / `[60, 60, 60]`
- methods: `default_setup_default_solve, linucb_v5_canonical8d_recommended_structured512_default_solve, linucb_physics_linear7d_recommended_structured512_default_solve, linucb_canonical_no_c_mean7d_recommended_structured512_default_solve, linucb_canonical_means_only3d_recommended_structured512_default_solve, linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6, linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6`
- execution: one shared stream, independent mutable learner state per branch, randomized per-instance method order
- timing/recovery: active joint-online bounded-recovery protocol

Follow the repository [setup instructions](../../../README.md), then `cd` to
this directory and run `OUTPUT_DIR=/new/path ./reproduce.sh`.
`stream_manifest.json` records the exact input stream. Plot generation runs
after the numerical runner and can be repeated separately.
