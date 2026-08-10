# n60_diffusion_advection_context_ablation_seed2_staged1000_5k_tol1e6

Matched new-seed 60^3/5K scalar anisotropic diffusion-advection context ablation. This changes only the PDE stream from the scalar-diffusion experiment: independently sampled a_x,a_y,a_z in [1,1000] are enabled. The same seven branches, recommended Tune-7 space, structured-512 candidate schedule, seeds, tolerances, cycle limit, and solve-controller activation are retained. The two new context branches remain setup-only with default solve; Recursive LSTDQ v3 remains present only for the original v5 and physics-linear branches and activates after case 1000.

- source config: `/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/joint/solve_control/configs/n60_diffusion_advection_context_ablation_seed2_staged1000_5k_tol1e6.json`
- stream SHA-256: `99855ecccfe660366d651a5b6856230a68f77d4ae876e6e3b365c4cae407606f`
- instances: `5000` (`0` warmup + `5000` online)
- problem/grid: `scalar_anisotropic_diffusion_advection` / `[60, 60, 60]`
- methods: `default_setup_default_solve, linucb_v5_canonical8d_recommended_structured512_default_solve, linucb_physics_linear7d_recommended_structured512_default_solve, linucb_canonical_no_c_mean7d_recommended_structured512_default_solve, linucb_canonical_means_only3d_recommended_structured512_default_solve, linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6, linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6`
- execution: one shared stream, independent mutable learner state per branch, randomized per-instance method order
- timing/recovery: active joint-online bounded-recovery protocol

`config.json` and `stream_manifest.json` are the resolved low-level protocol;
`experiment_config.json` is the reusable high-level configuration.
Run `OUTPUT_DIR=/new/path ./reproduce.sh` to reproduce without overwriting this directory.
When `reporting.generate_plots` is true, the high-level entry point invokes the separate plot-only generator after the runner finishes.
Plots can also be regenerated with `generate_joint_experiment_plots.py`.
