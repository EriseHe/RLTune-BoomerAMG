# n80_scalar_diffusion_paper_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6

Paper comparison on one strictly paired 80^3/5K scalar anisotropic diffusion stream. The five branches are default setup plus default solve; frozen LinUCB v5 canonical-8D setup plus default solve; physics-linear-7D LinUCB setup plus default solve; frozen LinUCB v5 plus Recursive LSTDQ v3 using the prior canonical solve context; and physics-linear LinUCB plus Recursive LSTDQ v3 using the aligned physics-linear solve context. Both solve controllers remain inactive through the first 1000 cases and learn online from case 1001. Learned setup branches share the recommended Tune-7 space, structured-512 candidate schedule, seeds, relative tolerance 1e-6, and 50-cycle limit.

- source config: `/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/joint/solve_control/configs/n80_scalar_diffusion_paper_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6.json`
- stream SHA-256: `148addb1f891440b8b06efc1fb0b2c3e4fcc7d560cd90cf411c683c2c305757c`
- instances: `5000` (`0` warmup + `5000` online)
- problem/grid: `scalar_anisotropic_diffusion` / `[80, 80, 80]`
- methods: `default_setup_default_solve, linucb_v5_canonical8d_recommended_structured512_default_solve, linucb_physics_linear7d_recommended_structured512_default_solve, linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6, linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6`
- execution: one shared stream, independent mutable learner state per branch, randomized per-instance method order
- timing/recovery: active joint-online bounded-recovery protocol

`config.json` and `stream_manifest.json` are the resolved low-level protocol;
`experiment_config.json` is the reusable high-level configuration.
Run `OUTPUT_DIR=/new/path ./reproduce.sh` to reproduce without overwriting this directory.
When `reporting.generate_plots` is true, the high-level entry point invokes the separate plot-only generator after the runner finishes.
Plots can also be regenerated with `generate_joint_experiment_plots.py`.
