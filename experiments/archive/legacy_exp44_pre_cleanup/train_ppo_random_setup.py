from __future__ import annotations

from train_ppo_setup_common import main


if __name__ == "__main__":
    main(
        setup_mode="random",
        default_model_basename="ppo_boomeramg_setup_random",
        default_vec_name="vecnormalize_setup_random.pkl",
        default_tb_dir="./ppo_logs_setup_random",
    )
