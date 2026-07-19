from __future__ import annotations

import os

from train_ppo_setup_common import main


if __name__ == "__main__":
    # Bandit-setup PPO defaults to the Win/mac-aligned setup policy:
    # Shared LinUCB v2 over the tune5 setup space.
    os.environ.setdefault("SETUP_TUNE_DIM", "5")
    os.environ.setdefault("SETUP_BANDIT_METHOD", "linucbv2")
    main(
        setup_mode="bandit",
        default_model_basename="ppo_boomeramg_setup_bandit",
        default_vec_name="vecnormalize_setup_bandit.pkl",
        default_tb_dir="./ppo_logs_setup_bandit",
    )
