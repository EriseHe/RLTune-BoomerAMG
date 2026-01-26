from amg_gym_env import BoomerAMGRelaxEnv
import numpy as np
env = BoomerAMGRelaxEnv()
obs, info = env.reset()
print("reset info:", {k:info[k] for k in info if k in ["r0","m0","m1","m2","m3","rhs_type"]})
for t in range(5):
    a = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(a)
    print(t, "rew", rew, "r", info["r"], "w", info["w"], "s", info["num_sweeps"], "term", term, "trunc", trunc)