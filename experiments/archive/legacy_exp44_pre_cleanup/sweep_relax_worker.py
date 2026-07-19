import os
import csv
import argparse
import numpy as np

from amg_gym_env import BoomerAMGRelaxEnv


def run(relax_type: int, w_val: float, episodes: int,
        grid_min: int, grid_max: int, use_grid_bias: bool, grid_bias: float,
        cmin: float, cmax: float, seed: int, out_csv: str):
    # Ensure relax type is applied at solver creation time
    os.environ["AMG_RELAX_TYPE"] = str(relax_type)

    env = BoomerAMGRelaxEnv(
        lib_path="./libamg_env.dylib",
        seed=seed,
        fixed_stencil=0,
        randomize_A=True,
        difconv_c_range=(cmin, cmax),
        difconv_a=(0.0, 0.0, 0.0),
        difconv_atype=0,
        randomize_grid=True,
        grid_min=grid_min,
        grid_max=grid_max,
        use_grid_bias=use_grid_bias,
        grid_bias=grid_bias,
        randomize_b=True,
        fixed_rhs_type=1,
        sweeps_min=1,
        sweeps_max=1,
        w_center=w_val,
        w_scale=0.0,
        max_cycles=20,
        tol=1e-8,
    )

    rows = []
    for ep in range(episodes):
        obs, info0 = env.reset()
        done = False
        ep_wall = 0.0
        ep_cycles = 0

        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        while not done:
            obs, rew, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            # Prefer dt_wall if present, else fallback to dt
            if "dt_wall" in info:
                ep_wall += float(info["dt_wall"])
            else:
                ep_wall += float(info.get("dt", 0.0))
            ep_cycles = int(info.get("cycle", ep_cycles + 1))

        rows.append((ep_wall, ep_cycles))

    env.close()

    wall = np.array([r[0] for r in rows], dtype=float)
    cyc = np.array([r[1] for r in rows], dtype=float)
    mean_time = float(np.mean(wall))
    med_time = float(np.median(wall))
    mean_cycles = float(np.mean(cyc))
    med_cycles = float(np.median(cyc))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relax_type", "w", "episodes",
                    "mean_time", "median_time",
                    "mean_cycles", "median_cycles"])
        w.writerow([relax_type, w_val, episodes,
                    mean_time, med_time,
                    mean_cycles, med_cycles])

    print(f"relax={relax_type:2d} w={w_val:.3f} | mean_time={mean_time:.6f} median={med_time:.6f} mean_cycles={mean_cycles:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--relax_type", type=int, required=True)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--grid_min", type=int, default=10)
    ap.add_argument("--grid_max", type=int, default=80)
    ap.add_argument("--use_grid_bias", action="store_true")
    ap.add_argument("--grid_bias", type=float, default=3.0)
    ap.add_argument("--cmin", type=float, default=1.0)
    ap.add_argument("--cmax", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default="worker_out.csv")
    args = ap.parse_args()

    run(args.relax_type, args.w, args.episodes,
        args.grid_min, args.grid_max, args.use_grid_bias, args.grid_bias,
        args.cmin, args.cmax, args.seed, args.out_csv)