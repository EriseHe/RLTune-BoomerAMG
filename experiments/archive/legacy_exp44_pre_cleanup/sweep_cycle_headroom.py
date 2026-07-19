import os
import csv
import argparse
import numpy as np
from amg_gym_env import BoomerAMGRelaxEnv

def run_setting(cycle_type: int, w_val: float, episodes: int,
                grid_min: int, grid_max: int, grid_bias: float,
                cmin: float, cmax: float, seed: int):

    os.environ["AMG_RELAX_TYPE"] = "18"
    os.environ["AMG_CYCLE_TYPE"] = str(cycle_type)  # 1=V, 2=W

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
        use_grid_bias=True,
        grid_bias=grid_bias,
        randomize_b=True,
        fixed_rhs_type=1,
        sweeps_min=1,
        sweeps_max=1,
        w_center=w_val,
        w_scale=0.0,     # fixed w
        max_cycles=20,
        tol=1e-8,
    )

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    wall_times, cycles_list = [], []

    for _ in range(episodes):
        obs, info0 = env.reset()
        done = False
        ep_t = 0.0
        ep_cycles = 0
        while not done:
            obs, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            # Your C dt_out is already wall time around BoomerAMGSolve
            ep_t += float(info.get("dt", 0.0))
            ep_cycles = int(info.get("cycle", ep_cycles + 1))
        wall_times.append(ep_t)
        cycles_list.append(ep_cycles)

    env.close()

    wall = np.array(wall_times)
    cyc = np.array(cycles_list)
    return dict(
        cycle_type=cycle_type, w=w_val, episodes=episodes,
        mean_time=float(wall.mean()), median_time=float(np.median(wall)),
        mean_cycles=float(cyc.mean()), median_cycles=float(np.median(cyc)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--grid_min", type=int, default=10)
    ap.add_argument("--grid_max", type=int, default=80)
    ap.add_argument("--grid_bias", type=float, default=3.0)
    ap.add_argument("--cmin", type=float, default=1.0)
    ap.add_argument("--cmax", type=float, default=1000.0)
    ap.add_argument("--w_list", default="1.0,1.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default="cycle_headroom.csv")
    args = ap.parse_args()

    w_list = [float(x) for x in args.w_list.split(",") if x.strip()]

    rows = []
    for ct in (1, 2):  # 1=V, 2=W
        for w in w_list:
            res = run_setting(ct, w, args.episodes,
                              args.grid_min, args.grid_max, args.grid_bias,
                              args.cmin, args.cmax, args.seed)
            rows.append(res)
            print(f"cycle_type={ct} w={w:.3f} | mean_time={res['mean_time']:.6f} mean_cycles={res['mean_cycles']:.2f}")

    with open(args.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cycle_type","w","episodes","mean_time","median_time","mean_cycles","median_cycles"])
        for r in rows:
            wr.writerow([r["cycle_type"], r["w"], r["episodes"],
                         r["mean_time"], r["median_time"],
                         r["mean_cycles"], r["median_cycles"]])

    best = min(rows, key=lambda x: x["mean_time"])
    print("\n=== BEST (by mean time) ===")
    print(best)
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()