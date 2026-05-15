import os
import csv
import argparse
import numpy as np
from amg_gym_env import BoomerAMGRelaxEnv

def eval_w(w_val: float, episodes: int, seed: int,
           grid_min: int, grid_max: int, grid_bias: float,
           cmin: float, cmax: float):

    os.environ["AMG_RELAX_TYPE"] = "18"
    os.environ["AMG_CYCLE_TYPE"] = "1"  # V-cycle

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
        w_scale=0.0,
        max_cycles=20,
        tol=1e-8,
    )

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    times = []
    cycles = []
    for _ in range(episodes):
        obs, info0 = env.reset()
        done = False
        t = 0.0
        c = 0
        while not done:
            obs, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            t += float(info.get("dt", 0.0))
            c = int(info.get("cycle", c + 1))
        times.append(t)
        cycles.append(c)

    env.close()
    times = np.array(times); cycles = np.array(cycles)
    return dict(
        w=w_val, episodes=episodes,
        mean_time=float(times.mean()), median_time=float(np.median(times)),
        mean_cycles=float(cycles.mean()), median_cycles=float(np.median(cycles)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--w_list", default="0.5,0.75,1.0,1.1,1.2,1.25,1.3,1.4,1.5,1.7,2.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid_min", type=int, default=10)
    ap.add_argument("--grid_max", type=int, default=80)
    ap.add_argument("--grid_bias", type=float, default=3.0)
    ap.add_argument("--cmin", type=float, default=1.0)
    ap.add_argument("--cmax", type=float, default=1000.0)
    ap.add_argument("--out_csv", default="w_headroom_relax18_v.csv")
    args = ap.parse_args()

    w_list = [float(x) for x in args.w_list.split(",") if x.strip()]

    rows = []
    for w in w_list:
        res = eval_w(w, args.episodes, args.seed,
                     args.grid_min, args.grid_max, args.grid_bias,
                     args.cmin, args.cmax)
        rows.append(res)
        print(f"w={w:.3f} | mean_time={res['mean_time']:.6f} mean_cycles={res['mean_cycles']:.2f}")

    with open(args.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["w","episodes","mean_time","median_time","mean_cycles","median_cycles"])
        for r in rows:
            wr.writerow([r["w"], r["episodes"], r["mean_time"], r["median_time"], r["mean_cycles"], r["median_cycles"]])

    best = min(rows, key=lambda x: x["mean_time"])
    print("\n=== BEST (by mean time) ===")
    print(best)
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()