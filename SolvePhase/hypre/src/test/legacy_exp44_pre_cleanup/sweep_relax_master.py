import csv
import subprocess
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_per_setting", type=int, default=60)
    ap.add_argument("--grid_min", type=int, default=10)
    ap.add_argument("--grid_max", type=int, default=80)
    ap.add_argument("--use_grid_bias", action="store_true")
    ap.add_argument("--grid_bias", type=float, default=3.0)
    ap.add_argument("--cmin", type=float, default=1.0)
    ap.add_argument("--cmax", type=float, default=1000.0)
    ap.add_argument("--w_list", default="1.0,1.25")
    ap.add_argument("--out_csv", default="relax_headroom.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w_list = [float(x) for x in args.w_list.split(",") if x.strip()]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relax_type", "w", "episodes",
                    "mean_time", "median_time",
                    "mean_cycles", "median_cycles"])

    for relax_type in range(1, 19):
        for w_val in w_list:
            tmp = f"tmp_relax_{relax_type}_w_{w_val}.csv"
            cmd = [
                "python", "sweep_relax_worker.py",
                "--relax_type", str(relax_type),
                "--w", str(w_val),
                "--episodes", str(args.episodes_per_setting),
                "--grid_min", str(args.grid_min),
                "--grid_max", str(args.grid_max),
                "--grid_bias", str(args.grid_bias),
                "--cmin", str(args.cmin),
                "--cmax", str(args.cmax),
                "--seed", str(args.seed),
                "--out_csv", tmp,
            ]
            if args.use_grid_bias:
                cmd.append("--use_grid_bias")

            # run worker in fresh process
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError:
                print(f"[WARN] relax={relax_type}, w={w_val}: worker crashed; skipping")
                continue

            # append worker result
            with open(tmp, "r") as f:
                r = list(csv.reader(f))
                if len(r) >= 2:
                    row = r[1]
                    with open(args.out_csv, "a", newline="") as out:
                        csv.writer(out).writerow(row)
            try:
                os.remove(tmp)
            except OSError:
                pass

    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()