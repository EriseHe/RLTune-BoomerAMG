"""Repeat Module 04 seeds 4/5/6, preserving originals, then run the queued Module 05."""
from __future__ import annotations

from experiments.runtime import (
    configure_single_thread, prevent_sleep, stop_sleep_prevention, single_thread_environment,
)

if __name__ == "__main__":
    configure_single_thread()

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/paper_final/04_online/20260926_macmini_module04_seeds4to6"
DEFAULT = ROOT / "results/paper_final/04_online/20260927_diffusion60_repeat_seeds4to6"


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n"); tmp.replace(p)


def prepare(output):
    if output.exists() and any(output.iterdir()): raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    original = read(OLD / "provenance/source_manifest.json")["source"]["files"]
    differences = [name for name, digest in original.items() if sha(ROOT/name) != digest]
    if differences: raise RuntimeError(f"Original execution sources changed: {differences}")
    from experiments.paper_final.run_05_policy import environment_check
    environment_check(output)
    configs = {}
    for seed in (4, 5, 6):
        name = f"diffusion_60_s{seed}"
        source = OLD / "protocol" / f"{name}.json"
        target = output / "protocol" / source.name
        target.parent.mkdir(exist_ok=True); shutil.copy2(source, target)
        suite = read(OLD / "protocol" / f"suite_seed_{seed}.json")
        suite.update(name=f"repeat_diffusion60_seed_{seed}", families=["diffusion"], grids=[60],
                     runs=[r for r in suite["runs"] if r["name"] == name],
                     purpose="Exact-input independent retraining repeat; original results preserved")
        write(output / "protocol" / f"suite_seed_{seed}.json", suite)
        assert sha(target) == suite["runs"][0]["config_sha256"]
        configs[str(seed)] = {"config_sha256": sha(target), "stream_sha256": suite["runs"][0]["stream_sha256"],
                              "original_directory": str(OLD / f"seed_{seed}" / name)}
    write(output / "repeat_protocol.json", {"created_at": now(), "seeds": [4, 5, 6], "cases": 5000,
        "workers": 3, "threads_per_worker": 1, "configs": configs, "original_source_files": original,
        "comparison": "Native setup plus solve first; all 5000 and last 1000; all attempts and recovery retained",
        "replacement_rule": "Preferred-results manifest points to new run iff audited full-5000 no-overhead paired reduction is higher; originals retained and the selection is disclosed",
        "module05_checkpoints": "New seed 4, original seeds 1,2,3,5,6; hierarchy choice recorded in next_stage.json",
        "caution": "This is a timing-sensitivity repeat, not three new independent training seeds. Better timing alone does not identify machine interference."})
    write(output / "status.json", {"status": "prepared", "phase": "module04", "at": now()})
    (output / "README.md").write_text("# Diffusion 60³ repeat: seeds 4, 5, 6\n\nOriginal configurations, random seeds, streams, and execution code; fresh learners.\nOriginal results remain intact. `preferred_results.json` records any outcome-selected\nreplacement and `comparison.json` retains both results. `next_stage.json` records\nthe authorized Module 05 follow-up. `status.json` and `supervisor.log` track progress.\n")


def compare(output):
    rows = []
    for seed in (4, 5, 6):
        name = f"diffusion_60_s{seed}"
        row = {"seed": seed}
        for version, root in (("original", OLD), ("repeat", output)):
            directory = root / f"seed_{seed}" / name
            data = {}
            for method in ("default_setup_default_solve", "bandit_lstdq"):
                with (directory / "trajectories" / f"{method}.jsonl").open() as stream:
                    records = [json.loads(line) for line in stream]
                records.sort(key=lambda r:r["online_index"])
                if [r["online_index"] for r in records] != list(range(5000)): raise ValueError("Incomplete comparison")
                data[method] = records
            for a,b in zip(*data.values()):
                if a["mkw"] != b["mkw"]: raise ValueError("Unpaired matrix inputs")
            values = {"directory": str(directory)}
            for size in (5000,1000):
                totals = {m:sum(r["outcome"]["native_runtime"] for r in rr[-size:]) for m,rr in data.items()}
                values[f"reduction_{size}_pct"] = 100*(1-totals["bandit_lstdq"]/totals["default_setup_default_solve"])
                values[f"native_totals_{size}_sec"] = totals
            values["final_failures"] = {m:sum(bool(r["outcome"]["unrecovered_failure"]) for r in rr) for m,rr in data.items()}
            row[version] = values
        row["change_5000_pp"] = row["repeat"]["reduction_5000_pct"]-row["original"]["reduction_5000_pct"]
        row["change_last1000_pp"] = row["repeat"]["reduction_1000_pct"]-row["original"]["reduction_1000_pct"]
        eligible = not any(row["repeat"]["final_failures"].values())
        row["preferred_version"] = "repeat" if eligible and row["change_5000_pp"] > 0 else "original"
        row["preferred_directory"] = row[row["preferred_version"]]["directory"]
        rows.append(row)
    write(output / "comparison.json", {"created_at": now(), "metric": "No-overhead setup plus solve versus paired Default", "seeds": rows})
    write(output / "preferred_results.json", {"selection": "Outcome-selected; does not replace the original prespecified six-seed evidence", "seeds": {str(r["seed"]):r["preferred_directory"] for r in rows}})
    return rows


def supervise(output):
    with (output/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(output/"repeat_protocol.json")
        for name,digest in protocol["original_source_files"].items():
            if sha(ROOT/name) != digest: raise RuntimeError(f"Original execution source changed: {name}")
        start = time.monotonic(); children=[]; logs=[]
        state={"status":"running","phase":"module04","started_at":now(),"at":now(),"pid":os.getpid()}
        awake = prevent_sleep()
        state["awake_pid"]=(awake.pid if awake is not None else None)
        def stop(signum,frame): raise KeyboardInterrupt(f"Signal {signum}")
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        try:
            for seed in (4,5,6):
                dest=output/f"seed_{seed}"; dest.mkdir(exist_ok=True)
                handle=(dest/"supervisor.log").open("a",buffering=1); logs.append(handle)
                command=[sys.executable,"-u","-m","experiments.paper_final.run_04_online","--run","--suite",str(output/"protocol"/f"suite_seed_{seed}.json"),"--output-root",str(dest)]
                child=subprocess.Popen(command,cwd=ROOT,env=dict(single_thread_environment(),MPLBACKEND="Agg",PYTHONUNBUFFERED="1"),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
            state["worker_pids"]=[p.pid for p in children]
            print(f"[{now()}] Started Module 04 seeds 4,5,6, one thread each",flush=True)
            last_print=0
            while True:
                codes=[c.poll() for c in children]
                if any(c not in (None,0) for c in codes): raise RuntimeError(f"Module 04 worker exited: {codes}")
                progress=[]
                for seed in (4,5,6):
                    p=output/f"seed_{seed}"/f"diffusion_60_s{seed}"/"progress.json"
                    try: n=read(p)["completed_online_instances"] if p.exists() else 0
                    except json.JSONDecodeError: n=0
                    progress.append({"seed":seed,"done":n,"total":5000})
                elapsed=time.monotonic()-start; done=sum(p["done"] for p in progress)
                remaining=max((elapsed*(5000-p["done"])/p["done"] for p in progress if p["done"]),default=None)
                state.update(at=now(),elapsed_sec=elapsed,progress=progress,done=done,total=15000,estimated_module04_remaining_sec=remaining)
                write(output/"status.json",state)
                if time.monotonic()-last_print>60:
                    print(f"[{now()}] Module 04: {done}/15000 paired problems; per seed {progress}",flush=True);last_print=time.monotonic()
                if all(c==0 for c in codes): break
                time.sleep(5)
            comparison=compare(output)
            write(output/"module04_complete.json",{"at":now(),"elapsed_sec":time.monotonic()-start,"comparison":comparison})
            print(f"[{now()}] Module 04 complete; paired comparison saved",flush=True)
            queued=read(output/"next_stage.json")
            state.update(phase="module05",at=now(),module05_output=queued["output"])
            write(output/"status.json",state)
            handle=(output/"module05.log").open("a",buffering=1);logs.append(handle)
            child=subprocess.Popen(queued["command"],cwd=ROOT,env=dict(single_thread_environment(),MPLBACKEND="Agg",PYTHONUNBUFFERED="1"),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child); state["module05_pid"]=child.pid;write(output/"status.json",state)
            while child.poll() is None:
                p=Path(queued["output"])/"status.json"
                if p.exists():
                    try: state["module05_progress"]=read(p)
                    except json.JSONDecodeError: pass
                state.update(at=now(),elapsed_sec=time.monotonic()-start);write(output/"status.json",state)
                time.sleep(5)
            if child.returncode: raise RuntimeError(f"Module 05 failed with exit {child.returncode}; see module05.log")
            state.update(status="complete",phase="complete",at=now(),elapsed_sec=time.monotonic()-start)
            write(output/"complete.json",state);write(output/"status.json",state)
            print(f"[{now()}] Both stages complete",flush=True)
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",at=now(),error=str(exc));write(output/"status.json",state)
            raise
        finally:
            for child in children:
                if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
            for handle in logs: handle.close()
            stop_sleep_prevention(awake)


def watch(output):
    while True:
        state=read(output/"status.json")
        print("\033[2J\033[H",end="")
        print("Diffusion 60³ repeat · Module 04 seeds 4,5,6 → Module 05",flush=True)
        print(f"{now()}  {state['status'].upper()} · {state['phase']}",flush=True)
        for p in state.get("progress",[]): print(f"Seed {p['seed']}: {p['done']:,} / 5,000 paired problems",flush=True)
        if state.get("module05_progress"): print(json.dumps(state["module05_progress"],indent=2),flush=True)
        remaining=state.get("estimated_module04_remaining_sec")
        if state['phase']=='module04' and remaining: print(f"Module 04 estimate: {remaining/60:.0f} minutes remaining; Module 05 follows.",flush=True)
        if state['status'] in ('complete','failed','interrupted'):
            if state.get('error'):print(state['error'],flush=True)
            return
        time.sleep(10)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","watch"))
    parser.add_argument("--output",type=Path,default=DEFAULT)
    args=parser.parse_args()
    {"prepare":prepare,"run":supervise,"watch":watch}[args.command](args.output.resolve())
