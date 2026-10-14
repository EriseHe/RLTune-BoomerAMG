# Experiments

Run commands from the repository root after installing the package and building
HYPRE. Use fresh output directories. Each online suite uses three single-thread
workers, one per seed, with four independent cold methods from case 1.

## Modules 1 and 3

For each JSON file in `data/module1/suites/` or `data/module3/suites/`, prepare and
run it sequentially. For example:

```sh
python -m experiments.paper_final.run_online prepare --suite data/module1/suites/diffusion_60_batch1.json --output results/module1/diffusion_60_batch1
python -m experiments.paper_final.run_online run --output results/module1/diffusion_60_batch1
```

Each seed has 5,000 paired inputs. Methods are Default, fixed-setup LSTDQ,
setup-only LinUCB, and joint LinUCB–LSTDQ. The configurations specify β=2,
1,024 setup candidates, aggressive levels 0–2 with interpolation 4/6/7, cycle
weights 1–3, tolerance 10⁻⁶, and a 50-cycle cap. Both paths apply the same
cap-first stopping rule. Native time includes all setup/solve attempts, retries,
and fallbacks. Inclusive time also includes learner work. Assembly and file I/O
are excluded. Module 3 retains failed cases and includes common-success comparisons.

## Module 2

The six supplied checkpoints come from Module 1 diffusion 60³.

```sh
python -m experiments.paper_final.run_policy prepare --config data/module2/configuration.json --output results/module2
python -m experiments.paper_final.run_policy run --output results/module2
```

Scan all 41 weights on 100 independent cases, freeze global/per-case choices,
then independently evaluate five policy roles with three repetitions. Identical
execution cells are shared. Continuation costs exclude the common primary setup
and include recovery costs; inclusive costs also include controller work.
