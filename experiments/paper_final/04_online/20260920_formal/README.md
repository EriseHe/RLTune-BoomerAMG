# Module 04 — formal six-group design

This is the frozen September 20 single-seed design for both PDE families, all
three grid sizes, and a 50-cycle cap. Its configurations are preserved unchanged.
The later accepted six-seed batches are indexed in the
[reproduction guide](../../../../docs/reproduction.md).

## Frozen protocol

- Group order: diffusion 40³, advection 40³, diffusion 60³, advection 60³,
  diffusion 80³, advection 80³. Groups run serially.
- Each group uses 5000 paired problems and three methods: Default,
  LinUCB V4 with default solve, and LinUCB V4 with recursive LSTDQ V3.
  Method order is randomized per problem with its recorded seed.
- Setup learning starts on problem 1; solve learning starts on problem 1001.
  There are no imported checkpoints or additional training problems.
- Both families use relative residual tolerance `1e-6`, cap 50, and the 18/18/9
  smoother profile. Advection uses the prescribed forward DifConv discretization.
- Setup contexts include the intercept: four features for diffusion and seven
  for advection. Action grids, candidates, exploration and LSTDQ settings are
  pinned by the JSON files.
- Construction failure allows up to three learned attempts, then one default
  fallback. Solve nonconvergence invokes default recovery directly. The
  `rollback_unrecovered` mode rolls provisional observations back if recovery
  also fails. All attempted work and recovery cost remain charged.

## Seeds and evidence

| RNG | Recorded base seed |
|---|---:|
| Problem stream | 56700120 |
| Bandit | 56760120 |
| Controller | 56766120 |
| Method order | 56772120 |

The V3 method retains its controller seed offset of 2018. The eight input seeds
are `56700120 + 6000*k`, for `k=0,...,7`; the stream shuffle seed is 56748120.
The six groups share this seed tuple and are different problem settings, not six
independent training seeds. This seed was examined in preceding development
work; the design is not an untouched holdout.

The [exact input captures](../../reproduction/online/streams/README.md) preserve
matrix arguments, RHS seeds, order and normalized contexts under the original
stream hashes. Replay avoids platform rounding differences in logarithms.
The original sampler remains available for other configurations.

## Validate and run

From the repository root in the active environment, validate without solving:

```sh
python -m experiments.paper_final.run_04_online --suite experiments/paper_final/04_online/20260920_formal/suite.json --validate-only
```

Run into a fresh directory and then analyze the completed measurements:

```sh
python -m experiments.paper_final.run_04_online --suite experiments/paper_final/04_online/20260920_formal/suite.json --output-root results/paper_final/04_online/formal_retiming --run
python -m experiments.paper_final.analyze_04_online --suite experiments/paper_final/04_online/20260920_formal/suite.json --output-root results/paper_final/04_online/formal_retiming
```

The optional `run.command` invokes the same suite with one compute thread.
It uses the active `python`; `PYTHON_BIN` and `OUTPUT_ROOT` can override its
interpreter and output directory. It has no host-specific power checks or paths.

The suite records source/configuration/native provenance, audits completed
groups, and writes combined `analysis/module04.json` and `.md` reports.
Group folders contain trajectories, summaries, plots and final checkpoints.
The runner refuses changed provenance or automatic restart of an incomplete group.

## Cost accounting

Report all 5000 attempted problems, the final 1000, and final failure counts.
Native cost includes attempted setup, solve and recovery work. Total cost adds
recurring learner/controller work. One-time initialization, candidate preparation,
matrix/RHS assembly and output I/O are outside these component timers;
subprocess elapsed time is recorded separately. When unrecovered failures remain,
attempted cost does not establish that every input was successfully solved.

Fresh runs record current binaries, packages and source. They preserve the
protocol and inputs but do not replace accepted September measurements or claim
identical wall-clock times. The complete suite is a substantial native run.
