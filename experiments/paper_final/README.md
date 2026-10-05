# Official paper experiments

This directory contains the two studies used in the SISC submission: Module 04
online autotuning and accepted Module 05 matched-hierarchy Run 05.

| Study | Entry point | Protocol |
|---|---|---|
| Module 04: Default, LinUCB, and LinUCB–LSTDQ over 5000 problems | `python -m experiments.paper_final.run_04_online` | [Frozen design](04_online/20260920_formal/README.md), [captured six-seed configurations](reproduction/online/six_seeds.json) |
| Module 05: frozen policies on matched hierarchies | `python -m experiments.paper_final.run_05_policy_minimax` | [Accepted Run 05](05_policy/RUN05.md), [captured protocol](reproduction/matched_policy/protocol.json) |

Start with the [reproduction guide](../../docs/reproduction.md) for commands,
required external artifacts, environment setup, and timing definitions.
The [reproduction manifest](reproduction/manifest.json) preserves the hashes of
captured configurations and compact accepted evidence.

`common/` contains internal helpers for durable artifact I/O, frozen evaluation,
matched cost aggregation, worker supervision, and figure presentation. Numerical
solver and learner implementations remain in their own packages.

Historical protocol variants, activation studies, earlier matched-policy runs,
and the separate solve-specific checkpoint and complete-method studies live in
[the development archive](../archive/paper_development/README.md). They are outside
the current submission. Existing measurements, source snapshots, and checkpoint
formats remain unchanged.
