# Accepted Module 05 frozen inputs

`protocol.json` and `summary.json` preserve the accepted September 29 Run 05
protocol and compact results byte for byte. Their hashes are recorded in the
parent [reproduction manifest](../manifest.json).

`frozen_inputs/` is a self-contained input bundle for retiming those same frozen
choices with the current submission source. Its `manifest.json` verifies all
included files. The approximately 14 MB bundle contains:

- six final LSTDQ checkpoint files, their encoders, and experiment configurations;
- exact accepted protocol, input, job, preflight, and selection files;
- the original prepared-input hashes and execution-source manifest;
- the accepted environment and three smoothing-theory/audit files;
- 29,520 compact fixed-grid selection records in six JSONL gzip files.

The runtime uses the saved hierarchy tuples in the jobs; historical setup-bandit
checkpoint files and earlier result directories are unnecessary. All included
checkpoint and input bytes match the hashes in `accepted_prepared.json`.

The compact records retain exactly the ten fields listed in
`fixed_scan_evidence.json`, including the recorded native continuation time,
success, repetition, and input/hierarchy identities. They preserve every row and
its order, and JSON numbers round-trip the original floating-point values.
Cycle traces and unused raw fields are omitted. The evidence metadata records
both original raw scan hashes and distinct derived-file hashes. The original
hashes describe source provenance; they are not hashes of the compact files.

```sh
python -c 'from experiments.paper_final.common.frozen_inputs import DEFAULT_BUNDLE, verify_bundle; print(verify_bundle(DEFAULT_BUNDLE))'
```

Verification checks accepted input hashes and reconstructs the stream-wide and
per-instance fixed selections from successful grid costs. The bundle contains
inputs for new measurements, not the original measurements themselves. The
[official Run 05 guide](../../05_policy/RUN05.md) gives preparation, execution,
rendering and packaging commands.
