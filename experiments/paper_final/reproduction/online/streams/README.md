# Exact Module 04 inputs

These 36 compressed JSONL files contain the 180,000 distinct input rows used by
the 42 recorded Module 04 configurations. Each row contains the matrix assembly
arguments (`mkw`) and the eight-field PDE context supplied to the learners.

The files were generated from the recorded sampling protocols on the reference
macOS arm64 host with NumPy 2.2.6. They are input captures, not copies of measured
solver trajectories or regenerated experimental results. Every uncompressed
file has the exact SHA-256 already recorded in its accepted configuration.
The host, capture time, applicable configurations, full sampling settings,
compressed file hashes, and historical stream manifests appear in
`manifest.json`.

The context normalization uses `log` and `log1p`. Different math libraries and
NumPy CPU implementations can round these operations differently. Replaying
these verified inputs preserves the original coefficient samples, right-hand-side
seeds, ordering, context values, and complete stream hashes across platforms.
The original sampler remains in the source and is used for configurations whose
stream hash is unpinned or is not among these accepted captures.

For a recognized hash, the loader requires the original sampling settings and
checks both the compressed file SHA-256 and the uncompressed canonical input
SHA-256. It also checks the row count, finite context fields, and exact JSON
round-trip. Changing sampling settings while retaining an old accepted hash is
an error. Numerical solver results and wall-clock measurements can still depend
on the compiler, hardware, and numerical libraries used for a new run.
