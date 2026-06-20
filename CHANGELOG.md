# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

### Added
- **`pid-python`** — Python bindings (PyO3 + maturin) exposing the `pid_core_rs` module: 15
  functions over NumPy arrays (MI, redundancy, co-information, 2-/3-source PID, discrete PID,
  Shannon invariants, geometry diagnostics, PCA/PLS/hash/standardize preprocessing), an abi3
  wheel for Python 3.11+, a `pyproject.toml`, a pytest smoke suite, and a CI `python` job
  (maturin build + import test on Linux and macOS). `extension-module` is an opt-in feature so
  the plain `cargo` workspace still builds/links without libpython. The crate is distributed as a Python wheel (via maturin) and is not published to crates.io (`publish = false`).

## [0.1.0] - 2026-06-17

Initial public release.

### Added

- **`pid-core`** — continuous and discrete information-decomposition estimators:
  - KSG mutual information (Kraskov et al. 2004), L∞ joint metric, strict-radius marginal
    counting, optional bit-identical `parallel` (rayon) path.
  - Continuous shared-exclusions redundancy `I^sx_∩` (Ehrlich et al. 2024), disjunction
    neighbourhoods.
  - 2- and 3-source PID atoms (`pid2_isx`, `pid3_isx`) whose Möbius identities hold by
    construction; discrete `I_min` PID over the full 18-antichain lattice.
  - Shannon invariants: co-information, O-information, average degrees of redundancy/vulnerability.
  - Geometry diagnostics (intrinsic dimension, distance concentration, Gromov hyperbolicity),
    preprocessing (standardisation, PCA, PLS, hash projection, seeded jitter), block bootstrap
    and permutation tests, and the `exp0` estimator-validation harness (a diagnostic GO/PIVOT/NO-GO gate; PIVOT/NO-GO is expected at high dimensions, and `--strict-gate` enforces GO in CI).
- **`pid-runlog`** — versioned, content-addressed run-log schema (per-record SHA-256 payload
  digests, a whole-trace replay hash, and a whole-file SHA-256 manifest; records are not
  prev-hash-chained) with a `pid-runlog-replay` validation CLI.
- Worked example (`cargo run --example ksg_and_pid`), CI (fmt / clippy `-D warnings` / tests /
  docs / MSRV / smoke), and an analytic-reference test suite (Gaussian-channel MI, XOR/COPY PID
  structure, PID identities to `1e-10`).

### Notes

This release incorporates fixes from an internal soundness audit: the default 2-source/
co-information paths no longer clamp MI terms before the algebraic identities; discrete-PID and
Shannon-invariant summation is now order-deterministic (`BTreeMap`); the permutation p-value uses
the add-one correction; and the public pipeline bootstrap/permutation helpers (`bootstrap_pid3`,
`permutation_pid3`, `bootstrap_rows_stats`, `permutation_rows_pvalue`) return `Err` instead of
panicking on invalid configuration (the lower-level `block_bootstrap`/`block_bootstrap_paired` keep
their documented `assert`-on-invalid-config contract). See
[Known limitations](README.md#known-limitations) for the tracked follow-ups.

[Unreleased]: https://github.com/sepehrmn/pid-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sepehrmn/pid-rs/releases/tag/v0.1.0
