# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

### Changed
- Extended the bit-identical `parallel` (rayon) path beyond bare KSG marginal counting to the
  cost-dominating estimators: continuous `I^sx_∩` (`isx_redundancy`, `EhrlichKsg`), the 3-source
  redundancy loop (`redundancy_for_antichain` in `pid3_isx`), and the bootstrap resample loops
  (`block_bootstrap`, `block_bootstrap_paired`, `bootstrap_pid3`). All use an index-ordered
  collect followed by an index-ordered reduction (RNG streams are still drawn serially), so the
  `parallel` feature stays **`f64::to_bits`-identical** to the serial path.

### Added
- **Genuine discrete shared-exclusions PID `i^sx_∩` (`sxpid` module).** New `discrete_sxpid2` /
  `discrete_sxpid3` implement the actual Makkeh–Gutknecht–Wibral (2021, Phys. Rev. E 103, 032149)
  SxPID redundancy — the discrete sibling of the continuous `I^sx_∩` (`isx`/`pid2`/`pid3`), so the
  library now decomposes information with **one** measure across regimes (the discrete path was
  previously only Williams–Beer `I_min`, the measure SxPID was built to replace). Redundancy of an
  antichain `α` is `i^sx_∩(t:α) = log[ P(𝔱 ∩ ⋃_j 𝔞_j) / (P(t)·P(⋃_j 𝔞_j)) ]` (informative
  `−log P(⋃𝔞_j)` minus misinformative `log[P(t)/P(𝔱∩⋃𝔞_j)]`), with `P(⋃𝔞_j)` by inclusion–exclusion
  over collections and standard Möbius inversion on the redundancy lattice (reusing the measure-
  agnostic `discrete_mobius_inversion_3`). Output is **pointwise** (per-realization, signed) *and*
  averaged atoms, each split into informative/misinformative parts. Units **nats**; atoms may be
  negative (never clamped). Exposed to Python as `compute_discrete_sxpid2/3` and the general
  `compute_discrete_sxpid_n` (2–4 sources).
  - **Bit-faithful validation** (`tests/sxpid_reference.rs`): pointwise atom vectors reproduce the
    Abzinger/SxPID reference (`testing/test_gates.py`) for XOR, AND, UNQ, RDN, COPY, PwUnq, SUM, the
    **non-uniform** RndErr gate (probability-weighted averaging, independently re-derived), and a
    **multi-dimensional** source; the averaged values match **IDTxl's own**
    `test_estimators_multivariate_pid.py` to `1e-12` (e.g. `shared(AND)=0.12255624891826572` bits,
    3-source HASH `shared=0.1926450779…`, `pairs=−0.22686079…`, `syn=0.24511249…` bits — ×`ln 2`).
    The informative/misinformative split is pinned at the bottom *and* non-bottom lattice nodes, and
    a realization-keyed check guards the realization↔atom assignment.
  - **General `n`-source path** (`discrete_sxpid_n`, `2 ≤ n ≤ 4`, the count IDTxl's SxPID
    supports): same measure over the full antichain lattice, with a brute-force antichain
    enumeration (the 4-source lattice has the correct **166** nodes) and general Möbius inversion.
    Validated to reproduce `discrete_sxpid2`/`discrete_sxpid3` **bit-for-bit** (1e-12) and to
    satisfy reconstruction + exact source-swap symmetry at 4 sources. Bootstrap CIs for the atoms
    via `bootstrap_discrete_sxpid2`.
  - **Axiom property tests** (`tests/sxpid_axioms.rs`): reconstruction (`Σ_α Π(α)=I(S;T)`),
    self-redundancy, source-swap symmetry, real negativity, and an honest identity-axiom comparison —
    on the two-bit COPY of independent sources `I_min` attributes the maximal **1 bit** of redundancy
    while `i^sx` attributes only `log(4/3)≈0.415` bits (SxPID does **not** force averaged red to 0;
    per Bertschinger et al. the identity axiom is incompatible with global non-negativity).
- **`exp0` `--strict-band` / analytically-grounded `--strict-gate`.** `--strict-gate` no longer
  enforces a verdict on the default high-dimension sweep (whose `PIVOT`/`NO-GO` is the documented,
  expected outcome). It now enforces `GO` (exit code 3 otherwise) only on a **curated band** where
  `GO` is legitimately expected and is checked against a **closed-form analytic ground truth**: a
  grid of jointly-Gaussian systems at `d=1`, `n=4000` (KSG's validated regime) whose three
  measure-independent MI terms `I(S1;T)`, `I(S2;T)`, `I(S1,S2;T)` must match their Cover–Thomas
  Gaussian values within the existing scale-aware tolerance (Barrett-2015 MMI atoms are printed for
  reference only — I^sx ≠ MMI). `--strict-gate` implies `--strict-band`, which runs and reports the
  band without enforcing. The four synthetic scenarios are still run at `d ∈ {2,4,8}` as a
  **non-gating** diagnostic alongside the band; they are a known non-`GO` regime (a reported finding,
  not a regression) and the gate's tolerances are deliberately not loosened to accommodate them.
- **`tests/gaussian_pid_atoms.rs` — cited analytic Gaussian PID-*atom* regression.** The previous
  Gaussian test covered MI only; this adds atom-level ground truth for the continuous `I^sx_∩`
  PID2 estimator. Identical sources (`S1==S2==T+noise`) assert Red ≈ I(X;T) and Unq1≈Unq2≈Syn≈0;
  independent additive sources (`S1⟂S2`, `T=S1+S2+noise`) assert the synergy-dominant regime. The
  measure-independent MI terms come from the closed-form Gaussian-channel MI `I=-½ln(1-ρ²)` (Kraskov
  2004; Cover & Thomas). A separate, clearly-labelled Barrett-2015 Gaussian **MMI** reference
  (`R_MMI=min(I(S1;T),I(S2;T))`) is a sanity comparison only (MMI ≠ I^sx).
- **Correction — `independent_additive` I^sx redundancy is positive, not zero.** An earlier version
  of `tests/gaussian_pid_atoms.rs` *assumed* `Red→0` for independent additive Gaussian sources
  ("derived, not assumed") and labelled the estimator's stable ~0.22 nats as over-attribution bias.
  That assumption was **wrong**. The bin-width→0 limit of the discrete shared-exclusions redundancy
  is `i^sx_∩(t:{1},{2}) → log[w1·e^{i1}+w2·e^{i2}]` (a probability-weighted average of pointwise-MI
  exponentials), which is **strictly positive** for this system. New
  `tests/sxpid_gaussian_oracle.rs` provides the **closed-form numerical oracle** (~0.225 nats) and
  asserts the KSG `I^sx_∩` estimator converges to it; the discrete `i^sx` in the fine-bin limit
  triangulates the same value. The false `Red==0` assertion and the "estimator bias" framing were
  removed from `gaussian_pid_atoms.rs`, `bin/exp0.rs`, and `AGENTS.md`.
- **Analytic discrete-PID ground-truth gates (`discrete_pid.rs` tests).** Two canonical
  Williams & Beer (2010) logic gates are now anchored to their closed-form `I_min` PID atoms at
  machine precision (`tol = 1e-9`), on an *exactly enumerated* input distribution (each of the four
  binary `(S1,S2)` states repeated equally, so the empirical law is exact and there is no sampling
  error): **XOR** is pure synergy (`Red=Unq1=Unq2=0`, `Syn=ln 2`, `I(S_i;T)=0`), and **AND** matches
  the derived `H(T)=¼ln4+¾ln(4/3)`, `I(S_i;T)=H(T)-½ln2`, `Red=I(S_i;T)`, `Unq_i=0`,
  `Syn=H(T)-I(S_i;T)` (all values derived in-comment, not tuned). Both also assert the PID identity
  `Red+Unq1+Unq2+Syn=I(S1,S2;T)` exactly.

### Fixed
- **`discrete_pid3_redundant_sources_dominant` tested the wrong lattice node.** The test read
  `redundancies[6]` and called it "Redundancy", but index 6 (antichain `{{0,1,2}}`) is the lattice
  **TOP**, whose `I_min` is the joint MI `I(S0,S1,S2;T)` — so the old `red > 0.3·I(S0;T)` assertion
  was vacuous (joint MI always exceeds a marginal MI). It now checks the scientifically meaningful
  claims for the near-copy-plus-noise system: the pairwise redundancy of the two near-copies
  (`redundancies[7]`, antichain `{{0},{1}}`) is sizable, the global all-singletons redundancy
  (`redundancies[16]`, diluted by the noise source S2) cannot exceed it, and the TOP node carries
  at least `I(S0;T)`.

- **`pid-runlog` logical trace hash** — `logical_trace_hash` / `logical_trace_hash_from_path`
  digest the ordered event sequence with wall-clock (`timestamp_ns`) fields excluded (the
  run-log filesystem URI/path is never part of an event, so it is excluded by construction).
  Two runs that are logically identical but differ only in timestamps now share the same
  `logical_trace_hash` while their `replay_trace_hash` differs. The hash is surfaced on
  `RunLogSummary` and `RunManifest`, the `pid-runlog-replay` CLI gains `--compare-logical
  <a> <b>` (and prints `logical_trace_hash` in its default report), and a regression test
  (`logical_trace_hash_ignores_timestamps_but_replay_hash_does_not`) pins the contract.
- **`pid-runlog` crash-safe live logging** — `RunLogWriter::sync_all()` / `flush_durable()`
  flush the buffer to the OS and `fsync` the underlying file so already-written events survive a
  crash/power loss.
- **`exp0` build provenance** — a `build_provenance` block (crate version, source git commit or
  `"unknown"`, rustc version, enabled feature set) is added to `exp0`'s run-log `config_json` and
  thereby folded into the SHA-256 `config_hash`, so a run certifies the exact binary that
  produced it. Commit/rustc are captured at compile time via a new `crates/pid-core/build.rs`.
- `tests/parallel_bit_identity.rs` — a serial==parallel bit-identity guard asserting
  `f64::to_bits` equality (against frozen serial reference bit-patterns) for `ksg_local_mi_terms`,
  the 2-/3-source PID atoms and redundancies, the continuous `I^sx_∩` redundancy, and a
  block-bootstrap result; runs in both the default and `--features parallel` configurations.

## [0.2.0] - 2026-06-20

### Added
- **`pid-python`** — Python bindings (PyO3 + maturin) exposing the `pid_core_rs` module: 15
  functions over NumPy arrays (MI, redundancy, co-information, 2-/3-source PID, discrete PID,
  Shannon invariants, geometry diagnostics, PCA/PLS/hash/standardize preprocessing), an abi3
  wheel for Python 3.11+, a `pyproject.toml`, a pytest smoke suite, and a CI `python` job
  (maturin build + import test on Linux and macOS). `extension-module` is an opt-in feature so
  the plain `cargo` workspace still builds/links without libpython. The crate is distributed as a
  Python wheel (via maturin) and is not published to crates.io (`publish = false`).

### Changed
- Repository moved to `github.com/sepahead/pid-rs` (GitHub account rename); all URLs updated.
- Documentation accuracy pass across every README/markdown file: scoped the `unsafe`-forbidden
  claim to `pid-core`/`pid-runlog`, corrected the `exp0`/`--strict-gate` framing (CI runs `exp0`
  without `--strict-gate`, so it does not enforce a `GO`), and aligned the build/test commands
  with CI.

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
    and permutation tests, and the `exp0` estimator-validation harness (a diagnostic
    GO/PIVOT/NO-GO gate that exits 0 by default; PIVOT/NO-GO is expected at high dimensions, and
    the opt-in `--strict-gate` flag exits non-zero unless the verdict is GO).
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

[Unreleased]: https://github.com/sepahead/pid-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sepahead/pid-rs/releases/tag/v0.2.0
[0.1.0]: https://github.com/sepahead/pid-rs/releases/tag/v0.1.0
