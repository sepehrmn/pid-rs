<h1 align="center">pid-rs</h1>

<p align="center">
  <strong>Partial Information Decomposition &amp; continuous mutual-information estimators in safe Rust.</strong>
</p>

<p align="center">
  <a href="https://github.com/sepahead/pid-rs/actions/workflows/ci.yml"><img src="https://github.com/sepahead/pid-rs/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License: MIT OR Apache-2.0"></a>
  <img src="https://img.shields.io/badge/rustc-1.80%2B-orange.svg" alt="MSRV 1.80">
  <img src="https://img.shields.io/badge/unsafe-forbidden-success.svg" alt="unsafe forbidden">
</p>

---

**pid-rs** implements the **shared-exclusions partial information decomposition** (`I^sx_∩`;
Makkeh–Gutknecht–Wibral 2021) and the continuous *k*-nearest-neighbour estimators it builds on —
KSG mutual information (Kraskov et al. 2004) and the continuous `I^sx_∩` estimator (Ehrlich et al.
2024) — together with a Shannon-invariant screening layer, discrete `I_min` PID, geometry
diagnostics, and dependence-aware uncertainty quantification.

It was built to diagnose how information from different sources (e.g. **vision** and **language**)
is integrated in multimodal policies, but every estimator here is **domain-agnostic**: give it
samples of sources `S1, S2, …` and a target `T` and it estimates how much of the information about
`T` is **redundant**, **unique**, or **synergistic**.

```text
                I(S1,S2; T)
              ┌──────┴───────┐
   Redundancy ·  Unique(S1) ·  Unique(S2) ·  Synergy
```

## Highlights

- **KSG mutual information** for continuous variables — L∞ joint metric, strict-radius marginal
  counting, digamma reference table.
- **Continuous `I^sx_∩`** (shared-exclusions redundancy) via the Ehrlich et al. 2024
  disjunction-neighbourhood kNN estimator — *not* a min-of-pointwise heuristic.
- **2- and 3-source PID atoms** whose Möbius identities (`Red + Unq₁ + Unq₂ + Syn = I(S1,S2;T)`)
  **hold by construction** and are asserted in tests.
- **Discrete `I_min` PID** over the full 18-antichain 3-source lattice (Williams & Beer 2010).
- **Shannon invariants** — co-information, O-information, and the average degrees of redundancy
  (`r̄`) and vulnerability (`v̄`) (Gutknecht et al. 2025) — as cheap screening statistics.
- **Geometry diagnostics** — intrinsic dimension (Levina–Bickel), distance concentration, Gromov
  hyperbolicity — to decide whether a continuous-kNN regime is even valid.
- **Preprocessing** — standardisation, PCA, hash projection, seeded jitter, and PLS.
- **Honest uncertainty** — block bootstrap and permutation tests that respect sample dependence.
- **Reproducible by construction** — content-addressed run-logs ([`pid-runlog`](crates/pid-runlog);
  per-record SHA-256 payloads + a whole-trace replay hash + a whole-file manifest), seeded RNG, and
  an optional `parallel` feature whose results are **bit-identical** to the serial path.
- The estimator core (`pid-core`) is `#![forbid(unsafe_code)]`, returns errors rather than
  `panic!`-ing on valid-but-degenerate input, and keeps a dependency-light tree. (`pid-runlog`
  is also unsafe-free; `pid-python` necessarily uses PyO3's `unsafe` internals.)

## Project status

`pid-rs` is at `0.1.0`. The estimator **core** is validated against analytic ground truth (see
[Validation](#validation)); the surrounding statistics, performance, and tooling layers are usable
but have tracked follow-ups. This section is a quick honest map of where things stand — it does not
repeat the per-claim detail in [Conventions](#conventions),
[Scientific cautions](#-scientific-cautions-read-before-trusting-results), or
[Known limitations](#known-limitations).

### What works today

| Capability | Notes |
|---|---|
| **KSG mutual information** | Continuous variables, L∞ joint metric, strict-radius marginal counting; checked vs the closed-form Gaussian-channel MI. |
| **Continuous `I^sx_∩`** | Ehrlich et al. 2024 disjunction-neighbourhood kNN redundancy (`IsxMethod::EhrlichKsg`); checked against a fixed-data reference. |
| **2- & 3-source PID atoms** | `pid2_isx` / `pid3_isx`; Möbius identities (`Red + Unq₁ + Unq₂ + Syn = I(S1,S2;T)`) hold by construction and are asserted in tests within `1e-10`. |
| **Discrete `I_min` PID** | `discrete_pid2` / `discrete_pid3` over the full 18-antichain 3-source lattice (Williams & Beer 2010), with equal-width quantization. |
| **Shannon invariants** | Co-information, O-information, `r̄`, `v̄` (Gutknecht et al. 2025) as cheap screening statistics. |
| **Geometry diagnostics** | Intrinsic dimension (Levina–Bickel), distance concentration, Gromov hyperbolicity — to decide whether a continuous-kNN regime is even valid. |
| **Preprocessing / PLS** | Standardisation, PCA, hash (CountSketch) projection, seeded jitter, and supervised PLS with CV component selection. |
| **Uncertainty quantification** | Moving-block bootstrap and permutation tests that respect sample dependence. |
| **Run-logs** | `pid-runlog`: versioned, content-addressed JSONL schema with per-record payload hashes, a whole-trace replay hash, and replay/validate/compare/sidecar CLIs. |
| **Python bindings** | `pid_core_rs` (PyO3 + maturin, `abi3` ≥ CPython 3.11) — 15 functions over C-contiguous `float64` NumPy arrays. Lives on `main` (post-`0.1.0`-tag). |
| **Reproducibility** | Seeded RNG; the optional `parallel` feature is **bit-identical** to the serial path; `#![forbid(unsafe_code)]`; errors (not panics) on degenerate input. |

### What needs further work

- **kNN is brute-force `O(n²)`.** `kth_neighbor_distance_*` and `count_neighbors_within` scan all
  pairs per query — there is no kd-tree / approximate-NN backend, so large `n` is slow.
- **No multiple-comparison correction.** Many atoms × sources × windows report raw per-atom
  p-values; apply your own FDR/FWER control.
- **`runlog --validate` is per-record, not whole-trace integrity.** It checks per-event invariants
  (payload/config-hash matches, monotone timestamps/steps, single `run_started`/`run_ended`, bridge
  causality, finite values). Whole-trace integrity is a separate path: the order-sensitive
  `replay_trace_hash` (`--compare`) and `--verify-sidecars`.
- **`exp0` is a diagnostic gate, not a pass/fail build step.** It emits a GO/PIVOT/NO-GO verdict
  from monotonicity / invariant / geometry counters and **exits 0 by default** (its default sweep
  deliberately enters regimes where kNN MI is known to break down). Use `--strict-gate` to make a
  non-`GO` verdict exit non-zero.
- **No crates.io release yet.** Depend on the Git repository; the Python crate is `publish = false`
  by design (shipped as a wheel via maturin).
- **External cross-validation pending.** Discrete-PID values are checked against an independent
  in-repo re-derivation and canonical-gate structure; an external `csxpid` cross-check is planned.

### Caveats

- **kNN failure modes are real.** Estimators assume **i.i.d.** samples (trajectory autocorrelation
  biases them — subsample or block-bootstrap); high ambient/intrinsic dimension causes **distance
  concentration** that degrades kNN geometry; and **strong (near-deterministic) dependence** can
  require prohibitive sample sizes (Gao et al. 2015). Run the geometry diagnostics and the `exp0`
  gate before interpreting results.
- **Negative atoms are real, not bugs.** `I^sx_∩` trades all-atom non-negativity for the target
  chain rule, so atoms (including redundancy) can be negative; the library never silently clamps
  them (`NegativeHandling` is an opt-in reporting choice).
- **Cross-estimator PID2 mixing.** In `pid2_isx`, `Unq`/`Syn` combine KSG MI with Ehrlich `I^sx`
  redundancy (different bias profiles), so small near-zero atoms can be an estimator artefact rather
  than structure. Likewise, do not pool continuous `I^sx_∩` atoms with discrete `I_min` atoms — they
  are different PID measures.

## Install

```toml
[dependencies]
pid-core = { git = "https://github.com/sepahead/pid-rs" }
```

> A crates.io release is planned; until then, depend on the Git repository.
>
> Using Python? See the [Python bindings](#python) below for `pip install maturin` and `maturin develop`.

## Quickstart

```rust
use pid_core::{ksg_mi, pid2_isx, IsxConfig, KsgConfig, MatRef, NegativeHandling, Pid2Config};

// Columns are dimensions, rows are samples. Here: scalar S1, S2, T (n samples each).
// (s1_data/s2_data/t_data/n are your own `&[f64]` buffers; see examples/ksg_and_pid.rs for a runnable version.)
let s1 = MatRef::new(&s1_data, n, 1)?;
let s2 = MatRef::new(&s2_data, n, 1)?;
let t  = MatRef::new(&t_data,  n, 1)?;

// Mutual information (nats).
let ksg = KsgConfig { negative_handling: NegativeHandling::Allow, ..Default::default() };
let mi = ksg_mi(s1, t, &ksg)?;

// 2-source PID atoms via I^sx_∩.
let pid = pid2_isx(s1, s2, t, &Pid2Config { ksg, isx: IsxConfig::default() })?;
println!("Red={:.3}  Unq1={:.3}  Unq2={:.3}  Syn={:.3}",
         pid.redundancy, pid.unique_s1, pid.unique_s2, pid.synergy);
# Ok::<(), pid_core::PidError>(())
```

Run the worked example end-to-end:

```bash
cargo run --release --example ksg_and_pid
```

## Conventions

- **Units:** all information quantities are in **nats** (natural log).
- **Co-information sign:** for 2 sources `CI₂ = Red − Syn`, so *negative ⇒ synergy-dominant*. This
  **does not** carry over to 3 sources — `CI₃` is parity-flipped (a pure 3-way synergy gives
  `CI₃ > 0`) and conflates atoms, so it is only a coarse screen.
- **Negative atoms are real:** `I^sx_∩` trades all-atom non-negativity for the target chain rule, so
  atoms (including redundancy) can be negative. The library never silently clamps them away — that
  is an opt-in reporting choice (`NegativeHandling`).

## ⚠️ Scientific cautions (read before trusting results)

kNN information estimators are powerful but have well-known failure modes. **Validate before you
interpret.**

- **i.i.d. assumption** — trajectory/time-series autocorrelation biases kNN MI. Subsample or use the
  block bootstrap.
- **Distance concentration** — in high ambient/intrinsic dimension, kNN geometry degrades; check the
  geometry diagnostics first.
- **Strong dependence** — near-deterministic relationships (very large true MI) can need prohibitive
  sample sizes (Gao et al. 2015).
- **Estimator ≠ truth** — do not interpret a downstream result without passing a validation gate on
  synthetic systems whose information quantities are known analytically.

The `exp0` binary is that diagnostic gate (synthetic systems with known MI, noise-dimension
invariance, strong-dependence sweeps). It sweeps dimensions up to 256 at n=500 — a range that
*deliberately* includes regimes where kNN MI is known to break down — so a `PIVOT`/`NO-GO` verdict on
the full default sweep is the expected, informative outcome, not a build failure. It reports
per-check counters (Monotonicity / Invariant / Geometry), and exits 0 by default; pass
`--strict-gate` to make it exit non-zero unless the verdict is `GO` (e.g. to enforce a regime you
have already validated in CI).

```bash
cargo run -p pid-core --bin exp0 -- --seeds 4 --summary-json summary.json --runlog run.jsonl
cargo run -p pid-runlog --bin pid-runlog-replay -- --validate run.jsonl
```

## Validation

Correctness is checked against **analytically known ground truth**, not just self-consistency:

- KSG MI vs the closed-form Gaussian-channel MI `I = −½ ln(1 − ρ²)`.
- Continuous `I^sx_∩` against a fixed-data reference computation.
- Discrete PID against the **independently re-derived** `I_min` and the known structure of canonical
  gates (XOR = pure synergy, COPY = pure redundancy, …).
- 2-/3-source PID identities (atoms reconstruct total MI) within `1e-10`.
- `parallel` feature results are **bit-identical** to the serial path.

See [`crates/pid-core/tests`](crates/pid-core/tests) for the suite.

## Known limitations

This is a `0.1.0` release. The estimator **core** (KSG, continuous `I^sx_∩`, discrete `I_min`, and
the PID identities) is validated against analytic ground truth, but the surrounding
statistics/convenience layer has tracked follow-ups (see the issue tracker):

- **No multiple-comparison correction.** When testing many atoms × sources × windows, apply your
  own FDR/FWER control; the library reports raw per-atom p-values.
- **Legacy `bootstrap_pid3` / `permutation_pid3` helpers** use a fixed-grid block bootstrap and a
  full-row-shuffle permutation that are *not* recommended for autocorrelated/kNN data. Prefer the
  moving-block `block_bootstrap` and the row-level resampling helpers, and a block-permutation null
  for trajectory data.
- **Cross-estimator PID2 atoms.** `Unq`/`Syn` combine KSG MI with Ehrlich `I^sx` redundancy
  (different bias profiles); small near-zero atoms can be an estimator artefact rather than
  structure.
- **External cross-validation provenance.** Discrete-PID values are checked against an independent
  in-repo re-derivation and the analytic structure of canonical gates; an external `csxpid`
  cross-check is planned.

None of these affects a single point estimate of MI or a PID atom — they concern *uncertainty
quantification* and *convenience-API ergonomics*.

## Estimators &amp; references

| Component | Reference |
|---|---|
| KSG mutual information | Kraskov, Stögbauer &amp; Grassberger (2004), *Phys. Rev. E* **69**, 066138 |
| Shared-exclusions redundancy `I^sx_∩` (discrete) | Makkeh, Gutknecht &amp; Wibral (2021), *Phys. Rev. E* **103**, 032149 |
| Continuous `I^sx_∩` kNN estimator | Ehrlich, Schick-Poland, Makkeh, Lanfermann, Wollstadt &amp; Wibral (2024), [arXiv:2311.06373](https://arxiv.org/abs/2311.06373) |
| `I_min` redundancy &amp; the PID lattice | Williams &amp; Beer (2010), [arXiv:1004.2515](https://arxiv.org/abs/1004.2515) |
| Shannon invariants (`r̄`, `v̄`, O-information) | Gutknecht et al. (2025), [arXiv:2504.15779](https://arxiv.org/abs/2504.15779) |
| PID non-negativity / chain-rule / invariance trilemma | Matthias, Makkeh, Wibral &amp; Gutknecht (2025), [arXiv:2512.16662](https://arxiv.org/abs/2512.16662) |
| kNN MI sample-complexity caveat | Gao, Ver Steeg &amp; Galstyan (2015), [arXiv:1411.2003](https://arxiv.org/abs/1411.2003) |

## Workspace

| Crate | Description |
|---|---|
| [`pid-core`](crates/pid-core) | The estimators, PID atoms, invariants, geometry, preprocessing, and the `exp0` validation harness. |
| [`pid-runlog`](crates/pid-runlog) | Versioned, content-addressed run-log schema + replay/validation CLI for reproducible pipelines. |
| [`pid-python`](crates/pid-python) | Python bindings (PyO3 + maturin); the `pid_core_rs` module — 15 functions over NumPy arrays. |

### Python

The `pid_core_rs` bindings live on `main` (added after the `0.1.0` tag) and are built as a
stable-ABI (`abi3`, CPython ≥ 3.11) wheel with maturin. Arrays are passed as **C-contiguous**
`float64` NumPy arrays (wrap transposed/`order='F'` arrays in `np.ascontiguousarray` first):

```bash
pip install maturin && maturin develop --release -m crates/pid-python/Cargo.toml
python -c "import numpy as np, pid_core_rs as p; print(p.compute_mi(np.random.randn(400,1), np.random.randn(400,1)))"
```

## Minimum supported Rust version

**1.80**. The MSRV is treated as a semver-relevant property and is exercised in CI.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For anything security-sensitive, see [SECURITY.md](SECURITY.md).

## Citation

If you use pid-rs in academic work, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub
renders a “Cite this repository” button) and cite the underlying estimator papers above.

## License

Licensed under either of

- **MIT** license ([LICENSE-MIT](LICENSE-MIT)), or
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Acknowledgements

The `I^sx_∩` measure and its continuous estimator are due to the Wibral group (Göttingen); this is
an independent, from-the-papers Rust implementation. Any errors are the maintainer's own.
