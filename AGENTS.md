# AGENTS.md

Guidance for AI coding agents (and humans) working in **pid-rs**. Tool-agnostic; Claude Code also
reads `CLAUDE.md`, which imports this file.

This file is the operational guide (policy, commands, conventions, code map). For the *scientific*
picture — what PID is, which estimator does what, the references, and the caveats — read
[`README.md`](README.md) first; per-crate docs live in each `crates/*/README.md`.

## Contents

- [Commit & attribution policy (READ FIRST)](#commit--attribution-policy-read-first)
- [What this project is](#what-this-project-is)
- [Workspace layout](#workspace-layout)
- [Where things live in `pid-core`](#where-things-live-in-pid-core)
- [Build / test / lint (mirror CI)](#build--test--lint-mirror-ci)
- [Conventions to preserve](#conventions-to-preserve)
- [Before you push](#before-you-push)

## Commit & attribution policy (READ FIRST)

- **Do not add AI/agent attribution to commits or pull requests.** Never append a
  `Co-Authored-By:` trailer that names Claude, an AI, or an agent, and never add
  "Generated with Claude Code" / "Co-authored with …" / any agent advertising to commit messages
  or PR descriptions. Commits are authored **solely by the human contributor**.
- **Do not sign commits or tags.** This repository sets `commit.gpgsign=false` and
  `tag.gpgsign=false` locally; leave them unsigned.
- This is enforced by `.claude/settings.json` (`attribution.commit` and `attribution.pr` are empty
  strings). Do not re-introduce attribution there or in any commit you author.

## What this project is

A safe-Rust workspace for **partial information decomposition** (the shared-exclusions `I^sx_∩`
measure) and the continuous **k-nearest-neighbour** estimators it builds on (KSG mutual
information), plus discrete `I_min` PID, Shannon invariants, geometry diagnostics, preprocessing/PLS,
dependence-aware uncertainty quantification, reproducible run-logs, and Python bindings.

## Workspace layout

| Crate | Path | Role |
|---|---|---|
| `pid-core` | `crates/pid-core` | The estimators, PID atoms, invariants, geometry, preprocessing, and the `exp0` validation/diagnostic binary. `#![forbid(unsafe_code)]`. |
| `pid-runlog` | `crates/pid-runlog` | Versioned, content-addressed run-log schema + the `pid-runlog-replay` CLI. |
| `pid-python` | `crates/pid-python` | PyO3 + maturin bindings (the `pid_core_rs` module). Built as an `abi3` wheel, not via plain `cargo`. |

## Where things live in `pid-core`

The public API is re-exported from `crates/pid-core/src/lib.rs`; the implementation is split by topic.
When you need to touch an estimator, start in the module below (each has unit/integration tests of
the same stem under `crates/pid-core/tests/`).

| Module (`src/…`) | Key public items | What it covers |
|---|---|---|
| `ksg.rs` | `ksg_mi`, `ksg_local_mi_terms`, `KsgConfig`, `NegativeHandling` | KSG continuous MI estimator. |
| `isx.rs` | `isx_redundancy`, `IsxConfig`, `IsxMethod` | Continuous `I^sx_∩` redundancy (Ehrlich et al. 2024). |
| `pid2.rs` | `pid2_isx`, `Pid2Config`, `Pid2Result` | 2-source PID atoms (Red/Unq1/Unq2/Syn). |
| `pid3.rs` | `pid3_isx`, `Pid3Config`, `Pid3Result`, `Antichain3` | 3-source PID atoms over the antichain lattice. |
| `discrete_pid.rs` | `discrete_pid2`, `discrete_pid3` | Discrete `I_min` PID (Williams & Beer 2010). |
| `invariants.rs` / `ci.rs` | `co_information_*`, Shannon invariants | Co-/O-information, `r̄`, `v̄` screening stats. |
| `geometry.rs` | intrinsic-dimension, distance, hyperbolicity | Geometry diagnostics for kNN-validity. |
| `preprocess.rs` / `pls.rs` | `Standardizer`, `PcaProjector`, `PlsProjector`, … | Standardisation, PCA, hash projection, jitter, PLS. |
| `bootstrap.rs` | `block_bootstrap`, `BootstrapConfig` | Dependence-aware uncertainty quantification. |
| `bin/exp0.rs` | — | The `exp0` validation/diagnostic binary (see below). |

A runnable end-to-end example (MI + 2-source PID on a synthetic system) lives at
`crates/pid-core/examples/ksg_and_pid.rs`.

## Build / test / lint (mirror CI)

```bash
cargo test --workspace --exclude pid-python                 # tests (pid-python is tested via maturin, below)
cargo test -p pid-core --features parallel                  # the exact data-parallel kNN path
cargo fmt --all --check                                     # formatting
cargo clippy --workspace --all-targets -- -D warnings       # lint (must be clean)
cargo clippy -p pid-core --all-targets --features parallel -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --exclude pid-python
# worked example: MI + 2-source PID on a synthetic system (fast sanity check)
cargo run --release --example ksg_and_pid
# smoke: the exp0 diagnostic + a run-log round-trip
cargo run -p pid-core --bin exp0 -- --seeds 1 --summary-json /tmp/summary.json --runlog /tmp/run.jsonl
cargo run -p pid-runlog --bin pid-runlog-replay -- --validate /tmp/run.jsonl
```

The example is the quickest "is the core working" check. Expected output (deterministic — the example
seeds its own RNG):

```text
Mutual information (nats):
  I(S1; T)     = 0.4209
  I(S2; T)     = 0.3798

2-source PID atoms (I^sx_∩), nats:
  Redundancy   = 0.1662
  Unique(S1)   = 0.2547
  Unique(S2)   = 0.2137
  Synergy      = 1.2350
  (sum of atoms = 1.8695 = I(S1,S2; T))
```

`pid-python` is a PyO3 extension module, so a plain `cargo test`/`cargo doc` over the whole workspace
fails locally (it links/loads `libpython`). Always `--exclude pid-python` for cargo, and exercise it
via maturin:

```bash
pip install maturin numpy pytest
maturin develop --release -m crates/pid-python/Cargo.toml
pytest crates/pid-python/tests -q
```

## Conventions to preserve

- **Units:** all information quantities are in **nats** (natural log).
- **PID identities:** MI terms that feed PID atoms must be computed with `NegativeHandling::Allow` —
  clamping a term before a subtraction breaks `Red + Unq1 + Unq2 + Syn = I(S1,S2;T)`.
- **Negative atoms are real:** `I^sx_∩` (and its atoms) can be negative; never silently clamp.
- **Determinism:** accumulate over count maps with `BTreeMap`/sorted keys (not `HashMap`); the
  `parallel` feature must stay bit-identical to the serial path; seed all RNGs explicitly.
- **`exp0` is a diagnostic gate, not a pass/fail test.** It emits a `GO`/`PIVOT`/`NO-GO` verdict
  from monotonicity / invariant / geometry counters and **exits 0 by default** — its default sweep
  goes to dimension 256 at n=500, deliberately entering regimes where kNN MI is known to break down,
  so `PIVOT`/`NO-GO` on the full sweep is the *expected, informative* outcome. Its checks use
  scale-aware tolerances. Pass `--strict-gate` to exit non-zero (code 3) unless the verdict is `GO`.
  Don't "fix" an expected `PIVOT` without understanding why.
- **Scientific changes:** a change that alters a numerical result must justify *why* the new value is
  correct (analytic ground truth or a cited paper), not merely that tests still pass.

## README-iff invariant (where READMEs may live, and how they wire in)

A directory gets a `README.md` **if and only if** it is one of:

- a **published artifact** (a crate published to crates.io, or a package published to PyPI), or
- a **directly-consumed unit** (something a human runs/imports on its own — a CLI, an example, a
  vendored tool), or
- a **browsed-asset directory** (a folder a reader lands in and expects orientation — e.g. the repo
  root, `crates/`).

No other directory should grow a stray `README.md`. If a folder is neither published, nor directly
consumed, nor browsed, it does not get one.

Wiring rules for the READMEs that do exist:

- **Rust library crates** (`pid-core`, `pid-runlog`): the crate README is the canonical crate-level
  doc and is wired into rustdoc via `#![doc = include_str!("../README.md")]` at the top of
  `src/lib.rs`. Because `include_str!` makes every ` ```rust ` and every **bare** ` ``` ` fence in
  the README a compiled-and-run doctest, audit the fences before wiring and re-fence:
  - prose / shell / commands / TOML / program output → ` ```text ` (never executed),
  - complete Rust that compiles but must not run → ` ```no_run `,
  - illustrative / incomplete / pseudocode Rust that won't compile (e.g. undefined vars like
    `s1_data` / `n`) → ` ```rust,ignore `.
  The bar is: `cargo test --doc -p <crate>` and
  `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p <crate>` both pass clean. Each such crate's
  `Cargo.toml` also carries `readme = "README.md"`, `documentation = "https://docs.rs/<crate>"`, and
  a `[package.metadata.docs.rs]` block (`all-features = true`, `rustdoc-args = ["--cfg", "docsrs"]`).
- **maturin / PyO3 extension crates** (`pid-python`): wire the README with the `readme = "README.md"`
  manifest key **only** — do **not** add `#![doc = include_str!(...)]`. Their rustdoc is not the
  primary documentation surface, and a standalone README plus `readme=` avoids any risk to the
  maturin/`abi3` build.

## Before you push

Run the build/test/lint block above (all must be clean), update `CHANGELOG.md` under
`[Unreleased]`, and keep PRs focused. For security issues, follow `SECURITY.md` (do not open a
public issue). See `CONTRIBUTING.md` for the full contributor guide.
