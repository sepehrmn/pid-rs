# AGENTS.md

Guidance for AI coding agents (and humans) working in **pid-rs**. Tool-agnostic; Claude Code also
reads `CLAUDE.md`, which imports this file.

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

## Build / test / lint (mirror CI)

```bash
cargo test --workspace --exclude pid-python                 # tests (pid-python is tested via maturin, below)
cargo test -p pid-core --features parallel                  # the exact data-parallel kNN path
cargo fmt --all --check                                     # formatting
cargo clippy --workspace --all-targets -- -D warnings       # lint (must be clean)
cargo clippy -p pid-core --all-targets --features parallel -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --exclude pid-python
# smoke: the exp0 diagnostic + a run-log round-trip
cargo run -p pid-core --bin exp0 -- --seeds 1 --summary-json /tmp/summary.json --runlog /tmp/run.jsonl
cargo run -p pid-runlog --bin pid-runlog-replay -- --validate /tmp/run.jsonl
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
- **`exp0` is a diagnostic gate, not a pass/fail test.** `PIVOT`/`NO-GO` is expected at high
  dimensions; its monotonicity/invariant checks use scale-aware tolerances. CI enforces `GO` only
  under `--strict-gate`. Don't "fix" an expected `PIVOT` without understanding why.
- **Scientific changes:** a change that alters a numerical result must justify *why* the new value is
  correct (analytic ground truth or a cited paper), not merely that tests still pass.

## Before you push

Run the build/test/lint block above (all must be clean), update `CHANGELOG.md` under
`[Unreleased]`, and keep PRs focused. For security issues, follow `SECURITY.md` (do not open a
public issue). See `CONTRIBUTING.md` for the full contributor guide.
