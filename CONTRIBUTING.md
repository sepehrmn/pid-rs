# Contributing to pid-rs

Thanks for your interest in improving pid-rs! Contributions of all kinds are welcome — bug
reports, documentation, tests, and code.

## Ground rules

- Be respectful. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- This is a **scientific** library: correctness and reproducibility come first. A change that
  alters a numerical result must explain *why* the new value is correct (ideally against an
  analytic ground truth or a cited paper), not merely that tests still pass.
- Found a security issue? Do **not** open a public issue — follow [SECURITY.md](SECURITY.md) instead.

## Development

```bash
git clone https://github.com/sepehrmn/pid-rs
cd pid-rs

cargo test --workspace --exclude pid-python  # tests (mirror CI)
cargo test -p pid-core --features parallel    # exact data-parallel path
cargo fmt --all                            # format
cargo clippy --workspace --all-targets -- -D warnings   # lint (must be clean)
cargo clippy -p pid-core --all-targets --features parallel -- -D warnings  # lint the parallel path
cargo run --release --example ksg_and_pid  # worked example
cargo run -p pid-core --bin exp0 -- --seeds 1 --summary-json /tmp/summary.json --runlog /tmp/run.jsonl  # exp0 diagnostic + run-log
cargo run -p pid-runlog --bin pid-runlog-replay -- --validate /tmp/run.jsonl  # replay/validate the run-log
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --exclude pid-python  # docs (CI denies warnings)
```

`pid-python` is a PyO3 extension module: a plain `cargo test`/`cargo doc` over the whole workspace
tries to link/run against `libpython` and fails locally, so it is excluded above and exercised via
`maturin`. The quickest local loop uses `maturin develop`; CI instead builds a wheel and installs
it (`maturin build --release --manifest-path crates/pid-python/Cargo.toml --out dist` then
`pip install --no-index --find-links dist pid-core-rs`), but both run the same pytest suite:

```bash
pip install maturin numpy pytest
maturin develop --release -m crates/pid-python/Cargo.toml
pytest crates/pid-python/tests -q
```

Optional but encouraged:

```bash
cargo deny check         # supply-chain / license check (see deny.toml)
```

## Pull requests

1. Open an issue first for anything non-trivial, so we can agree on the approach.
2. Keep PRs focused; one logical change per PR.
3. Add or update tests. For estimators, prefer a test against a **known analytic value**
   (Gaussian-channel MI, XOR = pure synergy, COPY = pure redundancy, independence → 0) over a
   self-consistency check.
4. Run `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, and
   `cargo test --workspace --exclude pid-python` before pushing.
5. Update `CHANGELOG.md` under `[Unreleased]`.

## Numerical conventions (please preserve)

- All information quantities are in **nats**.
- MI terms that feed PID identities must be computed with `NegativeHandling::Allow` (clamping a
  term before a subtraction breaks `Red + Unq1 + Unq2 + Syn = I(S1,S2;T)`).
- Accumulations over count maps must be **order-deterministic** (use `BTreeMap`/sorted keys, not
  `HashMap`) so results are bit-reproducible.
- `exp0` is a **diagnostic gate**, not a pass/fail test: PIVOT/NO-GO is expected at high
  dimensions, and its monotonicity/invariant checks use scale-aware tolerances. CI enforces a GO
  only under `--strict-gate`; don't "fix" an expected PIVOT without understanding why.

## Licensing of contributions

Unless you state otherwise, any contribution you submit is dual-licensed under
**MIT OR Apache-2.0**, matching the project license.
