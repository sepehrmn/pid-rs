<!-- Thanks for contributing to pid-rs! Please skim CONTRIBUTING.md (numerical conventions + test commands) first. -->

## Summary

<!-- What does this PR change, and why? -->

## Checklist

- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean
- [ ] `cargo test --workspace --exclude pid-python` passes (mirrors CI; `pid-python` is exercised via `maturin` + `pytest`)
- [ ] Tests added/updated (prefer an analytic ground-truth check for estimator changes)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If `pid-python` changed: `maturin develop --release -m crates/pid-python/Cargo.toml && pytest crates/pid-python/tests -q` passes

## Numerical impact

<!-- Does this change any numerical result? If so, explain why the new value is correct
     (cite a paper or an analytic value). If not, write "none". -->
