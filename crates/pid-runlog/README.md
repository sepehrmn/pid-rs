# pid-runlog

[![CI](https://github.com/sepahead/pid-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sepahead/pid-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A versioned, content-addressed **run-log schema** and replay/validation helpers for reproducible
partial-information-decomposition pipelines (used by [`pid-core`](../pid-core)). Each record's
payload is content-addressed (SHA-256); a run additionally carries a whole-trace replay hash and a
whole-file SHA-256 manifest, so it can be replayed and integrity-checked offline. The current run-log schema version is `1` (`RUN_LOG_SCHEMA_VERSION`). (Records are *not*
prev-hash-chained — tamper-evidence comes from the per-record and whole-trace/file digests.)

```text
# validate a run-log produced by an experiment
cargo run -p pid-runlog --bin pid-runlog-replay -- --validate run.jsonl
```

The `pid-runlog-replay` binary also supports:

```text
pid-runlog-replay <run-log.jsonl>                       # replay and print a summary
pid-runlog-replay --validate <run-log.jsonl>            # schema + integrity checks
pid-runlog-replay --compare <left.jsonl> <right.jsonl>  # compare whole-trace replay hashes
pid-runlog-replay --summary-json <run-log.jsonl> <out.json>
pid-runlog-replay --manifest-json <run-log.jsonl> <out.json>
pid-runlog-replay --write-sidecars <run-log.jsonl>     # write validation/summary/manifest sidecars
pid-runlog-replay --verify-sidecars <run-log.jsonl>    # re-derive and check sidecars
```

See the [repository README](https://github.com/sepahead/pid-rs) for context.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE) at your option.
