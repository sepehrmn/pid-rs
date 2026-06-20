# pid-core

[![CI](https://github.com/sepahead/pid-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sepahead/pid-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Continuous mutual-information and **shared-exclusions partial information decomposition**
(`I^sx_∩` PID) estimators in safe Rust (`#![forbid(unsafe_code)]`).

```rust
use pid_core::{pid2_isx, IsxConfig, KsgConfig, MatRef, Pid2Config};

// Columns are dimensions, rows are samples. Here: scalar S1, S2, T (n samples each).
let s1 = MatRef::new(&s1_data, n, 1)?;
let s2 = MatRef::new(&s2_data, n, 1)?;
let t  = MatRef::new(&t_data,  n, 1)?;
let pid = pid2_isx(s1, s2, t, &Pid2Config {
    ksg: KsgConfig::default(),
    isx: IsxConfig::default(),
})?;
println!("Red={:.3} Unq1={:.3} Unq2={:.3} Syn={:.3}",
         pid.redundancy, pid.unique_s1, pid.unique_s2, pid.synergy); // values in nats
# Ok::<(), pid_core::PidError>(())
```

See the [repository README](https://github.com/sepahead/pid-rs) for the full feature list,
estimator references, scientific cautions, and validation strategy.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE) at your option.
