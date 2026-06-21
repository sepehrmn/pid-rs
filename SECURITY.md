# Security Policy

## Supported versions

pid-rs is pre-1.0. Security fixes are shipped in a new patch release on the latest `0.x` line.

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for security-sensitive reports.

Instead, use GitHub's [private vulnerability reporting](https://github.com/sepahead/pid-rs/security/advisories/new)
("Report a vulnerability" under the repository's **Security** tab). Include a description, a
minimal reproduction, and the affected version/commit.

You can expect an initial acknowledgement within a few days. After triage we will work with you
on a fix and a coordinated disclosure before any public disclosure. The core estimator library
`pid-core` is `#![forbid(unsafe_code)]` and has no network or filesystem surface in its library
path, so the most likely classes of issue there are denial-of-service via panics on crafted input
or incorrect numerical results; both are treated seriously. The `pid-python` bindings use PyO3
(FFI) and the `pid-runlog` crate and the `exp0` binary read and write files on disk, so reports
touching those paths are equally in scope.
