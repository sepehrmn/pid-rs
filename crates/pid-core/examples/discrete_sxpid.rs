//! Discrete shared-exclusions PID (`i^sx_∩`, Makkeh–Gutknecht–Wibral 2021) on canonical logic
//! gates. Each gate is an exactly-enumerated distribution, so the output is deterministic and
//! matches the Abzinger/SxPID + IDTxl reference values exactly. Run with:
//!
//! ```text
//! cargo run --release --example discrete_sxpid
//! ```
//!
//! Expected output (nats; identities hold to ~1e-12):
//!
//! ```text
//! XOR  (pure synergy; redundancy is NEGATIVE — the shared-exclusions signature)
//!   Red = -0.4055   Unq1 = 0.4055   Unq2 = 0.4055   Syn = 0.2877   | Σ = 0.6931 = I(S1,S2;T)
//! AND  (I_min would give Red = 0.2158; i^sx attributes less)
//!   Red =  0.0849   Unq1 = 0.1308   Unq2 = 0.1308   Syn = 0.2158   | Σ = 0.5623 = I(S1,S2;T)
//! COPY (T = (S1,S2), independent sources)
//!   Red =  0.2877   Unq1 = 0.4055   Unq2 = 0.4055   Syn = 0.2877   | Σ = 1.3863 = I(S1,S2;T)
//! ```
use pid_core::{discrete_sxpid2, DiscreteSxPid2Result, MatRef};

/// Build an exactly-enumerated 2-input gate from `(s1, s2, t)` rows.
fn gate(rows: &[(usize, usize, usize)], num_bins: usize) -> DiscreteSxPid2Result {
    let (mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for &(a, b, c) in rows {
        s1.push(a as f64);
        s2.push(b as f64);
        t.push(c as f64);
    }
    let n = rows.len();
    let s1 = MatRef::new(&s1, n, 1).unwrap();
    let s2 = MatRef::new(&s2, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();
    discrete_sxpid2(s1, s2, t, num_bins).unwrap()
}

fn show(name: &str, note: &str, r: &DiscreteSxPid2Result) {
    let sum = r.unq1.net + r.unq2.net + r.syn.net + r.red.net;
    println!("{name}  ({note})");
    println!(
        "  Red = {:>7.4}   Unq1 = {:.4}   Unq2 = {:.4}   Syn = {:.4}   | Σ = {:.4} = I(S1,S2;T)",
        r.red.net, r.unq1.net, r.unq2.net, r.syn.net, sum
    );
}

fn main() {
    // XOR: T = S1 ⊕ S2.
    show(
        "XOR ",
        "pure synergy; redundancy is NEGATIVE — the shared-exclusions signature",
        &gate(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], 2),
    );
    // AND: T = S1 ∧ S2.
    show(
        "AND ",
        "I_min would give Red = 0.2158; i^sx attributes less",
        &gate(&[(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], 2),
    );
    // COPY: T = (S1, S2) encoded as 2*s1 + s2; independent sources.
    show(
        "COPY",
        "T = (S1,S2), independent sources",
        &gate(&[(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)], 4),
    );

    // The pointwise output is SxPID's signature: per-realization signed atoms.
    let xor = gate(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], 2);
    println!("\nXOR pointwise redundancy (each realization, nats):");
    for p in &xor.pointwise {
        println!(
            "  s1={:?} s2={:?} t={:?}  p={:.3}  red(net)={:+.4}  [inf {:+.4} − misinf {:+.4}]",
            p.s1, p.s2, p.t, p.prob, p.red.net, p.red.informative, p.red.misinformative
        );
    }
}
