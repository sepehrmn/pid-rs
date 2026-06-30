//! Bit-faithful regression of the discrete shared-exclusions PID (`i^sx_∩`) against the reference
//! implementation IDTxl wraps (Abzinger/SxPID `testing/test_gates.py`) and IDTxl's own
//! `test_estimators_multivariate_pid.py`.
//!
//! The reference values are **pointwise** signed atoms in **bits** (`log2`); this crate works in
//! **nats**, so every expected value is multiplied by `ln 2`. Pointwise vectors are compared as
//! an (encoding-independent) multiset, since binning relabels the realizations.

use pid_core::{discrete_sxpid2, discrete_sxpid3, DiscreteSxPid2Result, MatRef};
use std::f64::consts::LN_2;

/// Build an exactly-enumerated 2-input gate (each row once per `rep`, so the empirical pmf is
/// exact). Values are integers; `num_bins` must separate every variable's distinct values.
fn run2(rows: &[(usize, usize, usize)], num_bins: usize) -> DiscreteSxPid2Result {
    let reps = 4;
    let (mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        for &(a, b, c) in rows {
            s1.push(a as f64);
            s2.push(b as f64);
            t.push(c as f64);
        }
    }
    let n = rows.len() * reps;
    let s1 = MatRef::new(&s1, n, 1).unwrap();
    let s2 = MatRef::new(&s2, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();
    discrete_sxpid2(s1, s2, t, num_bins).unwrap()
}

fn key(v: &[f64; 4]) -> [i64; 4] {
    // Quantize to 1e-9 for a stable sort key.
    std::array::from_fn(|i| (v[i] * 1e9).round() as i64)
}

/// Assert the multiset of pointwise net-atom vectors `[unq1, unq2, syn, red]` equals `expected`
/// (each entry given as the *fractions* whose `ln` is the bit value × ln2, i.e. already in nats
/// once we take `ln`). `expected` holds fractions `f` meaning the atom value is `ln(f)`.
fn assert_pointwise(r: &DiscreteSxPid2Result, expected_fracs: &[[f64; 4]]) {
    let mut got: Vec<[f64; 4]> = r
        .pointwise
        .iter()
        .map(|p| [p.unq1.net, p.unq2.net, p.syn.net, p.red.net])
        .collect();
    let mut want: Vec<[f64; 4]> = expected_fracs
        .iter()
        .map(|f| std::array::from_fn(|i| f[i].ln()))
        .collect();
    assert_eq!(got.len(), want.len(), "realization count mismatch");
    got.sort_by_key(key);
    want.sort_by_key(key);
    for (g, w) in got.iter().zip(&want) {
        for i in 0..4 {
            assert!(
                (g[i] - w[i]).abs() < 1e-12,
                "atom[{i}] got {} want {} (full got {:?} want {:?})",
                g[i],
                w[i],
                g,
                w
            );
        }
    }
}

#[test]
fn xor_pointwise() {
    let r = run2(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], 2);
    let v = [1.5, 1.5, 4.0 / 3.0, 2.0 / 3.0];
    assert_pointwise(&r, &[v, v, v, v]);
}

#[test]
fn and_pointwise() {
    let r = run2(&[(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], 2);
    assert_pointwise(
        &r,
        &[
            [1.0, 1.0, 1.0, 4.0 / 3.0],             // (0,0,0)
            [1.5, 3.0 / 4.0, 4.0 / 3.0, 8.0 / 9.0], // (0,1,0)
            [3.0 / 4.0, 1.5, 4.0 / 3.0, 8.0 / 9.0], // (1,0,0)
            [1.5, 1.5, 4.0 / 3.0, 4.0 / 3.0],       // (1,1,1)
        ],
    );
    // IDTxl averaged shared(AND).
    assert!((r.red.net - 0.12255624891826572 * LN_2).abs() < 1e-12);
    // Informative / misinformative split of the averaged redundancy (independently re-derived):
    //   Π⁺_red = 0.4150374992788438 bits, Π⁻_red = 0.2924812503605781 bits, difference = net.
    assert!(
        (r.red.informative - 0.4150374992788438 * LN_2).abs() < 1e-12,
        "Π⁺={}",
        r.red.informative
    );
    assert!(
        (r.red.misinformative - 0.2924812503605781 * LN_2).abs() < 1e-12,
        "Π⁻={}",
        r.red.misinformative
    );
    // Informative atoms at NON-bottom nodes, independently hand-derived (uniform inputs ⇒ π⁺ is
    // constant across realizations): π⁺(unq1)=ln(3/2), π⁺(syn)=ln(4/3). This pins that the
    // informative/misinformative Möbius split is correct beyond the bottom (redundancy) node.
    assert!(
        (r.unq1.informative - 1.5_f64.ln()).abs() < 1e-12,
        "π⁺(unq1)={}",
        r.unq1.informative
    );
    assert!(
        (r.unq2.informative - 1.5_f64.ln()).abs() < 1e-12,
        "π⁺(unq2)={}",
        r.unq2.informative
    );
    assert!(
        (r.syn.informative - (4.0_f64 / 3.0).ln()).abs() < 1e-12,
        "π⁺(syn)={}",
        r.syn.informative
    );
}

/// Find the pointwise entry whose (binned) realization equals `(a, b, c)`. For binary inputs with
/// `num_bins = 2`, the bin label equals the value, so this verifies realization↔atom *assignment*
/// (which the multiset comparison in `assert_pointwise` deliberately does not).
fn find2(r: &DiscreteSxPid2Result, a: usize, b: usize, c: usize) -> [f64; 4] {
    let p = r
        .pointwise
        .iter()
        .find(|p| p.s1 == vec![a] && p.s2 == vec![b] && p.t == vec![c])
        .expect("realization present");
    [p.unq1.net, p.unq2.net, p.syn.net, p.red.net]
}

#[test]
fn and_pointwise_keyed_assignment() {
    // Same AND gate, but assert each SPECIFIC realization carries the right atom vector — guards
    // against a realization↔atom misassignment that a multiset comparison would miss.
    let r = run2(&[(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], 2);
    let chk = |got: [f64; 4], frac: [f64; 4]| {
        for i in 0..4 {
            assert!(
                (got[i] - frac[i].ln()).abs() < 1e-12,
                "atom[{i}] {} vs ln{}",
                got[i],
                frac[i]
            );
        }
    };
    chk(find2(&r, 0, 0, 0), [1.0, 1.0, 1.0, 4.0 / 3.0]);
    chk(find2(&r, 0, 1, 0), [1.5, 3.0 / 4.0, 4.0 / 3.0, 8.0 / 9.0]); // s1=0,s2=1 → unq1 carries
    chk(find2(&r, 1, 0, 0), [3.0 / 4.0, 1.5, 4.0 / 3.0, 8.0 / 9.0]); // s1=1,s2=0 → unq2 carries
    chk(find2(&r, 1, 1, 1), [1.5, 1.5, 4.0 / 3.0, 4.0 / 3.0]);
}

#[test]
fn unq_pointwise() {
    // T = S1.
    let r = run2(&[(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)], 2);
    let v = [1.5, 3.0 / 4.0, 4.0 / 3.0, 4.0 / 3.0];
    assert_pointwise(&r, &[v, v, v, v]);
}

#[test]
fn rdn_pointwise() {
    // Giant bit: T = S1 = S2.
    let r = run2(&[(0, 0, 0), (1, 1, 1)], 2);
    let v = [1.0, 1.0, 1.0, 2.0];
    assert_pointwise(&r, &[v, v]);
}

#[test]
fn copy_pointwise() {
    // T = (S1, S2) encoded as 2*s1 + s2 ∈ {0,1,2,3}.
    let r = run2(&[(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)], 4);
    let v = [1.5, 1.5, 4.0 / 3.0, 4.0 / 3.0];
    assert_pointwise(&r, &[v, v, v, v]);
}

#[test]
fn pwunq_pointwise() {
    // Pointwise-unique: each realization's info is carried entirely by one source.
    let r = run2(&[(0, 1, 1), (1, 0, 1), (0, 2, 2), (2, 0, 2)], 3);
    assert_pointwise(
        &r,
        &[
            [1.0, 2.0, 1.0, 1.0], // (0,1,1): unq2 = 1 bit
            [2.0, 1.0, 1.0, 1.0], // (1,0,1): unq1 = 1 bit
            [1.0, 2.0, 1.0, 1.0], // (0,2,2): unq2 = 1 bit
            [2.0, 1.0, 1.0, 1.0], // (2,0,2): unq1 = 1 bit
        ],
    );
}

#[test]
fn sum_pointwise() {
    // T = S1 + S2 ∈ {0,1,2}.
    let r = run2(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)], 3);
    assert_pointwise(
        &r,
        &[
            [1.5, 1.5, 4.0 / 3.0, 4.0 / 3.0], // (0,0,0)
            [1.5, 1.5, 4.0 / 3.0, 2.0 / 3.0], // (0,1,1)
            [1.5, 1.5, 4.0 / 3.0, 2.0 / 3.0], // (1,0,1)
            [1.5, 1.5, 4.0 / 3.0, 4.0 / 3.0], // (1,1,2)
        ],
    );
}

#[test]
fn rnderr_pointwise_nonuniform() {
    // RndErr — a NON-UNIFORM distribution (the one regime the uniform gates don't exercise):
    //   p(0,0,0)=p(1,1,1)=3/8, p(0,1,0)=p(1,0,1)=1/8.
    // Built with integer multiplicities 3,3,1,1 (×reps) so the empirical pmf is exact.
    let weighted: &[((usize, usize, usize), usize)] = &[
        ((0, 0, 0), 3),
        ((1, 1, 1), 3),
        ((0, 1, 0), 1),
        ((1, 0, 1), 1),
    ];
    let reps = 4;
    let (mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        for &((a, b, c), w) in weighted {
            for _ in 0..w {
                s1.push(a as f64);
                s2.push(b as f64);
                t.push(c as f64);
            }
        }
    }
    let n = s1.len();
    let s1 = MatRef::new(&s1, n, 1).unwrap();
    let s2 = MatRef::new(&s2, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();
    let r = discrete_sxpid2(s1, s2, t, 2).unwrap();

    // Reference (Abzinger/SxPID test_gates.py), confirmed by independent hand-derivation:
    let v1 = [5.0 / 4.0, 15.0 / 16.0, 16.0 / 15.0, 8.0 / 5.0]; // (0,0,0) & (1,1,1)
    let v2 = [7.0 / 4.0, 7.0 / 16.0, 16.0 / 7.0, 8.0 / 7.0]; // (0,1,0) & (1,0,1)
    assert_pointwise(&r, &[v1, v1, v2, v2]);
}

#[test]
fn multidim_source_equivalent_to_scalar() {
    // A 2-D source whose two columns are identical carries the same information as its 1-D
    // version, so the XOR atoms must match the scalar XOR exactly. Exercises the multi-column
    // (Vec<Vec<usize>>) realization path that the scalar gates don't.
    let rows = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)];
    let reps = 4;
    let (mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        for &(a, b, c) in &rows {
            s1.push(a as f64);
            s1.push(a as f64); // second, duplicated column
            s2.push(b as f64);
            t.push(c as f64);
        }
    }
    let n = rows.len() * reps;
    let s1 = MatRef::new(&s1, n, 2).unwrap(); // 2-D source
    let s2 = MatRef::new(&s2, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();
    let r = discrete_sxpid2(s1, s2, t, 2).unwrap();
    let v = [1.5, 1.5, 4.0 / 3.0, 2.0 / 3.0];
    assert_pointwise(&r, &[v, v, v, v]);
}

#[test]
fn hash_3source_averaged_matches_idtxl() {
    // 3-way XOR (HASH): T = S0 ⊕ S1 ⊕ S2, iid fair bits.
    let (mut s0, mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..4 {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    s0.push(a as f64);
                    s1.push(b as f64);
                    s2.push(c as f64);
                    t.push((a ^ b ^ c) as f64);
                }
            }
        }
    }
    let n = 4 * 8;
    let s0 = MatRef::new(&s0, n, 1).unwrap();
    let s1 = MatRef::new(&s1, n, 1).unwrap();
    let s2 = MatRef::new(&s2, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();
    let r = discrete_sxpid3(s0, s1, s2, t, 2).unwrap();

    // IDTxl test_estimators_multivariate_pid.py values (bits).
    let shared = r.atom(&[0b001, 0b010, 0b100]).unwrap();
    let pairs = r.atom(&[0b011, 0b101, 0b110]).unwrap();
    let syn = r.atom(&[0b111]).unwrap();
    assert!(
        (shared.net - 0.1926450779423959 * LN_2).abs() < 1e-12,
        "shared={}",
        shared.net
    );
    assert!(
        (pairs.net - (-0.22686079328030903) * LN_2).abs() < 1e-12,
        "pairs={}",
        pairs.net
    );
    assert!(
        (syn.net - 0.24511249783653177 * LN_2).abs() < 1e-12,
        "syn={}",
        syn.net
    );

    // Reconstruction: all 18 atoms sum to the joint MI (= log 2 for a 3-way XOR).
    let sum: f64 = r.atoms.iter().map(|a| a.net).sum();
    assert!((sum - r.mi_s0s1s2_t).abs() < 1e-9);
    assert!((sum - 2.0_f64.ln()).abs() < 1e-9);
}
