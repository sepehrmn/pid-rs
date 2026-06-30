//! Property/axiom tests for the discrete shared-exclusions PID (`i^sx_∩`), and the honest
//! `I_min`-vs-`i^sx` comparison on the two-bit COPY (the Harder et al. 2013 identity axiom).

use pid_core::{discrete_pid2, discrete_sxpid2, discrete_sxpid3, MatRef};

fn run2(rows: &[(usize, usize, usize)], num_bins: usize) -> pid_core::DiscreteSxPid2Result {
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

#[test]
fn reconstruction_and_self_redundancy() {
    // AND gate.
    let r = run2(&[(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], 2);
    let sum = r.unq1.net + r.unq2.net + r.syn.net + r.red.net;
    assert!(
        (sum - r.mi_s1s2_t).abs() < 1e-9,
        "reconstruction: {sum} vs {}",
        r.mi_s1s2_t
    );
    // Self-redundancy: information below the marginal node {i} sums to I(S_i;T).
    assert!((r.unq1.net + r.red.net - r.mi_s1_t).abs() < 1e-9);
    assert!((r.unq2.net + r.red.net - r.mi_s2_t).abs() < 1e-9);
}

#[test]
fn net_equals_informative_minus_misinformative() {
    let r = run2(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], 2); // XOR
    for p in &r.pointwise {
        for a in [p.unq1, p.unq2, p.syn, p.red] {
            assert!((a.net - (a.informative - a.misinformative)).abs() < 1e-12);
        }
    }
    for a in [r.unq1, r.unq2, r.syn, r.red] {
        assert!((a.net - (a.informative - a.misinformative)).abs() < 1e-12);
    }
}

#[test]
fn negative_atoms_are_real() {
    // XOR: pointwise AND averaged redundancy are negative — must not be clamped.
    let r = run2(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], 2);
    assert!(
        r.red.net < 0.0,
        "XOR averaged red should be negative; got {}",
        r.red.net
    );
    assert!(r.pointwise.iter().all(|p| p.red.net < 0.0));
}

#[test]
fn symmetry_under_source_swap() {
    let rows = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]; // AND
    let r = run2(&rows, 2);
    let swapped: Vec<(usize, usize, usize)> = rows.iter().map(|&(a, b, c)| (b, a, c)).collect();
    let rs = run2(&swapped, 2);
    assert!((r.unq1.net - rs.unq2.net).abs() < 1e-12);
    assert!((r.unq2.net - rs.unq1.net).abs() < 1e-12);
    assert!((r.red.net - rs.red.net).abs() < 1e-12);
    assert!((r.syn.net - rs.syn.net).abs() < 1e-12);
}

#[test]
fn sxpid3_reconstruction_and_symmetry() {
    // T = S0 (unique to source 0); S0,S1,S2 fully enumerated over {0,1}^3 so the empirical law is
    // EXACTLY symmetric under S1↔S2. Reconstruction: Σ_α Π(α) = I(S0,S1,S2;T) = H(S0) = ln 2.
    let (mut s0, mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..4 {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    s0.push(a as f64);
                    s1.push(b as f64);
                    s2.push(c as f64);
                    t.push(a as f64); // T = S0
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

    // Reconstruction.
    let sum: f64 = r.atoms.iter().map(|a| a.net).sum();
    assert!(
        (sum - r.mi_s0s1s2_t).abs() < 1e-9,
        "Σ={sum} vs joint MI {}",
        r.mi_s0s1s2_t
    );
    assert!((r.mi_s0s1s2_t - 2.0_f64.ln()).abs() < 1e-9);

    // Exact symmetry under S1↔S2: the unique-to-1 and unique-to-2 atoms coincide, as do the
    // {{0},{1}}-type pairs. (1=bit0 of source0, 2=bit1 of source1, 4=bit2 of source2.)
    let u1 = r.atom(&[0b010]).unwrap().net;
    let u2 = r.atom(&[0b100]).unwrap().net;
    assert!((u1 - u2).abs() < 1e-12, "unq1={u1} unq2={u2}");
    let p01 = r.atom(&[0b001, 0b010]).unwrap().net; // {{0},{1}}
    let p02 = r.atom(&[0b001, 0b100]).unwrap().net; // {{0},{2}}
    assert!(
        (p01 - p02).abs() < 1e-12,
        "{{0}}{{1}}={p01} {{0}}{{2}}={p02}"
    );

    // Net atom equals informative − misinformative everywhere.
    for a in &r.atoms {
        assert!((a.net - (a.informative - a.misinformative)).abs() < 1e-12);
    }
}

#[test]
fn identity_axiom_imin_overattributes_vs_sxpid() {
    // Two-bit COPY of INDEPENDENT sources: T = (S1, S2), encoded as 2*s1 + s2.
    // Harder et al. (2013) identity axiom: redundancy should be I(S1;S2) = 0.
    //   - I_min assigns the MAXIMAL 1 bit (= min of the marginal MIs) — the classic
    //     over-attribution that motivated SxPID.
    //   - i^sx assigns only log2(4/3) ≈ 0.415 bits — strictly less (though still > 0; SxPID
    //     trades the averaged identity axiom for pointwise structure, per Bertschinger et al.:
    //     identity is incompatible with global non-negativity).
    let rows = [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)];
    let reps = 4;
    let (mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        for &(a, b, c) in &rows {
            s1.push(a as f64);
            s2.push(b as f64);
            t.push(c as f64);
        }
    }
    let n = rows.len() * reps;
    let s1m = MatRef::new(&s1, n, 1).unwrap();
    let s2m = MatRef::new(&s2, n, 1).unwrap();
    let tm = MatRef::new(&t, n, 1).unwrap();

    let imin = discrete_pid2(s1m, s2m, tm, 4).unwrap();
    let sx = discrete_sxpid2(s1m, s2m, tm, 4).unwrap();

    let ln2 = 2.0_f64.ln();
    let ln_4_3 = (4.0_f64 / 3.0).ln();

    assert!(
        (imin.redundancy - ln2).abs() < 1e-9,
        "I_min copy redundancy should be 1 bit (ln2); got {}",
        imin.redundancy
    );
    assert!(
        (sx.red.net - ln_4_3).abs() < 1e-9,
        "i^sx copy redundancy should be log(4/3); got {}",
        sx.red.net
    );
    // The headline: SxPID attributes strictly less spurious redundancy than I_min.
    assert!(
        sx.red.net < imin.redundancy - 1e-3,
        "i^sx ({}) should attribute less redundancy than I_min ({})",
        sx.red.net,
        imin.redundancy
    );
}
