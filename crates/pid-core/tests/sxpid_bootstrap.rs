//! Dependence-aware bootstrap confidence intervals for discrete SxPID atoms.

use pid_core::{bootstrap_discrete_sxpid2, discrete_sxpid2, BootstrapConfig, MatRef};

fn and_gate(reps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let rows = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)];
    let (mut s1, mut s2, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        for &(a, b, c) in &rows {
            s1.push(a as f64);
            s2.push(b as f64);
            t.push(c as f64);
        }
    }
    (s1.clone(), s2.clone(), t.clone(), 4 * reps)
}

#[test]
fn bootstrap_sxpid2_point_estimate_and_ci() {
    let (s1, s2, t, n) = and_gate(40); // n = 160
    let s1 = MatRef::new(&s1, n, 1).unwrap();
    let s2 = MatRef::new(&s2, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();

    let cfg = BootstrapConfig {
        n_boot: 200,
        block_size: 1, // i.i.d. rows
        seed: 7,
        alpha: 0.05,
    };
    let boot = bootstrap_discrete_sxpid2(s1, s2, t, 2, &cfg).unwrap();

    // Point estimate equals the direct estimator exactly.
    let direct = discrete_sxpid2(s1, s2, t, 2).unwrap();
    assert!((boot.redundancy.point_estimate - direct.red.net).abs() < 1e-12);
    assert!((boot.synergy.point_estimate - direct.syn.net).abs() < 1e-12);

    // Discrete data ⇒ every resample is valid (no NaN/instability from duplicates).
    for s in [
        &boot.redundancy,
        &boot.unique_s1,
        &boot.unique_s2,
        &boot.synergy,
    ] {
        assert_eq!(
            s.n_valid, cfg.n_boot,
            "all discrete resamples should be valid"
        );
        assert!(s.boot_se.is_finite() && s.boot_se >= 0.0);
        assert!(s.ci_low <= s.ci_high, "CI must be ordered");
        // The bootstrap mean lies within its own percentile interval.
        assert!(s.ci_low <= s.boot_mean + 1e-12 && s.boot_mean <= s.ci_high + 1e-12);
    }
    // A balanced gate resampled with replacement has nonzero spread in the redundancy atom.
    assert!(boot.redundancy.boot_se > 0.0);
}
