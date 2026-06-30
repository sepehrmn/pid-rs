//! Property-based invariants for discrete shared-exclusions PID (`i^sx_∩`).
//!
//! These must hold for *every* discrete system, not just the canonical gates: the PID atoms
//! reconstruct the joint MI, and the down-set of each singleton node reconstructs that source's
//! MI (self-redundancy). We sweep many random systems (varying n, alphabet size, and skew) and
//! assert both identities to floating-point tolerance.

mod common;
use common::Rng64;

use pid_core::{discrete_sxpid2, discrete_sxpid3, MatRef};

/// Draw `n` integer labels in `0..alphabet`, with a deliberately skewed (non-uniform) law so the
/// probability-weighting is exercised. Returned as `f64` column data.
fn draw(rng: &mut Rng64, n: usize, alphabet: usize) -> Vec<f64> {
    (0..n)
        .map(|_| {
            // Square the uniform to skew toward small labels.
            let u = rng.next_f64();
            let v = (u * u * alphabet as f64) as usize;
            v.min(alphabet - 1) as f64
        })
        .collect()
}

#[test]
fn sxpid2_identities_hold_for_random_systems() {
    let mut rng = Rng64::new(0xA11CE);
    for trial in 0..60 {
        let n = 60 + (trial * 7) % 200;
        let alpha = 2 + (trial % 3); // 2..=4 distinct values per source
        let num_bins = alpha; // bins match the alphabet so labels separate
        let s1 = draw(&mut rng, n, alpha);
        let s2 = draw(&mut rng, n, alpha);
        // Target depends on both sources plus noise → all atoms generally nonzero.
        let t: Vec<f64> = (0..n)
            .map(|i| {
                let mix = s1[i] as usize + 2 * (s2[i] as usize) + (rng.next_u64() as usize % 2);
                (mix % (alpha + 1)) as f64
            })
            .collect();

        let s1m = MatRef::new(&s1, n, 1).unwrap();
        let s2m = MatRef::new(&s2, n, 1).unwrap();
        let tm = MatRef::new(&t, n, 1).unwrap();
        let r = discrete_sxpid2(s1m, s2m, tm, num_bins + 1).unwrap();

        let sum = r.unq1.net + r.unq2.net + r.syn.net + r.red.net;
        assert!(
            (sum - r.mi_s1s2_t).abs() < 1e-9,
            "trial {trial}: reconstruction {sum} != I(S1,S2;T) {}",
            r.mi_s1s2_t
        );
        assert!(
            (r.unq1.net + r.red.net - r.mi_s1_t).abs() < 1e-9,
            "trial {trial}: self-redundancy S1"
        );
        assert!(
            (r.unq2.net + r.red.net - r.mi_s2_t).abs() < 1e-9,
            "trial {trial}: self-redundancy S2"
        );
        // net == informative − misinformative, pointwise and averaged.
        for p in &r.pointwise {
            for a in [p.unq1, p.unq2, p.syn, p.red] {
                assert!((a.net - (a.informative - a.misinformative)).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn sxpid3_reconstruction_holds_for_random_systems() {
    let mut rng = Rng64::new(0xB0B);
    for trial in 0..40 {
        let n = 80 + (trial * 11) % 160;
        let alpha = 2 + (trial % 2); // 2..=3
        let s0 = draw(&mut rng, n, alpha);
        let s1 = draw(&mut rng, n, alpha);
        let s2 = draw(&mut rng, n, alpha);
        let t: Vec<f64> = (0..n)
            .map(|i| {
                let mix = s0[i] as usize + s1[i] as usize + s2[i] as usize;
                (mix % (alpha + 1)) as f64
            })
            .collect();

        let s0m = MatRef::new(&s0, n, 1).unwrap();
        let s1m = MatRef::new(&s1, n, 1).unwrap();
        let s2m = MatRef::new(&s2, n, 1).unwrap();
        let tm = MatRef::new(&t, n, 1).unwrap();
        let r = discrete_sxpid3(s0m, s1m, s2m, tm, alpha + 1).unwrap();

        let sum: f64 = r.atoms.iter().map(|a| a.net).sum();
        assert!(
            (sum - r.mi_s0s1s2_t).abs() < 1e-9,
            "trial {trial}: 3-source reconstruction {sum} != joint MI {}",
            r.mi_s0s1s2_t
        );
        assert_eq!(r.atoms.len(), 18);
    }
}
