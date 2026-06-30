//! Semi-analytic numerical ground truth for the continuous `I^sx_∩` redundancy on a
//! jointly-Gaussian system, and triangulation against the discrete `i^sx_∩`.
//!
//! The continuous limit of the discrete shared-exclusions redundancy of `{{1},{2}}` is
//! (h-factors cancel as the bin width → 0):
//!
//!   i^sx_∩(t:{1},{2})  →  log[ w1·exp(i1) + w2·exp(i2) ],
//!   w_a = p_{S_a}(s_a) / (p_{S1}(s1)+p_{S2}(s2)),   i_a = pointwise MI i(s_a; t).
//!
//! For standardized jointly-Gaussian (S_a, T) with correlation ρ_a, both `i_a` and the marginal
//! densities are closed form, so the redundancy is a *numerically exact* functional value (only
//! Monte-Carlo error, no kNN, no fabrication). We assert the KSG `I^sx_∩` estimator converges to
//! it. This is the genuine analytic anchor for the continuous redundancy *atom* in the
//! non-degenerate (independent-additive) regime — where, contrary to a naive "Red→0" guess, the
//! shared-exclusions redundancy is strictly POSITIVE.

use pid_core::{
    discrete_sxpid2, pid2_isx, IsxConfig, KsgConfig, MatRef, NegativeHandling, Pid2Config,
};

mod common;
use common::Rng64;

/// Pointwise MI for a standardized bivariate Gaussian `(a,b)` with correlation `r` (nats):
///   i = -½ln(1-r²) - (r²(a²+b²) - 2 r a b) / (2(1-r²)).
fn pointwise_gaussian_mi(a: f64, b: f64, r: f64) -> f64 {
    let r2 = r * r;
    -0.5 * (1.0 - r2).ln() - (r2 * (a * a + b * b) - 2.0 * r * a * b) / (2.0 * (1.0 - r2))
}

/// Build the independent-additive Gaussian system, standardized. Returns (s1, s2, t, n, rho).
fn additive_gaussian(
    seed: u64,
    n: usize,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, f64) {
    let mut rng = Rng64::new(seed);
    let (mut s1, mut s2, mut t) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n {
        let a = rng.normal();
        let b = rng.normal();
        let z = rng.normal();
        s1.push(a);
        s2.push(b);
        t.push(a + b + sigma * z);
    }
    let std = |v: &mut Vec<f64>| {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let sd = var.sqrt();
        for x in v.iter_mut() {
            *x = (*x - mean) / sd;
        }
    };
    std(&mut s1);
    std(&mut s2);
    std(&mut t);
    let rho = 1.0 / (2.0 + sigma * sigma).sqrt(); // corr(S_a, T), preserved by standardization
    (s1, s2, t, n, rho)
}

/// Numerically exact (oracle) continuous `I^sx_∩` redundancy via the closed-form continuous limit.
fn oracle_isx_red(s1: &[f64], s2: &[f64], t: &[f64], rho: f64) -> f64 {
    let n = s1.len();
    let mut acc = 0.0;
    for i in 0..n {
        let i1 = pointwise_gaussian_mi(s1[i], t[i], rho);
        let i2 = pointwise_gaussian_mi(s2[i], t[i], rho);
        // weights from standard-normal marginal densities (the 1/√(2π) constants cancel).
        let p1 = (-0.5 * s1[i] * s1[i]).exp();
        let p2 = (-0.5 * s2[i] * s2[i]).exp();
        let (w1, w2) = (p1 / (p1 + p2), p2 / (p1 + p2));
        let m = i1.max(i2);
        acc += m + (w1 * (i1 - m).exp() + w2 * (i2 - m).exp()).ln();
    }
    acc / n as f64
}

#[test]
fn ksg_isx_redundancy_matches_gaussian_oracle_additive() {
    // The continuous KSG I^sx estimator must converge to the numerically-exact functional value —
    // which is POSITIVE (~0.22 nats), NOT zero, for independent additive Gaussian sources.
    let sigma = 0.6;
    let (s1, s2, t, n, rho) = additive_gaussian(0x0AC1_E517, 4000, sigma);
    let oracle = oracle_isx_red(&s1, &s2, &t, rho);

    // Sanity: the oracle is clearly positive and well below I(S1;T) — the shared-exclusions
    // redundancy is real here (this is the corrected scientific picture; see the module docs).
    let i_s1_t = -0.5 * (1.0 - rho * rho).ln();
    assert!(
        oracle > 0.15,
        "oracle i^sx Red should be clearly positive; got {oracle:.4}"
    );
    assert!(
        oracle < i_s1_t,
        "oracle Red {oracle:.4} should be < I(S1;T) {i_s1_t:.4}"
    );

    let s1m = MatRef::new(&s1, n, 1).unwrap();
    let s2m = MatRef::new(&s2, n, 1).unwrap();
    let tm = MatRef::new(&t, n, 1).unwrap();
    let cfg = Pid2Config {
        ksg: KsgConfig {
            k: 3,
            negative_handling: NegativeHandling::Allow,
            ..Default::default()
        },
        isx: IsxConfig::default(),
    };
    let out = pid2_isx(s1m, s2m, tm, &cfg).unwrap();

    // KSG estimate converges to the oracle within the documented kNN-PID tolerance.
    assert!(
        (out.redundancy - oracle).abs() < 0.08,
        "KSG I^sx Red {:.4} should match Gaussian oracle {:.4} (NOT 0) within 0.08 nats",
        out.redundancy,
        oracle
    );
}

#[test]
#[ignore = "diagnostic: discrete i^sx on fine bins triangulates the oracle (heavier, O(D^2))"]
fn discrete_isx_triangulates_oracle() {
    let sigma = 0.6;
    let (s1, s2, t, n, rho) = additive_gaussian(0x0AC1_E518, 6000, sigma);
    let oracle = oracle_isx_red(&s1, &s2, &t, rho);
    let s1m = MatRef::new(&s1, n, 1).unwrap();
    let s2m = MatRef::new(&s2, n, 1).unwrap();
    let tm = MatRef::new(&t, n, 1).unwrap();
    eprintln!("oracle continuous-limit i^sx Red = {oracle:.4} nats");
    for &bins in &[6usize, 8, 10, 12, 14] {
        let r = discrete_sxpid2(s1m, s2m, tm, bins).unwrap();
        eprintln!("  discrete_sxpid2 bins={bins:>2}: Red={:.4}", r.red.net);
    }
}
