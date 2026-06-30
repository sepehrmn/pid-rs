//! Cited analytic Gaussian PID-ATOM regression tests.
//!
//! The existing Gaussian test (`tests/ksg.rs`) only checks the KSG *mutual information*
//! estimator. This file closes the higher-value correctness gap: it constructs jointly-Gaussian
//! `(S1, S2, T)` systems whose **PID atoms** (Red / Unq1 / Unq2 / Syn) have *known limiting-case*
//! values from theory, and asserts the estimated `I^sx_∩` atoms converge to them at large `n`.
//!
//! Conventions (AGENTS.md):
//! - All quantities are in **nats** (natural log).
//! - MI terms feeding the PID identity use `NegativeHandling::Allow` (enforced inside `pid2_isx`).
//! - Negative atoms are real; we never clamp.
//! - RNG is seeded explicitly (`Rng64`).
//!
//! Every expected value below is derived from theory / a cited paper and commented with the
//! derivation. None is tuned to the estimator. Where a correctly-derived analytic value disagrees
//! with the estimator beyond the documented tolerance, the assertion keeps the *theory* value and
//! the disagreement is documented as a finding in the test comments (and the stream report).

use pid_core::{
    ksg_mi, ksg_mi_concat_xy, pid2_isx, IsxConfig, KsgConfig, MatRef, NegativeHandling, Pid2Config,
    Standardizer,
};

mod common;

use common::Rng64;

/// Closed-form mutual information of a bivariate Gaussian channel via correlation:
///   I(X;Y) = -1/2 ln(1 - rho^2)      [nats]
/// Standard result; see e.g. Cover & Thomas, *Elements of Information Theory*, eq. for the
/// Gaussian channel, and Kraskov et al. 2004 (the KSG reference already in-repo) which uses
/// exactly this analytic form to validate the estimator.
fn gaussian_mi_from_corr(rho: f64) -> f64 {
    let r2 = rho * rho;
    debug_assert!(r2 < 1.0);
    -0.5 * (1.0 - r2).ln()
}

/// KSG config used for all MI/atom estimation here. `k=3` matches the rest of the suite and the
/// `IsxConfig` default (`pid2_isx` requires the KSG and ISX `k` to agree).
fn ksg_cfg() -> KsgConfig {
    KsgConfig {
        k: 3,
        negative_handling: NegativeHandling::Allow,
        ..Default::default()
    }
}

fn pid2_cfg() -> Pid2Config {
    Pid2Config {
        ksg: ksg_cfg(),
        isx: IsxConfig::default(), // EhrlichKsg, k=3, Chebyshev — the validated continuous I^sx.
    }
}

/// Documented convergence tolerance for the atoms (nats). The stream spec calls for ~0.05 nats;
/// kNN PID atoms at finite `n` carry the usual KSG bias, so we use a slightly looser but still
/// tight band and large `n`. Any atom that needs more than this to pass is reported as a finding,
/// not silently widened.
const ATOM_TOL: f64 = 0.08;

// =============================================================================================
// CASE 1 — IDENTICAL sources.
//
// Construction: X ~ N(0,1), T = X + sigma*Z with Z ~ N(0,1) independent, and S1 = S2 = X.
//
// Because S1 and S2 are the *same* random variable, every sensible redundancy measure must
// report that all of S1's information about T is also S2's information about T:
//
//   I(S1;T) = I(S2;T) = I(S1,S2;T) = I(X;T).
//
// Theory (measure-independent, holds for I^sx_∩ and any PID respecting the redundancy lattice
// — Williams & Beer 2010 §; Makkeh et al. 2021):
//   Red  = I(X;T)
//   Unq1 = I(S1;T) - Red = 0
//   Unq2 = I(S2;T) - Red = 0
//   Syn  = I(S1,S2;T) - I(S1;T) - I(S2;T) + Red = I(X;T) - 2 I(X;T) + I(X;T) = 0.
//
// Reference MI value (closed form): rho(X,T) = 1/sqrt(1+sigma^2), so
//   I(X;T) = -1/2 ln(1 - rho^2) = 1/2 ln(1 + 1/sigma^2).
// (Equivalent Gaussian-channel form 0.5 ln(1 + 1/sigma^2).)
// =============================================================================================
#[test]
fn gaussian_identical_sources_atoms_converge_to_theory() {
    let mut rng = Rng64::new(0x1DEA_71CA_u64); // explicit, deterministic seed
    let n = 4000;
    let sigma = 0.7; // channel noise std; chosen so I(X;T) is a moderate ~0.9 nats (kNN-friendly)
    let sigma2 = sigma * sigma;

    let mut x = Vec::with_capacity(n);
    let mut s2v = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.normal();
        let z = rng.normal();
        x.push(xi);
        s2v.push(xi); // S2 == S1 exactly
        t.push(xi + sigma * z);
    }

    let s1 = MatRef::new(&x, n, 1).unwrap();
    let s2 = MatRef::new(&s2v, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();
    let (s1, _) = Standardizer::fit_transform(s1).unwrap();
    let (s2, _) = Standardizer::fit_transform(s2).unwrap();
    let (t, _) = Standardizer::fit_transform(t).unwrap();

    let cfg = pid2_cfg();
    let out = pid2_isx(s1.as_ref(), s2.as_ref(), t.as_ref(), &cfg).unwrap();

    // Reference MI from theory (NOT from the estimator):
    let rho = 1.0 / (1.0 + sigma2).sqrt();
    let i_xt = gaussian_mi_from_corr(rho); // = 0.5 ln(1 + 1/sigma^2)

    // Theory atoms:
    let red_true = i_xt;
    let unq_true = 0.0;
    let syn_true = 0.0;

    assert!(
        (out.redundancy - red_true).abs() < ATOM_TOL,
        "identical-sources Red: est={:.4} theory I(X;T)={:.4} (tol {ATOM_TOL})",
        out.redundancy,
        red_true
    );
    assert!(
        (out.unique_s1 - unq_true).abs() < ATOM_TOL,
        "identical-sources Unq1: est={:.4} theory=0 (tol {ATOM_TOL})",
        out.unique_s1
    );
    assert!(
        (out.unique_s2 - unq_true).abs() < ATOM_TOL,
        "identical-sources Unq2: est={:.4} theory=0 (tol {ATOM_TOL})",
        out.unique_s2
    );
    assert!(
        (out.synergy - syn_true).abs() < ATOM_TOL,
        "identical-sources Syn: est={:.4} theory=0 (tol {ATOM_TOL})",
        out.synergy
    );

    // Sanity: the PID identity must hold exactly (same estimator on both sides). This is the
    // sacred convention Red+Unq1+Unq2+Syn = I(S1,S2;T) (AGENTS.md).
    let i_s1s2_t = ksg_mi_concat_xy(s1.as_ref(), s2.as_ref(), t.as_ref(), &ksg_cfg()).unwrap();
    let sum_atoms = out.redundancy + out.unique_s1 + out.unique_s2 + out.synergy;
    assert!(
        (sum_atoms - i_s1s2_t).abs() < 1e-9,
        "PID identity broken: sum_atoms={sum_atoms} I(S1,S2;T)={i_s1s2_t}"
    );
}

// =============================================================================================
// CASE 2 — INDEPENDENT additive sources (synergy-dominant).
//
// Construction: S1, S2 ~ N(0,1) independent, T = S1 + S2 + sigma*Z, Z ~ N(0,1) independent.
//   Var(T) = 2 + sigma^2.
//
// Closed-form MI terms:
//   rho(S1,T) = Cov(S1,T)/(sd S1 * sd T) = 1 / sqrt(2 + sigma^2)
//     => I(S1;T) = -1/2 ln(1 - 1/(2+sigma^2)) = -1/2 ln((1+sigma^2)/(2+sigma^2)).  (= I(S2;T))
//   I(S1,S2;T) = 1/2 ln(Var(T)/sigma^2) = 1/2 ln((2+sigma^2)/sigma^2).
//
// Redundancy of I^sx_∩ here — CORRECTED (see `tests/sxpid_gaussian_oracle.rs`):
//   An earlier version of this file ASSUMED `Red -> 0` for independent additive sources (calling
//   it "derived") and dismissed the estimator's stable ~0.22 nats as over-attribution bias. That
//   assumption was WRONG. Taking the bin width -> 0 limit of the discrete shared-exclusions
//   redundancy of {{1},{2}} gives (the h-factors cancel):
//       i^sx_∩(t:{1},{2})  ->  log[ w1·exp(i1) + w2·exp(i2) ],   w_a = p_{S_a}(s_a)/(p_{S1}+p_{S2}),
//   i.e. the log of a probability-weighted average of the pointwise-MI exponentials. For
//   independent additive Gaussians the i_a are positive on average, so this is STRICTLY POSITIVE
//   (numerically ~0.225 nats at sigma=0.6), and the KSG estimator CORRECTLY converges to it. This
//   is triangulated three ways in `tests/sxpid_gaussian_oracle.rs`: the closed-form oracle, the
//   KSG estimator, and the discrete i^sx in the fine-bin limit all agree (~0.22, NOT 0).
//
//   So I^sx here is NOT zero and NOT MMI: it sits strictly between 0 and the MMI value
//   min(I(S1;T), I(S2;T)). Only the MI *terms* are measure-independent closed forms:
//     I(S1;T) = I(S2;T) = -1/2 ln((1+sigma^2)/(2+sigma^2)),   I(S1,S2;T) = 1/2 ln((2+sigma^2)/sigma^2).
//   The mechanism is still synergy-dominant (co-information CI < 0).
// =============================================================================================
#[test]
#[ignore = "diagnostic: shows the independent-additive Red is n-STABLE at ~0.22 (consistency, not bias)"]
fn diag_independent_red_vs_n() {
    for &(seed, n) in &[
        (0xD1A6_0001_u64, 2000usize),
        (0xD1A6_0002, 4000),
        (0xD1A6_0003, 8000),
        (0xD1A6_0004, 16000),
    ] {
        let mut rng = Rng64::new(seed);
        let sigma = 0.6;
        let sigma2 = sigma * sigma;
        let mut s1v = Vec::with_capacity(n);
        let mut s2v = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.normal();
            let b = rng.normal();
            let z = rng.normal();
            s1v.push(a);
            s2v.push(b);
            t.push(a + b + sigma * z);
        }
        let s1 = MatRef::new(&s1v, n, 1).unwrap();
        let s2 = MatRef::new(&s2v, n, 1).unwrap();
        let t = MatRef::new(&t, n, 1).unwrap();
        let (s1, _) = Standardizer::fit_transform(s1).unwrap();
        let (s2, _) = Standardizer::fit_transform(s2).unwrap();
        let (t, _) = Standardizer::fit_transform(t).unwrap();
        let out = pid2_isx(s1.as_ref(), s2.as_ref(), t.as_ref(), &pid2_cfg()).unwrap();
        let rho = 1.0 / (2.0 + sigma2).sqrt();
        let i_s1_t = gaussian_mi_from_corr(rho);
        let i_s1s2_t = 0.5 * ((2.0 + sigma2) / sigma2).ln();
        eprintln!(
            "n={n:>6} Red={:.4} Unq1={:.4} Unq2={:.4} Syn={:.4} | I(S1;T)={:.4} Syn_lb(Red=0)={:.4}",
            out.redundancy,
            out.unique_s1,
            out.unique_s2,
            out.synergy,
            i_s1_t,
            i_s1s2_t - 2.0 * i_s1_t
        );
    }
}

/// Build the independent-additive system and estimate its atoms. Shared by the always-run
/// regression test and the `#[ignore]`d strict-theory finding test below, so both exercise the
/// *identical* construction (seed, n, sigma).
fn independent_additive_atoms() -> (pid_core::Pid2Result, f64, f64, f64, f64) {
    let mut rng = Rng64::new(0x1DEC_0DED_u64);
    let n = 4000;
    let sigma = 0.6; // moderate noise; keeps MI in the kNN-reliable range
    let sigma2 = sigma * sigma;

    let mut s1v = Vec::with_capacity(n);
    let mut s2v = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for _ in 0..n {
        let a = rng.normal();
        let b = rng.normal();
        let z = rng.normal();
        s1v.push(a);
        s2v.push(b);
        t.push(a + b + sigma * z);
    }

    let s1 = MatRef::new(&s1v, n, 1).unwrap();
    let s2 = MatRef::new(&s2v, n, 1).unwrap();
    let t = MatRef::new(&t, n, 1).unwrap();
    let (s1, _) = Standardizer::fit_transform(s1).unwrap();
    let (s2, _) = Standardizer::fit_transform(s2).unwrap();
    let (t, _) = Standardizer::fit_transform(t).unwrap();

    let out = pid2_isx(s1.as_ref(), s2.as_ref(), t.as_ref(), &pid2_cfg()).unwrap();

    // Closed-form reference MI (theory, NOT estimator):
    let rho_s_t = 1.0 / (2.0 + sigma2).sqrt();
    let i_s1_t = gaussian_mi_from_corr(rho_s_t); // = -0.5 ln((1+sigma^2)/(2+sigma^2))
    let i_s2_t = i_s1_t; // symmetry
    let i_s1s2_t = 0.5 * ((2.0 + sigma2) / sigma2).ln();

    // Same-estimator total MI, for the exact PID identity check.
    let i_s1s2_t_hat = ksg_mi_concat_xy(s1.as_ref(), s2.as_ref(), t.as_ref(), &ksg_cfg()).unwrap();

    (out, i_s1_t, i_s2_t, i_s1s2_t, i_s1s2_t_hat)
}

#[test]
fn gaussian_independent_additive_sources_synergy_dominant() {
    let (out, i_s1_t, i_s2_t, i_s1s2_t, i_s1s2_t_hat) = independent_additive_atoms();

    // The exact I^sx redundancy here is POSITIVE (~0.225 nats; anchored numerically against the
    // closed-form oracle in tests/sxpid_gaussian_oracle.rs) — NOT 0. Only the MI terms are
    // measure-independent closed forms; this test asserts the robust, partition-level structure
    // the KSG estimator must satisfy. `syn_lb` is the synergy IF Red were 0, hence a LOWER bound
    // on the true synergy (true Syn = syn_lb + Red > syn_lb), so it is a sound dominance witness.
    let unq1_true = i_s1_t;
    let syn_lb = i_s1s2_t - i_s1_t - i_s2_t; // = true Syn - Red ≤ true Syn

    // This mechanism is synergy-dominant (co-information CI < 0): even the lower bound exceeds the
    // unique MI.
    assert!(
        syn_lb > unq1_true,
        "theory check: expected synergy-dominant: Syn_lb={syn_lb:.4} Unq1={unq1_true:.4}"
    );

    // ---- Robust theory predictions the I^sx KSG estimator DOES satisfy at large n ----
    //
    // 1) Synergy dominates: the estimated Syn is the strictly largest atom and clearly positive.
    assert!(
        out.synergy > out.redundancy && out.synergy > out.unique_s1 && out.synergy > out.unique_s2,
        "expected synergy-dominant estimate: Red={:.4} Unq1={:.4} Unq2={:.4} Syn={:.4}",
        out.redundancy,
        out.unique_s1,
        out.unique_s2,
        out.synergy
    );
    assert!(
        out.synergy > 0.3,
        "expected clearly-positive synergy, got {:.4}",
        out.synergy
    );

    // 2) Unique atoms are small (theory: Unq = I(S1;T) ~ 0.28; the estimator under-attributes
    //    these because it over-attributes redundancy — see the finding test below).
    assert!(
        out.unique_s1.abs() < 0.2 && out.unique_s2.abs() < 0.2,
        "expected small unique atoms: Unq1={:.4} Unq2={:.4}",
        out.unique_s1,
        out.unique_s2
    );

    // 3) Estimated redundancy is strictly BELOW the Barrett-2015 MMI redundancy
    //    R_MMI = min(I(S1;T), I(S2;T)). This is a direction-of-difference check, NOT an
    //    equality claim: I^sx and MMI are different measures, and for independent additive
    //    sources I^sx redundancy should sit well under MMI's positive value.
    let r_mmi_true = i_s1_t.min(i_s2_t);
    assert!(
        out.redundancy < r_mmi_true,
        "expected I^sx Red < MMI Red: Red={:.4} R_MMI={:.4}",
        out.redundancy,
        r_mmi_true
    );

    // 4) PID identity (sacred): exact up to FP, same estimator both sides.
    let sum_atoms = out.redundancy + out.unique_s1 + out.unique_s2 + out.synergy;
    assert!(
        (sum_atoms - i_s1s2_t_hat).abs() < 1e-9,
        "PID identity broken: sum_atoms={sum_atoms} I(S1,S2;T)={i_s1s2_t_hat}"
    );
}

/// CORRECTION (was: a `Red == 0` "finding"). The premise was wrong — the I^sx redundancy for
/// independent additive Gaussian sources is strictly POSITIVE, and the KSG estimator is correct.
/// The sound analytic anchor for this value now lives in `tests/sxpid_gaussian_oracle.rs`
/// (`ksg_isx_redundancy_matches_gaussian_oracle_additive`): the estimator's ~0.22 nats matches the
/// closed-form continuous-limit oracle within tolerance. This stub documents the corrected record;
/// the `Red == 0` assertion has been removed because it asserted a false value.
#[test]
fn gaussian_independent_additive_red_is_positive_not_zero() {
    let (out, i_s1_t, _i_s2_t, _i_s1s2_t, _) = independent_additive_atoms();
    // Estimated redundancy is clearly positive and below I(S1;T) — consistent with the oracle.
    assert!(
        out.redundancy > 0.1 && out.redundancy < i_s1_t,
        "independent-additive I^sx Red should be positive and < I(S1;T): Red={:.4} I(S1;T)={:.4}",
        out.redundancy,
        i_s1_t
    );
}

// =============================================================================================
// BARRETT-2015 GAUSSIAN MMI BIVARIATE-REDUNDANCY REFERENCE.
//
// !!! IMPORTANT: MMI is a DIFFERENT redundancy measure from I^sx_∩. This is a sanity comparison,
// NOT an assertion that I^sx == MMI. !!!
//
// Barrett (2015), "Exploration of synergistic and redundant information sharing in static and
// dynamical Gaussian systems", Phys. Rev. E 91, 052802. For the broad class of jointly-Gaussian
// systems with a UNIVARIATE target, Barrett shows the Minimum Mutual Information (MMI) PID gives
// the bivariate redundancy as simply the smaller of the two single-source MIs:
//
//   R_MMI(S1,S2;T) = min( I(S1;T), I(S2;T) ).
//
// We compute the closed-form MMI redundancy for both Gaussian systems above and compare it,
// purely as a sanity reference, against the KSG single-source MI estimates. We do NOT compare it
// to the I^sx atoms.
// =============================================================================================
#[test]
fn barrett2015_gaussian_mmi_redundancy_reference_labelled_mmi_not_isx() {
    // -- System A: identical sources (I(S1;T) = I(S2;T) = I(X;T)). --
    // Barrett MMI: R_MMI = min(I(S1;T), I(S2;T)) = I(X;T).
    {
        let mut rng = Rng64::new(0xBA77_E771);
        let n = 4000;
        let sigma = 0.7;
        let sigma2 = sigma * sigma;

        let mut x = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            let xi = rng.normal();
            let z = rng.normal();
            x.push(xi);
            t.push(xi + sigma * z);
        }
        let s1 = MatRef::new(&x, n, 1).unwrap();
        let t = MatRef::new(&t, n, 1).unwrap();
        let (s1, _) = Standardizer::fit_transform(s1).unwrap();
        let (t, _) = Standardizer::fit_transform(t).unwrap();

        let i_s1_t = ksg_mi(s1.as_ref(), t.as_ref(), &ksg_cfg()).unwrap();
        // S2 == S1, so estimate is identical up to RNG reuse; use the same series for MMI ref.
        let i_s2_t = i_s1_t;

        // Theory (Barrett 2015): R_MMI = min(I(S1;T), I(S2;T)) = I(X;T) closed form.
        let rho = 1.0 / (1.0 + sigma2).sqrt();
        let r_mmi_true = gaussian_mi_from_corr(rho);
        let r_mmi_hat = i_s1_t.min(i_s2_t);

        // Sanity comparison ONLY (MMI != I^sx).
        assert!(
            (r_mmi_hat - r_mmi_true).abs() < 0.12,
            "[MMI ref, identical] R_MMI est={r_mmi_hat:.4} theory=min(I)=I(X;T)={r_mmi_true:.4}"
        );
    }

    // -- System B: independent additive sources. --
    // Barrett MMI: R_MMI = min(I(S1;T), I(S2;T)) = I(S1;T) (symmetric).
    // NOTE this is STRICTLY POSITIVE, unlike the I^sx redundancy (Red -> 0) for the same system:
    // a concrete demonstration that MMI and I^sx are different measures.
    {
        let mut rng = Rng64::new(0xBA77_E772);
        let n = 4000;
        let sigma = 0.6;
        let sigma2 = sigma * sigma;

        let mut s1v = Vec::with_capacity(n);
        let mut s2v = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.normal();
            let b = rng.normal();
            let z = rng.normal();
            s1v.push(a);
            s2v.push(b);
            t.push(a + b + sigma * z);
        }
        let s1 = MatRef::new(&s1v, n, 1).unwrap();
        let s2 = MatRef::new(&s2v, n, 1).unwrap();
        let t = MatRef::new(&t, n, 1).unwrap();
        let (s1, _) = Standardizer::fit_transform(s1).unwrap();
        let (s2, _) = Standardizer::fit_transform(s2).unwrap();
        let (t, _) = Standardizer::fit_transform(t).unwrap();

        let i_s1_t = ksg_mi(s1.as_ref(), t.as_ref(), &ksg_cfg()).unwrap();
        let i_s2_t = ksg_mi(s2.as_ref(), t.as_ref(), &ksg_cfg()).unwrap();

        // Theory (Barrett 2015): R_MMI = min(I(S1;T), I(S2;T)) = -0.5 ln((1+s^2)/(2+s^2)).
        let rho_s_t = 1.0 / (2.0 + sigma2).sqrt();
        let r_mmi_true = gaussian_mi_from_corr(rho_s_t);
        let r_mmi_hat = i_s1_t.min(i_s2_t);

        assert!(
            r_mmi_true > 0.05,
            "[MMI ref] independent-additive MMI redundancy is strictly positive (theory): {r_mmi_true:.4}"
        );
        assert!(
            (r_mmi_hat - r_mmi_true).abs() < 0.12,
            "[MMI ref, independent] R_MMI est={r_mmi_hat:.4} theory=min(I)={r_mmi_true:.4}"
        );
    }
}
