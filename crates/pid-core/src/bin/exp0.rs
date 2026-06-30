use pid_core::{
    average_degree_of_redundancy, average_degree_of_vulnerability, bootstrap_rows_stats,
    co_information_pairwise, concat_horiz, distance_concentration_stats,
    intrinsic_dimension_levina_bickel, isx_redundancy, ksg_mi, ksg_mi_concat_xy,
    permutation_rows_pvalue, BootstrapConfig, DistanceConcentrationConfig, HashProjector,
    IntrinsicDimConfig, IsxConfig, IsxMethod, KsgConfig, MatRef, Metric, NegativeHandling,
    PcaProjector, RowResampleScheme, Standardizer,
};
use pid_runlog::{RunLogEvent, RunLogWriter, RunStatus, RUN_LOG_SCHEMA_VERSION};
use serde_json::json;
use std::fs::File;
use std::io::{self, Write};

#[derive(Debug, Clone)]
struct Args {
    csv: bool,
    seeds: usize,
    strict_gate: bool,
    strict_band: bool,
    summary_json: Option<String>,
    runlog: Option<String>,
    uncertainty: UncertaintyConfig,
}

/// Opt-in uncertainty-quantification configuration for the Exp0 gate.
///
/// Both `n_boot` and `n_perm` default to 0 (disabled), which keeps the default
/// runner output byte-for-byte identical to the pre-uncertainty behaviour (the
/// CI smoke path and the runlog unit tests rely on this). When enabled, the
/// runner adds block-subsample bootstrap CIs and single-source permutation
/// p-values on the d=`uncertainty_dim` cases and folds preregistered
/// ground-truth checks into the GO/PIVOT/NO-GO verdict.
#[derive(Debug, Clone, Copy)]
struct UncertaintyConfig {
    /// Number of subsample-bootstrap resamples (0 disables bootstrap CIs).
    n_boot: usize,
    /// Number of permutations for single-source null tests (0 disables them).
    n_perm: usize,
    /// Moving-block length for the resamplers (1 = i.i.d., correct for these
    /// non-temporal synthetic scenarios).
    block_size: usize,
    /// Significance level for CIs and permutation decisions.
    alpha: f64,
    /// Base seed for the resamplers (kept separate from the data seeds).
    seed: u64,
}

impl UncertaintyConfig {
    fn enabled(&self) -> bool {
        self.n_boot > 0 || self.n_perm > 0
    }
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            n_boot: 0,
            n_perm: 0,
            block_size: 1,
            alpha: 0.05,
            seed: 0xC0FFEE,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CaseCommon<'a> {
    csv: bool,
    n: usize,
    ksg_cfg: &'a KsgConfig,
    hash_project_to: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct CaseSpec<'a> {
    name: &'a str,
    d: usize,
    seed: u64,
}

#[derive(Debug, Clone, Copy)]
struct Exp0RunConfig<'a> {
    n: usize,
    k: usize,
    dims: &'a [usize],
    seeds: &'a [u64],
    hash_project_to: Option<usize>,
}

#[derive(Debug)]
enum Exp0Error {
    Pid(pid_core::PidError),
    Io(io::Error),
    RunLog(anyhow::Error),
    StrictGate(String),
    Config(String),
}

impl From<pid_core::PidError> for Exp0Error {
    fn from(value: pid_core::PidError) -> Self {
        Self::Pid(value)
    }
}

impl From<io::Error> for Exp0Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<anyhow::Error> for Exp0Error {
    fn from(value: anyhow::Error) -> Self {
        Self::RunLog(value)
    }
}

fn main() {
    let args = match parse_args() {
        Ok(Some(a)) => a,
        Ok(None) => {
            let mut out = io::BufWriter::new(io::stdout());
            if let Err(e) = print_usage(&mut out) {
                // If someone does `exp0 --help | head`, avoid panicking.
                if e.kind() == io::ErrorKind::BrokenPipe {
                    return;
                }
                eprintln!("exp0: failed to write help: {e}");
            }
            return;
        }
        Err(msg) => {
            eprintln!("exp0: {msg}");
            eprintln!();
            let mut out = io::BufWriter::new(io::stderr());
            let _ = print_usage(&mut out);
            std::process::exit(2);
        }
    };

    let mut out = io::BufWriter::new(io::stdout());
    if let Err(err) = run(&mut out, args) {
        match err {
            Exp0Error::Io(e) if e.kind() == io::ErrorKind::BrokenPipe => (),
            Exp0Error::Pid(e) => {
                eprintln!("exp0: estimator error: {e}");
                std::process::exit(1);
            }
            Exp0Error::Io(e) => {
                eprintln!("exp0: IO error: {e}");
                std::process::exit(1);
            }
            Exp0Error::RunLog(e) => {
                eprintln!("exp0: run-log error: {e}");
                std::process::exit(1);
            }
            Exp0Error::StrictGate(status) => {
                eprintln!("exp0: --strict-gate: gate status is {status}, expected GO");
                std::process::exit(3);
            }
            Exp0Error::Config(msg) => {
                eprintln!("exp0: {msg}");
                std::process::exit(2);
            }
        }
    }
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut csv = false;
    let mut seeds = 3usize;
    let mut strict_gate = false;
    let mut strict_band = false;
    let mut summary_json = None;
    let mut runlog = None;
    let mut uncertainty = UncertaintyConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--csv" => csv = true,
            "--strict-gate" => strict_gate = true,
            "--strict-band" => strict_band = true,
            "--seeds" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--seeds requires a positive integer".to_string())?;
                seeds = raw
                    .parse::<usize>()
                    .map_err(|_| "--seeds requires a positive integer".to_string())?;
                if seeds == 0 {
                    return Err("--seeds requires a positive integer".to_string());
                }
            }
            "--summary-json" => {
                summary_json = Some(
                    args.next()
                        .ok_or_else(|| "--summary-json requires a path".to_string())?,
                );
            }
            "--runlog" => {
                runlog = Some(
                    args.next()
                        .ok_or_else(|| "--runlog requires a path".to_string())?,
                );
            }
            "--bootstrap" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--bootstrap requires a non-negative integer".to_string())?;
                uncertainty.n_boot = raw
                    .parse::<usize>()
                    .map_err(|_| "--bootstrap requires a non-negative integer".to_string())?;
            }
            "--permutation" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--permutation requires a non-negative integer".to_string())?;
                uncertainty.n_perm = raw
                    .parse::<usize>()
                    .map_err(|_| "--permutation requires a non-negative integer".to_string())?;
            }
            "--block-size" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--block-size requires a positive integer".to_string())?;
                uncertainty.block_size = raw
                    .parse::<usize>()
                    .map_err(|_| "--block-size requires a positive integer".to_string())?;
                if uncertainty.block_size == 0 {
                    return Err("--block-size requires a positive integer".to_string());
                }
            }
            "--alpha" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--alpha requires a float in (0,1)".to_string())?;
                uncertainty.alpha = raw
                    .parse::<f64>()
                    .map_err(|_| "--alpha requires a float in (0,1)".to_string())?;
                if !(uncertainty.alpha > 0.0 && uncertainty.alpha < 1.0) {
                    return Err("--alpha requires a float in (0,1)".to_string());
                }
            }
            "--help" | "-h" => return Ok(None),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Some(Args {
        csv,
        seeds,
        strict_gate,
        strict_band,
        summary_json,
        runlog,
        uncertainty,
    }))
}

fn print_usage(out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "Usage: exp0 [--csv] [--seeds N] [--summary-json PATH] [--runlog PATH]"
    )?;
    writeln!(
        out,
        "            [--bootstrap N] [--permutation N] [--block-size N] [--alpha F]"
    )?;
    writeln!(out, "            [--strict-band] [--strict-gate]")?;
    writeln!(out)?;
    writeln!(out, "  --csv   Emit machine-readable CSV (two tables).")?;
    writeln!(
        out,
        "  --seeds N   Run N deterministic seeds per case (default: 3)."
    )?;
    writeln!(
        out,
        "  --summary-json PATH   Write gate summary metadata as JSON."
    )?;
    writeln!(
        out,
        "  --runlog PATH   Write canonical run-log events for the Exp0 summary."
    )?;
    writeln!(
        out,
        "  --bootstrap N   Subsample-bootstrap CIs (N resamples) on the d={UNCERTAINTY_DIM} cases."
    )?;
    writeln!(
        out,
        "  --permutation N   Single-source permutation null tests (N permutations) on the d={UNCERTAINTY_DIM} cases."
    )?;
    writeln!(
        out,
        "  --block-size N   Moving-block length for resamplers (default: 1 = i.i.d.)."
    )?;
    writeln!(
        out,
        "  --alpha F   Significance level for CIs / permutation decisions (default: 0.05)."
    )?;
    writeln!(
        out,
        "  --strict-band   Also run the curated band (analytic d=1 Gaussian MI gate at n={STRICT_BAND_GATE_N},"
    )?;
    writeln!(
        out,
        "                  plus an informational d<=8 scenario diagnostic sweep)."
    )?;
    writeln!(
        out,
        "  --strict-gate   Exit with code 3 unless the gate is GO. Enforced on the curated"
    )?;
    writeln!(
        out,
        "                  low-dimension band (implies --strict-band), NOT the default high-d sweep,"
    )?;
    writeln!(
        out,
        "                  whose PIVOT/NO-GO at high dimension is the expected, informative outcome."
    )?;
    writeln!(out, "  -h, --help   Show this help.")?;
    Ok(())
}

fn make_seeds(n: usize) -> Vec<u64> {
    (0..n)
        .map(|i| 42u64.wrapping_add((i as u64).wrapping_mul(1_000_003)))
        .collect()
}

/// Ambient dimension at which uncertainty quantification (bootstrap/permutation)
/// is run. The smallest dimension is the regime where continuous kNN estimation is
/// most plausible; running UQ there gives the gate its best chance of healthy
/// recovery, so a failure here is a strong NO-GO signal rather than an expected
/// curse-of-dimensionality artefact.
const UNCERTAINTY_DIM: usize = 10;

// ---------------------------------------------------------------------------
// Curated strict band (--strict-band / --strict-gate target)
// ---------------------------------------------------------------------------
//
// The DEFAULT sweep deliberately runs to dimension 256 at n=500, entering regimes
// where continuous kNN MI is known to break down (Kraskov 2004 §IV; Gao 2015): a
// PIVOT/NO-GO verdict on the full default sweep is the EXPECTED, informative outcome,
// not a build failure (see AGENTS.md "exp0 is a diagnostic gate"). It must therefore
// never be the target of a hard pass/fail gate.
//
// `--strict-gate` instead enforces GO on a CURATED BAND where GO is legitimately
// expected AND is checked against an ANALYTIC closed form (not against the estimator's
// own output — AGENTS.md: "a numerical result must be justified by an analytic closed
// form or a cited paper, NEVER tuned to match the estimator").
//
// WHAT THE GATE CHECKS: a small grid of jointly-GAUSSIAN systems at d=1 (pure signal, no
// noise dimensions) and n=500, the regime where the KSG estimator is validated and
// accurate (cf. the strong-dependence sweep and tests/ksg.rs / tests/gaussian_pid_atoms.rs
// at d=1, moderate sigma). The pass/fail items are the three MEASURE-INDEPENDENT mutual
// information terms I(S1;T), I(S2;T), I(S1,S2;T), each compared to its Cover–Thomas
// Gaussian closed form within the scale-aware tolerance used elsewhere in the gate. GO ⇔
// every MI term across the grid is within tolerance. These terms have a genuine analytic
// ground truth and do not depend on which redundancy MEASURE is chosen.
//
// WHY NOT GATE ON THE FOUR SYNTHETIC SCENARIOS AT d<=8: empirically they are NOT a GO
// regime, for two reasons that are genuine, reported FINDINGS — not bugs and not things
// to tune away:
//     * `independent_additive`: this scenario compares the estimated atoms against an
//       MMI/zero-redundancy expectation, which I^sx does not satisfy. NOTE (corrected): the true
//       I^sx redundancy here is genuinely POSITIVE (~0.2 nats), not zero — confirmed against a
//       closed-form oracle in tests/sxpid_gaussian_oracle.rs — so the EhrlichKsg estimate is
//       CORRECT, and the mismatch is a measure/convention difference, NOT estimator over-
//       attribution (an earlier comment here mis-stated this as a bias).
//   * `redundant_copy`/`unique_s1` carry very high MI; KSG underestimates the JOINT
//     (concatenated-source) MI relative to a marginal, tripping the monotonicity counter
//     — the well-known KSG joint-space bias under strong dependence (Kraskov 2004 §IV;
//     Gao 2015). The d-1 pure-noise source coordinates also dominate the Chebyshev
//     neighbour structure and collapse the estimate.
// Gating GO on those would require loosening the checks, which the conventions forbid. The
// scenarios are still RUN at d in STRICT_BAND_DIAG_DIMS as an INFORMATIONAL diagnostic
// (printed, NOT gated) so the documented d<=8 sweep is exercised and the findings surfaced.
const STRICT_BAND_N: usize = 500;
/// Sample size for the ANALYTIC d=1 Gaussian gate. Larger than `STRICT_BAND_N` because the gate
/// asserts recovery of the closed-form MI terms within the scale-aware noise floor (0.05 nats),
/// which requires KSG's low-bias regime: at n=500 the finite-sample bias (~0.06 nats at moderate
/// MI) sits right at that floor, whereas n=4000 is the validated atom-recovery regime used by
/// tests/gaussian_pid_atoms.rs. Using the n where the estimator is accurate keeps the gate honest
/// (we do NOT loosen the tolerance to accommodate finite-sample bias).
const STRICT_BAND_GATE_N: usize = 4000;
/// Informational (NON-gating) low-dimension scenario sweep run alongside the analytic gate so
/// the documented d<=8 scenarios are still exercised and their estimator-hostility surfaced.
const STRICT_BAND_DIAG_DIMS: [usize; 3] = [2, 4, 8];
const STRICT_BAND_SEEDS: usize = 3;
/// The d=1 jointly-Gaussian gate grid as `(a, b, c)` coefficients of `T = a*S1 + b*S2 + c*Z`,
/// with S1,S2,Z ~ N(0,1) independent. Moderate MI keeps every term inside KSG's accurate
/// regime; the mix spans redundant-leaning (a==b), unique-leaning (a > b), and balanced cases
/// so the gate is non-trivial. See `gaussian_atom_truth` for the closed-form MI/atoms.
const STRICT_BAND_GAUSS_GRID: [(f64, f64, f64); 3] =
    [(1.0, 1.0, 1.0), (1.0, 0.3, 1.0), (0.7, 0.7, 1.0)];

/// The four synthetic scenarios, with their preregistered ground-truth marginal
/// informativeness. A source is "marginally informative" iff `I(source; T) > 0` in
/// the data-generating process — independent of any estimator.
///
/// This table is the falsifiable contract the permutation null tests check:
/// the KSG-based permutation test must call a source significant iff that source
/// is marginally informative by construction.
const SCENARIOS: [&str; 4] = [
    "independent_additive",
    "redundant_copy",
    "unique_s1",
    "xor_like",
];

/// Returns `(s1_informative, s2_informative)` ground truth for a scenario.
fn marginal_truth(scenario: &str) -> (bool, bool) {
    match scenario {
        // T = s1[0] + s2[0] + noise → both sources marginally informative.
        "independent_additive" => (true, true),
        // s1[0], s2[0] are noisy copies of T → both marginally informative.
        "redundant_copy" => (true, true),
        // T = s1[0] + noise → only s1 marginally informative.
        "unique_s1" => (true, false),
        // T = sign(s1[0]*s2[0]) → neither source marginally informative (synergy only).
        "xor_like" => (false, false),
        _ => unreachable!("unknown scenario: {scenario}"),
    }
}

/// Per-scenario uncertainty result at `UNCERTAINTY_DIM`.
#[derive(Debug, Clone)]
struct ScenarioUncertainty {
    name: &'static str,
    /// Bootstrap CIs for [I(S1;T), I(S2;T), I(S1,S2;T), Red_ehrlich], if enabled.
    boot: Option<BootQuad>,
    /// Permutation p-value for shuffle-S1 / statistic I(S1;T), if enabled.
    perm_s1_p: Option<f64>,
    /// Permutation p-value for shuffle-S2 / statistic I(S2;T), if enabled.
    perm_s2_p: Option<f64>,
    /// Number of permutations that produced a finite statistic (S1 test).
    perm_s1_valid: usize,
    /// Number of permutations that produced a finite statistic (S2 test).
    perm_s2_valid: usize,
}

/// Bootstrap CI quad for the four key MI quantities.
#[derive(Debug, Clone, Copy)]
struct BootQuad {
    i1: CiTriple,
    i2: CiTriple,
    i12: CiTriple,
    red: CiTriple,
}

#[derive(Debug, Clone, Copy)]
struct CiTriple {
    point: f64,
    ci_low: f64,
    ci_high: f64,
    n_valid: usize,
}

/// Aggregate uncertainty summary across scenarios, with the derived gate checks.
#[derive(Debug, Clone, Default)]
struct UncertaintySummary {
    enabled: bool,
    n_boot: usize,
    n_perm: usize,
    block_size: usize,
    subsample_len: usize,
    alpha: f64,
    scenarios: Vec<ScenarioUncertainty>,
    /// Number of preregistered permutation marginal-significance checks performed.
    permutation_checks: usize,
    /// Number of those checks where the estimator agreed with ground truth.
    permutation_agreements: usize,
    /// Scenarios where the joint-MI bootstrap failed on > half the resamples
    /// (an estimator-instability signal at the most favourable dimension).
    bootstrap_instabilities: usize,
}

/// The MI/redundancy statistic vector used for bootstrap CIs:
/// `[I(S1;T), I(S2;T), I(S1,S2;T), Red_ehrlich(S1,S2;T)]`.
fn uncertainty_stat_vec(mats: &[MatRef<'_>], ksg_cfg: &KsgConfig) -> pid_core::PidResult<Vec<f64>> {
    let s1 = mats[0];
    let s2 = mats[1];
    let t = mats[2];
    let i1 = ksg_mi(s1, t, ksg_cfg)?;
    let i2 = ksg_mi(s2, t, ksg_cfg)?;
    let i12 = ksg_mi_concat_xy(s1, s2, t, ksg_cfg)?;
    let red = isx_redundancy(
        s1,
        s2,
        t,
        &IsxConfig {
            k: ksg_cfg.k,
            metric: ksg_cfg.metric,
            tie_epsilon: ksg_cfg.tie_epsilon,
            method: IsxMethod::EhrlichKsg,
        },
    )?;
    Ok(vec![i1, i2, i12, red])
}

/// Compute opt-in uncertainty quantification for all scenarios at `UNCERTAINTY_DIM`.
///
/// Determinism: uses a single fixed data seed (`make_seeds(1)[0]`) and the
/// resampler seed from `cfg`, so output is reproducible and runtime is bounded
/// independent of `--seeds`.
fn compute_uncertainty(
    n: usize,
    ksg_cfg: &KsgConfig,
    cfg: UncertaintyConfig,
) -> Result<UncertaintySummary, Exp0Error> {
    let data_seed = make_seeds(1)[0];
    let noise_std = 0.05;
    let d = UNCERTAINTY_DIM;
    // The subsample spans half the rows in whole blocks, so a block larger than n/2 leaves
    // zero whole blocks (subsample_len == 0) and surfaces an opaque downstream estimator
    // error. Reject it here with a clear message. (block_size == 0 is already rejected at
    // parse time, which also avoids the division-by-zero below.)
    if cfg.block_size > n / 2 {
        return Err(Exp0Error::Config(format!(
            "--block-size must be <= n/2 (= {}) so the subsample spans at least one whole block",
            n / 2
        )));
    }
    // Subsample length: half the rows, in whole blocks. This is the
    // Politis–Romano subsampling regime; the resulting CI is conservative
    // (overstates n-sample uncertainty by ~sqrt(2)) but valid for kNN MI, which
    // a naive with-replacement bootstrap is not (see pipeline::RowResampleScheme).
    let subsample_len = ((n / 2) / cfg.block_size) * cfg.block_size;

    let mut summary = UncertaintySummary {
        enabled: true,
        n_boot: cfg.n_boot,
        n_perm: cfg.n_perm,
        block_size: cfg.block_size,
        subsample_len,
        alpha: cfg.alpha,
        ..Default::default()
    };

    for &name in &SCENARIOS {
        let (s1, s2, t) = match name {
            "independent_additive" => gen_independent_additive(n, d, noise_std, data_seed),
            "redundant_copy" => gen_redundant_copy(n, d, noise_std, data_seed),
            "unique_s1" => gen_unique_s1(n, d, noise_std, data_seed),
            "xor_like" => gen_xor_like(n, d, noise_std, data_seed),
            _ => unreachable!(),
        };
        let s1 = MatRef::new(&s1, n, d)?;
        let s2 = MatRef::new(&s2, n, d)?;
        let t = MatRef::new(&t, n, 1)?;
        let (s1z, _) = Standardizer::fit_transform(s1)?;
        let (s2z, _) = Standardizer::fit_transform(s2)?;
        let (tz, _) = Standardizer::fit_transform(t)?;
        let mats: [MatRef<'_>; 3] = [s1z.as_ref(), s2z.as_ref(), tz.as_ref()];

        let mut scen = ScenarioUncertainty {
            name,
            boot: None,
            perm_s1_p: None,
            perm_s2_p: None,
            perm_s1_valid: 0,
            perm_s2_valid: 0,
        };

        if cfg.n_boot > 0 {
            let boot_cfg = BootstrapConfig {
                n_boot: cfg.n_boot,
                block_size: cfg.block_size,
                seed: cfg.seed,
                alpha: cfg.alpha,
            };
            let scheme = RowResampleScheme::Subsample { subsample_len };
            let res = bootstrap_rows_stats(&mats, &boot_cfg, scheme, |m| {
                uncertainty_stat_vec(m, ksg_cfg)
            })
            .map_err(Exp0Error::Pid)?;
            let to_triple = |s: &pid_core::RowBootstrapStat| CiTriple {
                point: s.point_estimate,
                ci_low: s.ci_low,
                ci_high: s.ci_high,
                n_valid: s.n_valid,
            };
            let quad = BootQuad {
                i1: to_triple(&res.stats[0]),
                i2: to_triple(&res.stats[1]),
                i12: to_triple(&res.stats[2]),
                red: to_triple(&res.stats[3]),
            };
            // Instability: joint MI bootstrap failed on > half the resamples.
            if quad.i12.n_valid * 2 < cfg.n_boot {
                summary.bootstrap_instabilities += 1;
            }
            scen.boot = Some(quad);
        }

        if cfg.n_perm > 0 {
            // Shuffle S1 (index 0); statistic = I(S1;T).
            let perm_s1 = permutation_rows_pvalue(&mats, 0, cfg.n_perm, cfg.seed, |m| {
                ksg_mi(m[0], m[2], ksg_cfg)
            })
            .map_err(Exp0Error::Pid)?;
            // Shuffle S2 (index 1); statistic = I(S2;T).
            let perm_s2 =
                permutation_rows_pvalue(&mats, 1, cfg.n_perm, cfg.seed.wrapping_add(1), |m| {
                    ksg_mi(m[1], m[2], ksg_cfg)
                })
                .map_err(Exp0Error::Pid)?;

            let (truth_s1, truth_s2) = marginal_truth(name);
            // A check "agrees" iff the significance decision matches ground truth.
            for (p, valid, truth) in [
                (perm_s1.p_value, perm_s1.n_valid, truth_s1),
                (perm_s2.p_value, perm_s2.n_valid, truth_s2),
            ] {
                summary.permutation_checks += 1;
                // Only count a finite p-value; a degenerate test (all resamples
                // failed) is recorded as a non-agreement (it cannot confirm truth).
                if valid > 0 && p.is_finite() {
                    let significant = p < cfg.alpha;
                    if significant == truth {
                        summary.permutation_agreements += 1;
                    }
                }
            }
            scen.perm_s1_p = Some(perm_s1.p_value);
            scen.perm_s2_p = Some(perm_s2.p_value);
            scen.perm_s1_valid = perm_s1.n_valid;
            scen.perm_s2_valid = perm_s2.n_valid;
        }

        summary.scenarios.push(scen);
    }

    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn write_summary_json(
    path: &str,
    gates: &GateSummary,
    n: usize,
    k: usize,
    dims: &[usize],
    seeds: &[u64],
    hash_project_to: Option<usize>,
    uncertainty: Option<&UncertaintySummary>,
) -> io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut file = File::create(path)?;
    let config_hash = config_hash(n, k, dims, seeds, hash_project_to);
    writeln!(file, "{{")?;
    writeln!(file, "  \"config_hash\": \"{config_hash:016x}\",")?;
    writeln!(file, "  \"n\": {n},")?;
    writeln!(file, "  \"k\": {k},")?;
    writeln!(file, "  \"dims\": {},", json_usize_array(dims))?;
    writeln!(file, "  \"seeds\": {},", json_u64_array(seeds))?;
    match hash_project_to {
        Some(v) => writeln!(file, "  \"hash_project_to\": {v},")?,
        None => writeln!(file, "  \"hash_project_to\": null,")?,
    }
    writeln!(file, "  \"case_results\": {},", gates.case_results)?;
    writeln!(file, "  \"red_zero_checks\": {},", gates.red_zero_checks)?;
    writeln!(file, "  \"red_zero_passes\": {},", gates.red_zero_passes)?;
    writeln!(
        file,
        "  \"monotonicity_violations\": {},",
        gates.monotonicity_violations
    )?;
    writeln!(
        file,
        "  \"invariant_violations\": {},",
        gates.invariant_violations
    )?;
    writeln!(
        file,
        "  \"geometry_warnings\": {},",
        gates.geometry_warnings
    )?;
    // Uncertainty block is emitted only when UQ ran, keeping default output identical.
    if let Some(u) = uncertainty {
        writeln!(file, "  \"status\": \"{}\",", gates.status())?;
        let value = uncertainty_json(u);
        let rendered = serde_json::to_string_pretty(&value)
            .unwrap_or_else(|_| "{}".to_string())
            .replace('\n', "\n  ");
        writeln!(file, "  \"uncertainty\": {rendered}")?;
    } else {
        writeln!(file, "  \"status\": \"{}\"", gates.status())?;
    }
    writeln!(file, "}}")?;
    Ok(())
}

/// Build the JSON value describing an uncertainty run (used by the summary JSON and
/// as the structured payload for run-log evaluation events).
fn uncertainty_json(u: &UncertaintySummary) -> serde_json::Value {
    let scenarios: Vec<serde_json::Value> = u
        .scenarios
        .iter()
        .map(|s| {
            let (truth_s1, truth_s2) = marginal_truth(s.name);
            let boot = s.boot.map(|b| {
                json!({
                    "i1": ci_json(&b.i1),
                    "i2": ci_json(&b.i2),
                    "i12": ci_json(&b.i12),
                    "red_ehrlich": ci_json(&b.red),
                })
            });
            json!({
                "name": s.name,
                "truth_s1_informative": truth_s1,
                "truth_s2_informative": truth_s2,
                "perm_s1_p": s.perm_s1_p,
                "perm_s2_p": s.perm_s2_p,
                "perm_s1_valid": s.perm_s1_valid,
                "perm_s2_valid": s.perm_s2_valid,
                "bootstrap": boot,
            })
        })
        .collect();
    json!({
        "dim": UNCERTAINTY_DIM,
        "n_boot": u.n_boot,
        "n_perm": u.n_perm,
        "block_size": u.block_size,
        "subsample_len": u.subsample_len,
        "subsample_scheme": "politis_romano_without_replacement",
        "alpha": u.alpha,
        "permutation_checks": u.permutation_checks,
        "permutation_agreements": u.permutation_agreements,
        "bootstrap_instabilities": u.bootstrap_instabilities,
        "scenarios": scenarios,
    })
}

fn ci_json(c: &CiTriple) -> serde_json::Value {
    json!({
        "point": json_float(c.point),
        "ci_low": json_float(c.ci_low),
        "ci_high": json_float(c.ci_high),
        "n_valid": c.n_valid,
    })
}

/// JSON-safe float: non-finite values (NaN/inf can arise from degenerate
/// resamples) serialize to `null` rather than producing invalid JSON.
fn json_float(x: f64) -> serde_json::Value {
    if x.is_finite() {
        json!(x)
    } else {
        serde_json::Value::Null
    }
}

fn write_exp0_runlog(
    path: &str,
    summary_json_path: Option<&str>,
    gates: &GateSummary,
    config: Exp0RunConfig<'_>,
    uncertainty: Option<&UncertaintySummary>,
) -> Result<(), Exp0Error> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let config_json = json!({
        "experiment": "exp0",
        "n": config.n,
        "k": config.k,
        "dims": config.dims,
        "seeds": config.seeds,
        "hash_project_to": config.hash_project_to,
        // Build-provenance block: folding this into config_json means the SHA-256 config_hash
        // certifies the exact binary (crate version + source revision + toolchain + feature set)
        // that produced the run, not merely its numeric parameters.
        "build_provenance": build_provenance(),
    });
    let config_hash = pid_runlog::canonical_json_hash(&config_json)?;
    let mut writer = RunLogWriter::create(path)?;
    writer.append(&RunLogEvent::RunStarted {
        schema_version: RUN_LOG_SCHEMA_VERSION,
        run_id: "exp0-rust-quick-run".to_string(),
        timestamp_ns: 0,
        config_hash: config_hash.clone(),
        metadata: [
            ("source".to_string(), "pid-core-exp0".to_string()),
            ("status".to_string(), gates.status().to_string()),
        ]
        .into_iter()
        .collect(),
    })?;
    writer.append(&RunLogEvent::ConfigLogged {
        timestamp_ns: 0,
        config_hash,
        config: config_json,
    })?;
    write_exp0_metric_events(&mut writer, gates)?;
    if let Some(u) = uncertainty {
        write_exp0_uncertainty_events(&mut writer, u)?;
    }
    if let Some(summary_path) = summary_json_path {
        writer.append(&RunLogEvent::ArtifactLogged {
            timestamp_ns: 8,
            name: "exp0_summary_json".to_string(),
            kind: "summary_json".to_string(),
            uri: summary_path.to_string(),
            sha256: pid_runlog::sha256_file(summary_path).ok(),
            metadata: [("status".to_string(), gates.status().to_string())]
                .into_iter()
                .collect(),
        })?;
    }
    if gates.status() != "GO" {
        writer.append(&RunLogEvent::ErrorLogged {
            step: Some(8),
            timestamp_ns: 9,
            message: format!("Experiment 0 gate status: {}", gates.status()),
            recoverable: true,
        })?;
    }
    writer.append(&RunLogEvent::RunEnded {
        run_id: "exp0-rust-quick-run".to_string(),
        timestamp_ns: 10,
        status: RunStatus::Succeeded,
        message: Some(format!("Exp0 scientific gate status: {}", gates.status())),
    })?;
    writer.flush()?;
    Ok(())
}

fn write_exp0_metric_events<W: Write>(
    writer: &mut RunLogWriter<W>,
    gates: &GateSummary,
) -> Result<(), Exp0Error> {
    for (idx, (name, value)) in [
        ("exp0.case_results", gates.case_results),
        ("exp0.red_zero_checks", gates.red_zero_checks),
        ("exp0.red_zero_passes", gates.red_zero_passes),
        (
            "exp0.monotonicity_violations",
            gates.monotonicity_violations,
        ),
        ("exp0.invariant_violations", gates.invariant_violations),
        ("exp0.geometry_warnings", gates.geometry_warnings),
        ("exp0.status_code", gates.status_code()),
    ]
    .into_iter()
    .enumerate()
    {
        writer.append(&RunLogEvent::PidMetric {
            step: idx as u64,
            timestamp_ns: (idx + 1) as u64,
            name: name.to_string(),
            value: value as f64,
            metadata: [("status".to_string(), gates.status().to_string())]
                .into_iter()
                .collect(),
        })?;
    }
    Ok(())
}

/// Emit uncertainty results as `EvaluationMetric` events (kept distinct from the
/// `PidMetric` gate events so `pid_metrics` is unchanged). All events share a fixed
/// step/timestamp (step 7, ts 8) just past the 7 gate metrics (steps 0–6, ts 1–7),
/// so timestamps and steps stay nondecreasing ahead of the artifact/error/run-ended
/// tail. Non-finite statistics are skipped (run-log values must be finite for
/// replay), with the validity counts carried in the aggregate metrics and the
/// summary JSON.
fn write_exp0_uncertainty_events<W: Write>(
    writer: &mut RunLogWriter<W>,
    u: &UncertaintySummary,
) -> Result<(), Exp0Error> {
    const STEP: u64 = 7;
    const TS: u64 = 8;
    let base_meta = || -> std::collections::BTreeMap<String, String> {
        [("kind".to_string(), "uncertainty".to_string())]
            .into_iter()
            .collect()
    };
    let emit = |writer: &mut RunLogWriter<W>,
                name: String,
                value: f64,
                extra: Option<(&str, String)>|
     -> Result<(), Exp0Error> {
        let mut metadata = base_meta();
        if let Some((k, v)) = extra {
            metadata.insert(k.to_string(), v);
        }
        writer.append(&RunLogEvent::EvaluationMetric {
            step: STEP,
            timestamp_ns: TS,
            name,
            value,
            metadata,
        })?;
        Ok(())
    };

    emit(
        writer,
        "exp0.uncertainty.permutation_checks".to_string(),
        u.permutation_checks as f64,
        None,
    )?;
    emit(
        writer,
        "exp0.uncertainty.permutation_agreements".to_string(),
        u.permutation_agreements as f64,
        None,
    )?;
    emit(
        writer,
        "exp0.uncertainty.bootstrap_instabilities".to_string(),
        u.bootstrap_instabilities as f64,
        None,
    )?;
    emit(
        writer,
        "exp0.uncertainty.subsample_len".to_string(),
        u.subsample_len as f64,
        None,
    )?;

    for s in &u.scenarios {
        let (truth_s1, truth_s2) = marginal_truth(s.name);
        for (suffix, p, truth) in [
            ("perm_s1_p", s.perm_s1_p, truth_s1),
            ("perm_s2_p", s.perm_s2_p, truth_s2),
        ] {
            if let Some(p) = p {
                if p.is_finite() {
                    emit(
                        writer,
                        format!("exp0.uncertainty.{}.{}", s.name, suffix),
                        p,
                        Some(("truth_informative", truth.to_string())),
                    )?;
                }
            }
        }
        if let Some(b) = &s.boot {
            for (suffix, triple) in [
                ("i1", &b.i1),
                ("i2", &b.i2),
                ("i12", &b.i12),
                ("red_ehrlich", &b.red),
            ] {
                for (bound_name, bound) in [("ci_low", triple.ci_low), ("ci_high", triple.ci_high)]
                {
                    if bound.is_finite() {
                        emit(
                            writer,
                            format!("exp0.uncertainty.{}.{}_{}", s.name, suffix, bound_name),
                            bound,
                            Some(("n_valid", triple.n_valid.to_string())),
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn json_usize_array(values: &[usize]) -> String {
    let parts: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn json_u64_array(values: &[u64]) -> String {
    let parts: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[{}]", parts.join(","))
}

/// Build-provenance block: the crate version, source git commit (or `"unknown"` when git was
/// unavailable at build time), the rustc version that compiled the binary, and the enabled
/// feature set. Captured at compile time via `build.rs` (commit/rustc) and `cfg!` (features), so
/// the value is baked into the binary and is deterministic for a given build. Folding this into
/// `config_json` lets the run-log's `config_hash` certify the binary, not just its parameters.
fn build_provenance() -> serde_json::Value {
    // Enabled features, sorted for determinism (BTreeSet semantics via a sorted Vec).
    let mut features: Vec<&str> = Vec::new();
    if cfg!(feature = "parallel") {
        features.push("parallel");
    }
    features.sort_unstable();
    json!({
        "crate_version": env!("CARGO_PKG_VERSION"),
        "git_commit": env!("PID_CORE_GIT_COMMIT"),
        "rustc_version": env!("PID_CORE_RUSTC_VERSION"),
        "features": features,
    })
}

fn config_hash(
    n: usize,
    k: usize,
    dims: &[usize],
    seeds: &[u64],
    hash_project_to: Option<usize>,
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    mix_u64(&mut h, n as u64);
    mix_u64(&mut h, k as u64);
    mix_u64(&mut h, dims.len() as u64);
    for &d in dims {
        mix_u64(&mut h, d as u64);
    }
    mix_u64(&mut h, seeds.len() as u64);
    for &seed in seeds {
        mix_u64(&mut h, seed);
    }
    mix_u64(&mut h, hash_project_to.map_or(u64::MAX, |v| v as u64));
    h
}

fn mix_u64(h: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn run(out: &mut dyn Write, args: Args) -> Result<(), Exp0Error> {
    // Minimal Experiment 0 runner (Rust-side).
    //
    // This is intentionally small and brute-force; it exists to exercise the estimators end-to-end
    // on synthetic systems and to provide a place to iterate while building the full harness.

    let n = 500usize;
    let k = 3usize;
    let dims = [10usize, 64, 256];
    let hash_project_to = Some(64usize);
    let seeds = make_seeds(args.seeds);

    let ksg_cfg = KsgConfig {
        k,
        metric: Metric::Chebyshev,
        tie_epsilon: 0.0,
        negative_handling: NegativeHandling::ClampToZero,
    };

    if args.csv {
        write_case_csv_header(out)?;
    } else {
        writeln!(out, "Experiment 0 (Rust quick run)")?;
        writeln!(out, "n={n}, k={k}, dims={dims:?}, seeds={seeds:?}")?;
        writeln!(
            out,
            "project_to={hash_project_to:?} (projection baselines: hash + PCA; S1,S2 only)"
        )?;
        writeln!(out)?;
    }

    let mut gates = GateSummary::default();

    let common = CaseCommon {
        csv: args.csv,
        n,
        ksg_cfg: &ksg_cfg,
        hash_project_to,
    };
    for d in dims {
        for &seed in &seeds {
            for name in [
                "independent_additive",
                "redundant_copy",
                "unique_s1",
                "xor_like",
            ] {
                let res = run_case(out, common, CaseSpec { name, d, seed })?;
                gates.observe_case(name, d, res.metrics, res.diag);
            }
            if !common.csv {
                writeln!(out)?;
            }
        }
        if !common.csv {
            writeln!(out)?;
        }
    }

    if common.csv {
        writeln!(out)?;
        write_gaussian_csv_header(out)?;
    }
    run_gaussian_channel_strong_dependence_sweep(out, common.csv, 900, &ksg_cfg, 0x51A7_2026)?;

    // Opt-in uncertainty quantification at the most favourable dimension.
    let uncertainty = if args.uncertainty.enabled() {
        let u = compute_uncertainty(n, &ksg_cfg, args.uncertainty)?;
        gates.observe_uncertainty(&u);
        Some(u)
    } else {
        None
    };

    if !args.csv {
        if let Some(u) = uncertainty.as_ref() {
            print_uncertainty(out, u)?;
        }
        writeln!(out, "--- Experiment 0 Summary ---")?;
        gates.print(out)?;
    }

    if let Some(path) = args.summary_json.as_deref() {
        write_summary_json(
            path,
            &gates,
            n,
            k,
            &dims,
            &seeds,
            hash_project_to,
            uncertainty.as_ref(),
        )?;
    }
    if let Some(path) = args.runlog.as_deref() {
        write_exp0_runlog(
            path,
            args.summary_json.as_deref(),
            &gates,
            Exp0RunConfig {
                n,
                k,
                dims: &dims,
                seeds: &seeds,
                hash_project_to,
            },
            uncertainty.as_ref(),
        )?;
    }

    // Curated low-dimension band. `--strict-gate` enforces GO HERE (not on the default
    // high-d sweep, whose PIVOT/NO-GO is the documented, expected outcome). Requesting the
    // gate implies running the band; `--strict-band` runs+reports it without enforcing.
    let run_band = args.strict_band || args.strict_gate;
    if run_band {
        let band = run_strict_band(out, args.csv, &ksg_cfg)?;
        if !args.csv {
            writeln!(out, "--- Strict Band Summary (curated low-d) ---")?;
            band.print(out)?;
        }
        if args.strict_gate && band.status() != "GO" {
            return Err(Exp0Error::StrictGate(band.status().to_string()));
        }
    }

    Ok(())
}

/// Compute the curated band's GATING summary: the analytic d=1 Gaussian grid
/// (`STRICT_BAND_GAUSS_GRID` at `STRICT_BAND_GATE_N`). This is the only sweep `--strict-gate`
/// is allowed to enforce GO on, because (a) GO is legitimately expected there — d=1, moderate
/// MI is the KSG estimator's validated regime — and (b) the pass/fail items (the three
/// measure-independent MI terms) are checked against a closed-form analytic ground truth, not
/// the estimator's own output (see the `STRICT_BAND_*` rationale block). Kept cheap and separate
/// from the informational diagnostic so the gate can be unit-tested without the slow geometry pass.
fn strict_band_gate(
    out: &mut dyn Write,
    csv: bool,
    ksg_cfg: &KsgConfig,
) -> Result<GateSummary, Exp0Error> {
    let gate_n = STRICT_BAND_GATE_N;
    if !csv {
        writeln!(out)?;
        writeln!(
            out,
            "Strict band GATE (analytic d=1 Gaussian MI, n={gate_n}): MI terms vs Cover-Thomas closed form"
        )?;
    }
    let mut band = GateSummary::default();
    let mut seed = 0x6A55_1A20_u64;
    for &(a, b, c) in &STRICT_BAND_GAUSS_GRID {
        let atom = run_gaussian_atom_check(out, csv, gate_n, ksg_cfg, a, b, c, seed)?;
        band.observe_gaussian_atom_check(&atom);
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    }
    Ok(band)
}

/// Run the curated band and return the GATING summary, then also run the four synthetic
/// scenarios at `STRICT_BAND_DIAG_DIMS` as an INFORMATIONAL diagnostic: their gate counters
/// are printed, NOT folded into the returned (gating) summary, because they are a known non-GO
/// regime (documented findings, not regressions — see the `STRICT_BAND_*` rationale block).
fn run_strict_band(
    out: &mut dyn Write,
    csv: bool,
    ksg_cfg: &KsgConfig,
) -> Result<GateSummary, Exp0Error> {
    // --- Gating: analytic d=1 Gaussian grid (GO legitimately expected) ---
    let band = strict_band_gate(out, csv, ksg_cfg)?;

    // --- Informational (NON-gating) low-dimension scenario diagnostic ---
    let seeds = make_seeds(STRICT_BAND_SEEDS);
    if !csv {
        writeln!(out)?;
        writeln!(
            out,
            "Strict band DIAGNOSTIC (non-gating): four scenarios, dims={STRICT_BAND_DIAG_DIMS:?}, seeds={seeds:?}"
        )?;
    }
    let mut diag_summary = GateSummary::default();
    // No projection baselines: dims are already small and < the default hash_project_to.
    let common = CaseCommon {
        csv,
        n: STRICT_BAND_N,
        ksg_cfg,
        hash_project_to: None,
    };
    for d in STRICT_BAND_DIAG_DIMS {
        for &seed in &seeds {
            for name in [
                "independent_additive",
                "redundant_copy",
                "unique_s1",
                "xor_like",
            ] {
                let res = run_case(out, common, CaseSpec { name, d, seed })?;
                diag_summary.observe_case(name, d, res.metrics, res.diag);
            }
            if !csv {
                writeln!(out)?;
            }
        }
    }
    if !csv {
        writeln!(
            out,
            "  [diagnostic only, NOT gated] scenario verdict={} (known non-GO regime: see STRICT_BAND rationale)",
            diag_summary.status()
        )?;
        writeln!(
            out,
            "  [diagnostic only] monotonicity_violations={} invariant_violations={} red_zero={}/{} geometry_warnings={}",
            diag_summary.monotonicity_violations,
            diag_summary.invariant_violations,
            diag_summary.red_zero_passes,
            diag_summary.red_zero_checks,
            diag_summary.geometry_warnings,
        )?;
    }

    Ok(band)
}

/// Human-readable uncertainty report.
fn print_uncertainty(out: &mut dyn Write, u: &UncertaintySummary) -> io::Result<()> {
    writeln!(out)?;
    writeln!(
        out,
        "--- Uncertainty Quantification (d={UNCERTAINTY_DIM}, n_boot={}, n_perm={}, block={}, subsample={}, alpha={}) ---",
        u.n_boot, u.n_perm, u.block_size, u.subsample_len, u.alpha
    )?;
    writeln!(
        out,
        "Subsampling is Politis–Romano without replacement (KSG-safe); CIs are conservative (overstate n-sample uncertainty)."
    )?;
    for s in &u.scenarios {
        let (truth_s1, truth_s2) = marginal_truth(s.name);
        writeln!(
            out,
            "  {:>22}: truth(S1 info={truth_s1}, S2 info={truth_s2})",
            s.name
        )?;
        if let Some(b) = &s.boot {
            writeln!(
                out,
                "{:>26}  I1=[{:.3},{:.3}] I2=[{:.3},{:.3}] I12=[{:.3},{:.3}] Red=[{:.3},{:.3}] (valid I12: {}/{})",
                "",
                b.i1.ci_low, b.i1.ci_high,
                b.i2.ci_low, b.i2.ci_high,
                b.i12.ci_low, b.i12.ci_high,
                b.red.ci_low, b.red.ci_high,
                b.i12.n_valid, u.n_boot,
            )?;
        }
        if let (Some(p1), Some(p2)) = (s.perm_s1_p, s.perm_s2_p) {
            let mark = |p: f64, truth: bool| -> &'static str {
                if !p.is_finite() {
                    "??"
                } else if (p < u.alpha) == truth {
                    "ok"
                } else {
                    "XX"
                }
            };
            writeln!(
                out,
                "{:>26}  perm p(S1;T)={:.4} [{}]  p(S2;T)={:.4} [{}]",
                "",
                p1,
                mark(p1, truth_s1),
                p2,
                mark(p2, truth_s2),
            )?;
        }
    }
    writeln!(
        out,
        "  permutation agreements: {}/{}; bootstrap instabilities: {}",
        u.permutation_agreements, u.permutation_checks, u.bootstrap_instabilities
    )?;
    Ok(())
}

fn run_case(
    out: &mut dyn Write,
    common: CaseCommon<'_>,
    spec: CaseSpec<'_>,
) -> Result<CaseResult, Exp0Error> {
    let noise_std = 0.05;
    let n = common.n;
    let d = spec.d;
    let seed = spec.seed;
    let (s1, s2, t) = match spec.name {
        "independent_additive" => gen_independent_additive(n, d, noise_std, seed),
        "redundant_copy" => gen_redundant_copy(n, d, noise_std, seed),
        "unique_s1" => gen_unique_s1(n, d, noise_std, seed),
        "xor_like" => gen_xor_like(n, d, noise_std, seed),
        _ => unreachable!("unknown case: {}", spec.name),
    };

    let s1 = MatRef::new(&s1, n, d)?;
    let s2 = MatRef::new(&s2, n, d)?;
    let t = MatRef::new(&t, n, 1)?;

    let (s1z, _) = Standardizer::fit_transform(s1)?;
    let (s2z, _) = Standardizer::fit_transform(s2)?;
    let (tz, _) = Standardizer::fit_transform(t)?;

    let baseline = compute_metrics(s1z.as_ref(), s2z.as_ref(), tz.as_ref(), common.ksg_cfg)?;
    let diag = compute_diagnostics(
        s1z.as_ref(),
        s2z.as_ref(),
        tz.as_ref(),
        common.ksg_cfg.metric,
    );

    if common.csv {
        write_case_csv_row(
            out,
            common.ksg_cfg,
            CaseCsvRow {
                name: spec.name,
                seed: spec.seed,
                projection: ProjectionMethod::None,
                d,
                n,
                project_to: None,
                metrics: baseline,
                diag,
            },
        )?;
    } else {
        print_metrics(out, spec.name, d, spec.seed, baseline)?;
        print_intrinsic_dims(out, diag)?;
    }

    if let Some(dout) = common.hash_project_to {
        if d > dout {
            let p1 = HashProjector::new(d, dout, 0xA11CE_u64 ^ seed)?;
            let p2 = HashProjector::new(d, dout, 0xB22CE_u64 ^ seed)?;

            let s1p = p1.transform(s1z.as_ref())?;
            let s2p = p2.transform(s2z.as_ref())?;

            // Re-standardize after projection so Chebyshev distance has comparable scale.
            let (s1p, _) = Standardizer::fit_transform(s1p.as_ref())?;
            let (s2p, _) = Standardizer::fit_transform(s2p.as_ref())?;

            let projected =
                compute_metrics(s1p.as_ref(), s2p.as_ref(), tz.as_ref(), common.ksg_cfg)?;
            let diag_p = compute_diagnostics(
                s1p.as_ref(),
                s2p.as_ref(),
                tz.as_ref(),
                common.ksg_cfg.metric,
            );
            let case_name = format!("{}_hashproj", spec.name);
            if common.csv {
                write_case_csv_row(
                    out,
                    common.ksg_cfg,
                    CaseCsvRow {
                        name: &case_name,
                        seed: spec.seed,
                        projection: ProjectionMethod::Hash,
                        d: dout,
                        n,
                        project_to: Some(dout),
                        metrics: projected,
                        diag: diag_p,
                    },
                )?;
            } else {
                print_metrics(out, &case_name, dout, spec.seed, projected)?;
                print_intrinsic_dims(out, diag_p)?;
            }

            // PCA projection baseline (deterministic; no external deps).
            let (s1p, _) = PcaProjector::fit_transform(s1z.as_ref(), dout)?;
            let (s2p, _) = PcaProjector::fit_transform(s2z.as_ref(), dout)?;

            // Re-standardize after projection so Chebyshev distance has comparable scale.
            let (s1p, _) = Standardizer::fit_transform(s1p.as_ref())?;
            let (s2p, _) = Standardizer::fit_transform(s2p.as_ref())?;

            let projected =
                compute_metrics(s1p.as_ref(), s2p.as_ref(), tz.as_ref(), common.ksg_cfg)?;
            let diag_p = compute_diagnostics(
                s1p.as_ref(),
                s2p.as_ref(),
                tz.as_ref(),
                common.ksg_cfg.metric,
            );
            let case_name = format!("{}_pca", spec.name);
            if common.csv {
                write_case_csv_row(
                    out,
                    common.ksg_cfg,
                    CaseCsvRow {
                        name: &case_name,
                        seed: spec.seed,
                        projection: ProjectionMethod::Pca,
                        d: dout,
                        n,
                        project_to: Some(dout),
                        metrics: projected,
                        diag: diag_p,
                    },
                )?;
            } else {
                print_metrics(out, &case_name, dout, spec.seed, projected)?;
                print_intrinsic_dims(out, diag_p)?;
            }
        }
    }
    Ok(CaseResult {
        metrics: baseline,
        diag,
    })
}

struct CaseResult {
    metrics: Metrics,
    diag: Diagnostics,
}

#[derive(Debug, Clone, Copy)]
struct Diagnostics {
    id_s1: f64,
    id_s2: f64,
    id_t: f64,
    id_s12: f64,

    dc_cv_s1: f64,
    dc_nnr_s1: f64,
    dc_cv_s2: f64,
    dc_nnr_s2: f64,
    dc_cv_s12: f64,
    dc_nnr_s12: f64,

    gromov_s1: f64,
    gromov_s2: f64,
    gromov_s12: f64,
    gromov_t: f64,

    diam_s1: f64,
    diam_s2: f64,
    diam_s12: f64,
    diam_t: f64,
}

fn compute_diagnostics(
    s1: MatRef<'_>,
    s2: MatRef<'_>,
    t: MatRef<'_>,
    metric: Metric,
) -> Diagnostics {
    let cfg = IntrinsicDimConfig { k: 10, metric };

    let id_s1 = intrinsic_dimension_levina_bickel(s1, &cfg).unwrap_or(f64::NAN);
    let id_s2 = intrinsic_dimension_levina_bickel(s2, &cfg).unwrap_or(f64::NAN);
    let id_t = intrinsic_dimension_levina_bickel(t, &cfg).unwrap_or(f64::NAN);
    let id_s12 = concat_horiz(s1, s2)
        .ok()
        .and_then(|s12| intrinsic_dimension_levina_bickel(s12.as_ref(), &cfg).ok())
        .unwrap_or(f64::NAN);

    let dcfg = DistanceConcentrationConfig { metric };
    let ds1 = distance_concentration_stats(s1, &dcfg).ok();
    let ds2 = distance_concentration_stats(s2, &dcfg).ok();
    let ds12 = concat_horiz(s1, s2)
        .ok()
        .and_then(|s12| distance_concentration_stats(s12.as_ref(), &dcfg).ok());
    let dt = distance_concentration_stats(t, &dcfg).ok();

    let hcfg = pid_core::HyperbolicityConfig {
        n_samples: 500,
        metric,
        seed: 42,
    };
    let gromov_s1 = pid_core::gromov_hyperbolicity(s1, &hcfg).unwrap_or(f64::NAN);
    let gromov_s2 = pid_core::gromov_hyperbolicity(s2, &hcfg).unwrap_or(f64::NAN);
    let gromov_t = pid_core::gromov_hyperbolicity(t, &hcfg).unwrap_or(f64::NAN);
    let gromov_s12 = concat_horiz(s1, s2)
        .ok()
        .and_then(|s12| pid_core::gromov_hyperbolicity(s12.as_ref(), &hcfg).ok())
        .unwrap_or(f64::NAN);

    Diagnostics {
        id_s1,
        id_s2,
        id_t,
        id_s12,
        dc_cv_s1: ds1.map(|s| s.pairwise_cv).unwrap_or(f64::NAN),
        dc_nnr_s1: ds1.map(|s| s.nn_over_pairwise_mean).unwrap_or(f64::NAN),
        dc_cv_s2: ds2.map(|s| s.pairwise_cv).unwrap_or(f64::NAN),
        dc_nnr_s2: ds2.map(|s| s.nn_over_pairwise_mean).unwrap_or(f64::NAN),
        dc_cv_s12: ds12.map(|s| s.pairwise_cv).unwrap_or(f64::NAN),
        dc_nnr_s12: ds12.map(|s| s.nn_over_pairwise_mean).unwrap_or(f64::NAN),
        gromov_s1,
        gromov_s2,
        gromov_s12,
        gromov_t,
        diam_s1: ds1.map(|s| s.pairwise_max).unwrap_or(f64::NAN),
        diam_s2: ds2.map(|s| s.pairwise_max).unwrap_or(f64::NAN),
        diam_s12: ds12.map(|s| s.pairwise_max).unwrap_or(f64::NAN),
        diam_t: dt.map(|s| s.pairwise_max).unwrap_or(f64::NAN),
    }
}

fn print_intrinsic_dims(out: &mut dyn Write, d: Diagnostics) -> io::Result<()> {
    writeln!(
        out,
        "{:>20} {:>7} | ID(s1)={:>6.2} ID(s2)={:>6.2} ID(t)={:>6.2} ID(s1,s2)={:>6.2}",
        "", "", d.id_s1, d.id_s2, d.id_t, d.id_s12
    )?;
    writeln!(
        out,
        "{:>20} {:>7} | DCcv(s1)={:>6.3} nn/mean={:>6.3} | DCcv(s2)={:>6.3} nn/mean={:>6.3} | DCcv(s1,s2)={:>6.3} nn/mean={:>6.3}",
        "",
        "",
        d.dc_cv_s1,
        d.dc_nnr_s1,
        d.dc_cv_s2,
        d.dc_nnr_s2,
        d.dc_cv_s12,
        d.dc_nnr_s12
    )?;

    let dr_s1 = relative_delta(d.gromov_s1, d.diam_s1);
    let dr_s2 = relative_delta(d.gromov_s2, d.diam_s2);
    let dr_s12 = relative_delta(d.gromov_s12, d.diam_s12);
    let dr_t = relative_delta(d.gromov_t, d.diam_t);

    writeln!(
        out,
        "{:>20} {:>7} | d_rel(s1)={:>6.3} | d_rel(s2)={:>6.3} | d_rel(s1,s2)={:>6.3} | d_rel(t)={:>6.3}",
        "", "", dr_s1, dr_s2, dr_s12, dr_t
    )?;
    Ok(())
}

fn run_gaussian_channel_strong_dependence_sweep(
    out: &mut dyn Write,
    csv: bool,
    n: usize,
    ksg_cfg: &KsgConfig,
    seed: u64,
) -> Result<(), Exp0Error> {
    // Strong-dependence sweep (separate axis from "high d"):
    // X ~ N(0,1), Y = X + σN, N~N(0,1), so analytic MI is:
    // I(X;Y) = 0.5 ln(1 + 1/σ²).
    let sigmas = [1.0, 0.3, 0.1, 0.03, 0.01];

    let mut rng = Rng64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut noise = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(rng.normal());
        noise.push(rng.normal());
    }

    let xref = MatRef::new(&x, n, 1)?;
    let (xstd, _) = Standardizer::fit_transform(xref)?;

    if !csv {
        writeln!(out, "Strong-dependence sweep (Gaussian channel, 1D)")?;
        writeln!(out, "n={n}, k={}, metric={:?}", ksg_cfg.k, ksg_cfg.metric)?;
    }
    for &sigma in &sigmas {
        let mut y = Vec::with_capacity(n);
        for (&xi, &ni) in x.iter().zip(noise.iter()) {
            y.push(xi + sigma * ni);
        }

        let yref = MatRef::new(&y, n, 1)?;
        let (ystd, _) = Standardizer::fit_transform(yref)?;

        let mi_hat = ksg_mi(xstd.as_ref(), ystd.as_ref(), ksg_cfg)?;
        let mi_true = gaussian_channel_mi(sigma);
        if csv {
            write_gaussian_csv_row(out, sigma, n, ksg_cfg, mi_hat, mi_true)?;
        } else {
            writeln!(
                out,
                "  sigma={:<7.3}  MI_hat={:>8.3}  MI_true={:>8.3}  err={:>8.3}",
                sigma,
                mi_hat,
                mi_true,
                mi_hat - mi_true
            )?;
        }
    }
    if !csv {
        writeln!(out)?;
    }
    Ok(())
}

fn gaussian_channel_mi(sigma: f64) -> f64 {
    debug_assert!(sigma.is_finite());
    debug_assert!(sigma > 0.0);
    0.5 * (1.0 + 1.0 / (sigma * sigma)).ln()
}

// ---------------------------------------------------------------------------
// Analytic Gaussian PID ground truth (Barrett 2015)
// ---------------------------------------------------------------------------
//
// System (jointly Gaussian, the only case with a closed-form PID):
//   S1, S2 ~ N(0,1) independent (unit variance, uncorrelated),
//   T = a*S1[0] + b*S2[0] + c*Z,  Z ~ N(0,1) independent.
// Only the first coordinate of each source carries signal; the remaining d-1
// coordinates are independent N(0,1) noise (so the band exercises multivariate
// sources without changing the analytic MI, which depends only on the signal
// coordinate). Because (S1,S2,T) is jointly Gaussian and S1 ⟂ S2:
//
//   Var(T)            = a^2 + b^2 + c^2
//   Var(T | S1,S2)    = c^2
//   I(S1,S2; T) = 0.5 * ln(Var(T) / Var(T|S1,S2)) = 0.5 * ln((a^2+b^2+c^2)/c^2)
//   I(S1; T)    = 0.5 * ln(Var(T) / Var(T|S1))    = 0.5 * ln((a^2+b^2+c^2)/(b^2+c^2))
//   I(S2; T)    = 0.5 * ln((a^2+b^2+c^2)/(a^2+c^2))
// (Cover & Thomas, "Elements of Information Theory", §8.5: differential entropy
// of a Gaussian; conditional variances from the standard Gaussian regression.)
//
// PID atoms (Barrett 2015, Phys. Rev. E 91, 052802): for Gaussian systems the
// Williams–Beer redundancy reduces to the MINIMUM MUTUAL INFORMATION (MMI)
// redundancy, which is the unique PID consistent with the standard axioms:
//   Red  = min(I(S1;T), I(S2;T))
//   Unq1 = I(S1;T) - Red
//   Unq2 = I(S2;T) - Red
//   Syn  = I(S1,S2;T) - I(S1;T) - I(S2;T) + Red
//
// IMPORTANT distinction: exp0's estimator computes the continuous I^sx_∩
// redundancy (Ehrlich et al. 2024), which is NOT the MMI redundancy and need not
// equal it even in the population limit. The measure-INDEPENDENT ground truth here
// is therefore the three MI terms (I1, I2, I12); those are what the band gates on.
// The Barrett MMI atoms are computed for REPORTING and as a sanity reference, not
// tuned against the estimator (per AGENTS.md: a disagreement is a finding).
#[derive(Debug, Clone, Copy)]
struct GaussianAtomTruth {
    i1: f64,
    i2: f64,
    i12: f64,
    red_mmi: f64,
    unq1_mmi: f64,
    unq2_mmi: f64,
    syn_mmi: f64,
}

/// Closed-form MI terms and Barrett-2015 MMI atoms for the jointly-Gaussian system
/// `T = a*S1[0] + b*S2[0] + c*Z`. All in nats.
fn gaussian_atom_truth(a: f64, b: f64, c: f64) -> GaussianAtomTruth {
    let var_t = a * a + b * b + c * c;
    let i12 = 0.5 * (var_t / (c * c)).ln();
    let i1 = 0.5 * (var_t / (b * b + c * c)).ln();
    let i2 = 0.5 * (var_t / (a * a + c * c)).ln();
    let red_mmi = i1.min(i2);
    GaussianAtomTruth {
        i1,
        i2,
        i12,
        red_mmi,
        unq1_mmi: i1 - red_mmi,
        unq2_mmi: i2 - red_mmi,
        syn_mmi: i12 - i1 - i2 + red_mmi,
    }
}

/// Generate the jointly-Gaussian atom-check system into `(s1, s2, t)` row-major buffers.
/// Signal lives only in coordinate 0; coordinates 1..d are independent N(0,1) noise.
fn gen_gaussian_atom_system(
    n: usize,
    d: usize,
    a: f64,
    b: f64,
    c: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng64::new(seed);
    let mut s1 = vec![0.0; n * d];
    let mut s2 = vec![0.0; n * d];
    let mut t = vec![0.0; n];
    for i in 0..n {
        for j in 0..d {
            s1[i * d + j] = rng.normal();
            s2[i * d + j] = rng.normal();
        }
        t[i] = a * s1[i * d] + b * s2[i * d] + c * rng.normal();
    }
    (s1, s2, t)
}

/// Result of the analytic Gaussian-atom accuracy check.
#[derive(Debug, Clone, Copy)]
struct GaussianAtomCheck {
    /// Number of MI-term comparisons performed (measure-independent ground truth).
    mi_checks: usize,
    /// Number of those within the scale-aware tolerance.
    mi_passes: usize,
}

/// Run the analytic Gaussian-atom accuracy check: estimate the MI terms / atoms on a
/// jointly-Gaussian system and compare the MI terms against the Cover–Thomas closed form.
///
/// The hard, quantitative gate is on the three measure-independent MI terms
/// (`I1, I2, I12`). The Barrett-2015 MMI atoms are derived and printed alongside the
/// estimator's I^sx atoms for reference; their difference is reported, never tuned away
/// (I^sx ≠ MMI in general — see `gaussian_atom_truth`).
#[allow(clippy::too_many_arguments)]
fn run_gaussian_atom_check(
    out: &mut dyn Write,
    csv: bool,
    n: usize,
    ksg_cfg: &KsgConfig,
    a: f64,
    b: f64,
    c: f64,
    seed: u64,
) -> Result<GaussianAtomCheck, Exp0Error> {
    // d=1: pure signal, no noise dimensions to dilute the Chebyshev neighbour structure.
    // This is the KSG estimator's validated regime, so the closed-form MI terms are
    // recovered within tolerance and GO is legitimately attainable.
    let d = 1usize;
    let truth = gaussian_atom_truth(a, b, c);

    let (s1, s2, t) = gen_gaussian_atom_system(n, d, a, b, c, seed);
    let s1 = MatRef::new(&s1, n, d)?;
    let s2 = MatRef::new(&s2, n, d)?;
    let t = MatRef::new(&t, n, 1)?;
    let (s1z, _) = Standardizer::fit_transform(s1)?;
    let (s2z, _) = Standardizer::fit_transform(s2)?;
    let (tz, _) = Standardizer::fit_transform(t)?;

    // Estimate ONLY what the gate / report needs: the three MI terms (gated) and the single
    // EhrlichKsg I^sx redundancy (reported, not gated). Computing these directly — rather than
    // via `compute_metrics`, which also runs two extra redundancy methods and the co-information
    // — keeps the n=4000 gate (and its unit test) cheap. All MI terms use the same KSG config
    // and `NegativeHandling::Allow`-respecting downstream identities as the rest of exp0.
    let i1 = ksg_mi(s1z.as_ref(), tz.as_ref(), ksg_cfg)?;
    let i2 = ksg_mi(s2z.as_ref(), tz.as_ref(), ksg_cfg)?;
    let i12 = ksg_mi_concat_xy(s1z.as_ref(), s2z.as_ref(), tz.as_ref(), ksg_cfg)?;

    // Compare the measure-independent MI terms with a scale-aware tolerance (the same
    // noise model used elsewhere in the gate); these are the quantitative pass/fail items.
    let mut mi_checks = 0usize;
    let mut mi_passes = 0usize;
    for (hat, truth_val) in [(i1, truth.i1), (i2, truth.i2), (i12, truth.i12)] {
        mi_checks += 1;
        if (hat - truth_val).abs() <= estimate_tol(truth_val) {
            mi_passes += 1;
        }
    }

    if !csv {
        let red_ehrlich = isx_redundancy(
            s1z.as_ref(),
            s2z.as_ref(),
            tz.as_ref(),
            &IsxConfig {
                k: ksg_cfg.k,
                metric: ksg_cfg.metric,
                tie_epsilon: ksg_cfg.tie_epsilon,
                method: IsxMethod::EhrlichKsg,
            },
        )?;
        let syn_ehrlich = i12 - i1 - i2 + red_ehrlich;
        writeln!(
            out,
            "Gaussian atom check (Barrett 2015 MMI; system T = {a}*S1 + {b}*S2 + {c}*Z, d={d}, n={n})"
        )?;
        writeln!(
            out,
            "  MI terms (nats): I1 hat/true = {:.3}/{:.3}  I2 = {:.3}/{:.3}  I12 = {:.3}/{:.3}  [{}/{} within tol]",
            i1, truth.i1, i2, truth.i2, i12, truth.i12, mi_passes, mi_checks
        )?;
        writeln!(
            out,
            "  Barrett MMI atoms (analytic): Red={:.3} Unq1={:.3} Unq2={:.3} Syn={:.3}",
            truth.red_mmi, truth.unq1_mmi, truth.unq2_mmi, truth.syn_mmi
        )?;
        writeln!(
            out,
            "  Estimator I^sx atoms:         Red={:.3}            Syn={:.3}  (I^sx != MMI; difference is informational, not a gate)",
            red_ehrlich, syn_ehrlich
        )?;
    }

    Ok(GaussianAtomCheck {
        mi_checks,
        mi_passes,
    })
}

#[derive(Clone, Copy)]
struct Metrics {
    mi_s1_t: f64,
    mi_s2_t: f64,
    mi_s1s2_t: f64,
    ci: f64,
    r_bar: f64,
    v_bar: f64,
    red_ehrlich: f64,
    red_local_min: f64,
    red_disjunction: f64,
    syn_ehrlich: f64,
}

#[derive(Debug, Default, Clone)]
struct GateSummary {
    case_results: usize,
    red_zero_checks: usize,
    red_zero_passes: usize,
    monotonicity_violations: usize,
    invariant_violations: usize,
    geometry_warnings: usize,
    // Uncertainty-quantification contribution (all zero / false when UQ disabled,
    // which keeps the default verdict and metric counts unchanged).
    uncertainty_enabled: bool,
    permutation_checks: usize,
    permutation_agreements: usize,
    bootstrap_instabilities: usize,
}

impl GateSummary {
    fn observe_case(&mut self, name: &str, d: usize, metrics: Metrics, diag: Diagnostics) {
        self.case_results += 1;

        // Monotonicity of MI under adding a source: I(S1,S2;T) >= I(Si;T). For the
        // joint-vs-marginal case this IS the conditional-MI nonnegativity condition
        // (I(S1;T|S2) = I(S1,S2;T) - I(S2;T) >= 0 ⇔ I(S1,S2;T) >= I(S2;T)), so a
        // separate "CMI nonnegativity" counter would be identical by construction —
        // it is reported once, here.
        //
        // These are NOISY kNN estimates (SE ~0.01–0.05 nats), not exact identities, so we
        // compare with a scale-aware tolerance: an exact-equality tolerance (1e-9) counts
        // pure estimator noise as a violation on essentially every case.
        if metrics.mi_s1s2_t + estimate_tol(metrics.mi_s1_t) < metrics.mi_s1_t {
            self.monotonicity_violations += 1;
        }
        if metrics.mi_s1s2_t + estimate_tol(metrics.mi_s2_t) < metrics.mi_s2_t {
            self.monotonicity_violations += 1;
        }

        // r̄/v̄ are ratios with the joint MI as denominator. At the estimator noise floor
        // (e.g. a pure-synergy system where I(S1;T)=I(S2;T)=I(S1,S2;T)=0) they are 0/0 = NaN,
        // a CORRECT "no redundancy/vulnerability structure" result — not a bound violation.
        // Only test the [0,2] bound when the joint MI is resolvable, and with the same
        // scale-aware tolerance. For n=2, v̄ = 2 − r̄, so the two checks are equivalent.
        const INVARIANT_MI_FLOOR: f64 = 0.05;
        if metrics.mi_s1s2_t >= INVARIANT_MI_FLOOR
            && (!bounded_degree(metrics.r_bar, 0.0, 2.0, estimate_tol(metrics.r_bar))
                || !bounded_degree(metrics.v_bar, 0.0, 2.0, estimate_tol(metrics.v_bar)))
        {
            self.invariant_violations += 1;
        }

        if name == "independent_additive" {
            self.red_zero_checks += 1;
            if metrics.red_ehrlich.abs() < red_zero_threshold(d) {
                self.red_zero_passes += 1;
            }
            let dr_s1 = relative_delta(diag.gromov_s1, diag.diam_s1);
            if diag.id_s1 > 20.0 || diag.dc_cv_s1 < 0.1 || dr_s1 < 0.1 {
                self.geometry_warnings += 1;
            }
        }
    }

    /// Fold the analytic Gaussian-atom MI-term check into the gate. Each system counts as a
    /// case result; each MI term that disagrees with its Cover–Thomas closed form beyond the
    /// scale-aware tolerance counts as an invariant violation, so a quantitative analytic
    /// disagreement blocks GO on the curated band. (Only the measure-independent MI terms are
    /// gated; the Barrett MMI vs I^sx atom difference is reported, not gated — see
    /// `run_gaussian_atom_check`.)
    fn observe_gaussian_atom_check(&mut self, c: &GaussianAtomCheck) {
        self.case_results += 1;
        self.invariant_violations += c.mi_checks - c.mi_passes;
    }

    /// Absorb the derived gate checks from an opt-in uncertainty run.
    fn observe_uncertainty(&mut self, u: &UncertaintySummary) {
        self.uncertainty_enabled = u.enabled;
        self.permutation_checks = u.permutation_checks;
        self.permutation_agreements = u.permutation_agreements;
        self.bootstrap_instabilities = u.bootstrap_instabilities;
    }

    /// Total uncertainty-side violations: permutation disagreements with the
    /// preregistered ground-truth marginal-significance table, plus joint-MI
    /// bootstrap instabilities at the most favourable dimension. Zero when UQ
    /// is disabled.
    fn uncertainty_violations(&self) -> usize {
        let perm_disagreements = self.permutation_checks - self.permutation_agreements;
        perm_disagreements + self.bootstrap_instabilities
    }

    fn status(&self) -> &'static str {
        if self.case_results == 0 {
            return "NO-GO";
        }
        if self.monotonicity_violations == 0
            && self.invariant_violations == 0
            && self.geometry_warnings == 0
            && self.red_zero_checks == self.red_zero_passes
            && self.uncertainty_violations() == 0
        {
            "GO"
        } else if self.red_zero_checks > 0
            && self.red_zero_passes * 2 >= self.red_zero_checks
            && self.invariant_violations == 0
        {
            "PIVOT"
        } else {
            "NO-GO"
        }
    }

    fn status_code(&self) -> usize {
        match self.status() {
            "GO" => 0,
            "PIVOT" => 1,
            _ => 2,
        }
    }

    fn print(&self, out: &mut dyn Write) -> io::Result<()> {
        writeln!(
            out,
            "Passes (Independent Additive Zero-Redundancy check): {}/{}",
            self.red_zero_passes, self.red_zero_checks
        )?;
        writeln!(out, "Case Results: {}", self.case_results)?;
        writeln!(out, "Geometry Warnings: {}", self.geometry_warnings)?;
        writeln!(
            out,
            "Monotonicity Violations (= CMI nonnegativity): {}",
            self.monotonicity_violations
        )?;
        writeln!(
            out,
            "Invariant Bound Violations: {}",
            self.invariant_violations
        )?;
        if self.uncertainty_enabled {
            writeln!(
                out,
                "Permutation Marginal-Significance Agreements: {}/{}",
                self.permutation_agreements, self.permutation_checks
            )?;
            writeln!(
                out,
                "Bootstrap Joint-MI Instabilities: {}",
                self.bootstrap_instabilities
            )?;
        }
        writeln!(out, "Status: {}", self.status())?;
        Ok(())
    }
}

fn bounded_degree(value: f64, lo: f64, hi: f64, tol: f64) -> bool {
    value.is_finite() && value >= lo - tol && value <= hi + tol
}

/// Tolerance for declaring a *genuine* violation when comparing noisy kNN MI estimates (or
/// degrees derived from them). KSG estimates carry finite-sample noise on the order of
/// 0.01–0.05 nats, so comparing them with an exact-identity tolerance (1e-9) reports estimator
/// noise as a violation. A violation counts only when it exceeds the larger of an absolute
/// noise floor and a relative fraction of the quantity's own magnitude. Reserve 1e-9 for exact
/// algebraic identities (e.g. the PID atom-sum reconstruction), not for cross-estimate checks.
fn estimate_tol(scale: f64) -> f64 {
    const ABS_TOL: f64 = 0.05;
    const REL_TOL: f64 = 0.1;
    ABS_TOL.max(REL_TOL * scale.abs())
}

fn red_zero_threshold(d: usize) -> f64 {
    if d <= 10 {
        0.1
    } else if d <= 100 {
        0.2
    } else {
        0.3
    }
}

fn relative_delta(delta: f64, diameter: f64) -> f64 {
    if delta.is_finite() && diameter.is_finite() && diameter > 0.0 {
        2.0 * delta / diameter
    } else {
        f64::NAN
    }
}

fn compute_metrics(
    s1: MatRef<'_>,
    s2: MatRef<'_>,
    t: MatRef<'_>,
    ksg_cfg: &KsgConfig,
) -> pid_core::PidResult<Metrics> {
    let mi_s1_t = ksg_mi(s1, t, ksg_cfg)?;
    let mi_s2_t = ksg_mi(s2, t, ksg_cfg)?;
    let mi_s1s2_t = ksg_mi_concat_xy(s1, s2, t, ksg_cfg)?;
    let ci = co_information_pairwise(s1, s2, t, ksg_cfg)?;

    let red_ehrlich = isx_redundancy(
        s1,
        s2,
        t,
        &IsxConfig {
            k: ksg_cfg.k,
            metric: ksg_cfg.metric,
            tie_epsilon: ksg_cfg.tie_epsilon,
            method: IsxMethod::EhrlichKsg,
        },
    )?;

    let red_local_min = isx_redundancy(
        s1,
        s2,
        t,
        &IsxConfig {
            k: ksg_cfg.k,
            metric: ksg_cfg.metric,
            tie_epsilon: ksg_cfg.tie_epsilon,
            method: IsxMethod::LocalMinKsg,
        },
    )?;

    let red_disjunction = isx_redundancy(
        s1,
        s2,
        t,
        &IsxConfig {
            k: ksg_cfg.k,
            metric: ksg_cfg.metric,
            tie_epsilon: ksg_cfg.tie_epsilon,
            method: IsxMethod::DisjunctionFromLocalMi,
        },
    )
    .unwrap_or(f64::NAN);

    let r_bar = average_degree_of_redundancy(&[mi_s1_t, mi_s2_t], mi_s1s2_t);
    let v_bar = average_degree_of_vulnerability(mi_s1s2_t, &[mi_s2_t, mi_s1_t]);

    Ok(Metrics {
        mi_s1_t,
        mi_s2_t,
        mi_s1s2_t,
        ci,
        r_bar,
        v_bar,
        red_ehrlich,
        red_local_min,
        red_disjunction,
        syn_ehrlich: mi_s1s2_t - mi_s1_t - mi_s2_t + red_ehrlich,
    })
}

fn print_metrics(
    out: &mut dyn Write,
    name: &str,
    d: usize,
    seed: u64,
    m: Metrics,
) -> io::Result<()> {
    writeln!(
        out,
        "{name:>20} d={d:<4} seed={seed:<10} | I1={:>7.3} I2={:>7.3} I12={:>7.3} CI={:>7.3} | r_bar={:>5.2} v_bar={:>5.2} | Red(ehr)={:>7.3} Syn(ehr)={:>7.3} | Red(disj)={:>7.3}",
        m.mi_s1_t,
        m.mi_s2_t,
        m.mi_s1s2_t,
        m.ci,
        m.r_bar,
        m.v_bar,
        m.red_ehrlich,
        m.syn_ehrlich,
        m.red_disjunction,
    )?;
    Ok(())
}

fn write_case_csv_header(out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "case_name,seed,projection,d,n,k,metric,project_to,mi_s1_t,mi_s2_t,mi_s1s2_t,ci,r_bar,v_bar,red_ehrlich,red_local_min,red_disjunction,syn_ehrlich,id_s1,id_s2,id_t,id_s12,dc_cv_s1,dc_nnratio_s1,dc_cv_s2,dc_nnratio_s2,dc_cv_s12,dc_nnratio_s12,gromov_s1,gromov_s2,gromov_s12,gromov_t,dr_s1,dr_s2,dr_s12,dr_t"
    )
}

#[derive(Clone, Copy)]
enum ProjectionMethod {
    None,
    Hash,
    Pca,
}

impl ProjectionMethod {
    fn as_str(self) -> &'static str {
        match self {
            ProjectionMethod::None => "none",
            ProjectionMethod::Hash => "hash",
            ProjectionMethod::Pca => "pca",
        }
    }
}

struct CaseCsvRow<'a> {
    name: &'a str,
    seed: u64,
    projection: ProjectionMethod,
    d: usize,
    n: usize,
    project_to: Option<usize>,
    metrics: Metrics,
    diag: Diagnostics,
}

fn write_case_csv_row(
    out: &mut dyn Write,
    ksg_cfg: &KsgConfig,
    row: CaseCsvRow<'_>,
) -> io::Result<()> {
    let project_to = row.project_to.map_or_else(String::new, |v| v.to_string());
    let dr_s1 = relative_delta(row.diag.gromov_s1, row.diag.diam_s1);
    let dr_s2 = relative_delta(row.diag.gromov_s2, row.diag.diam_s2);
    let dr_s12 = relative_delta(row.diag.gromov_s12, row.diag.diam_s12);
    let dr_t = relative_delta(row.diag.gromov_t, row.diag.diam_t);

    writeln!(
        out,
        "{},{},{},{},{},{},{:?},{project_to},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e}",
        row.name,
        row.seed,
        row.projection.as_str(),
        row.d,
        row.n,
        ksg_cfg.k,
        ksg_cfg.metric,
        row.metrics.mi_s1_t,
        row.metrics.mi_s2_t,
        row.metrics.mi_s1s2_t,
        row.metrics.ci,
        row.metrics.r_bar,
        row.metrics.v_bar,
        row.metrics.red_ehrlich,
        row.metrics.red_local_min,
        row.metrics.red_disjunction,
        row.metrics.syn_ehrlich,
        row.diag.id_s1,
        row.diag.id_s2,
        row.diag.id_t,
        row.diag.id_s12,
        row.diag.dc_cv_s1,
        row.diag.dc_nnr_s1,
        row.diag.dc_cv_s2,
        row.diag.dc_nnr_s2,
        row.diag.dc_cv_s12,
        row.diag.dc_nnr_s12,
        row.diag.gromov_s1,
        row.diag.gromov_s2,
        row.diag.gromov_s12,
        row.diag.gromov_t,
        dr_s1,
        dr_s2,
        dr_s12,
        dr_t,
    )
}

fn write_gaussian_csv_header(out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "sigma,n,k,metric,mi_hat,mi_true,err")
}

fn write_gaussian_csv_row(
    out: &mut dyn Write,
    sigma: f64,
    n: usize,
    ksg_cfg: &KsgConfig,
    mi_hat: f64,
    mi_true: f64,
) -> io::Result<()> {
    writeln!(
        out,
        "{sigma:.15e},{n},{},{:?},{mi_hat:.15e},{mi_true:.15e},{:.15e}",
        ksg_cfg.k,
        ksg_cfg.metric,
        mi_hat - mi_true
    )
}

fn gen_independent_additive(
    n: usize,
    d: usize,
    noise_std: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng64::new(seed);
    let mut s1 = vec![0.0; n * d];
    let mut s2 = vec![0.0; n * d];
    let mut t = vec![0.0; n];

    for i in 0..n {
        for j in 0..d {
            s1[i * d + j] = rng.normal();
            s2[i * d + j] = rng.normal();
        }
        t[i] = s1[i * d] + s2[i * d] + noise_std * rng.normal();
    }
    (s1, s2, t)
}

fn gen_redundant_copy(
    n: usize,
    d: usize,
    noise_std: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng64::new(seed);
    let mut s1 = vec![0.0; n * d];
    let mut s2 = vec![0.0; n * d];
    let mut t = vec![0.0; n];

    for i in 0..n {
        let base = rng.normal();
        t[i] = base;
        s1[i * d] = base + noise_std * rng.normal();
        s2[i * d] = base + noise_std * rng.normal();
        for j in 1..d {
            s1[i * d + j] = rng.normal();
            s2[i * d + j] = rng.normal();
        }
    }
    (s1, s2, t)
}

fn gen_unique_s1(n: usize, d: usize, noise_std: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng64::new(seed);
    let mut s1 = vec![0.0; n * d];
    let mut s2 = vec![0.0; n * d];
    let mut t = vec![0.0; n];

    for i in 0..n {
        for j in 0..d {
            s1[i * d + j] = rng.normal();
            s2[i * d + j] = rng.normal();
        }
        t[i] = s1[i * d] + noise_std * rng.normal();
    }
    (s1, s2, t)
}

fn gen_xor_like(n: usize, d: usize, noise_std: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng64::new(seed);
    let mut s1 = vec![0.0; n * d];
    let mut s2 = vec![0.0; n * d];
    let mut t = vec![0.0; n];

    for i in 0..n {
        let a = rng.normal();
        let b = rng.normal();
        s1[i * d] = a;
        s2[i * d] = b;

        // XOR-like: target depends on the interaction sign(a*b) rather than either alone.
        let sign = if a * b > 0.0 { 1.0 } else { -1.0 };
        t[i] = sign + noise_std * rng.normal();

        for j in 1..d {
            s1[i * d + j] = rng.normal();
            s2[i * d + j] = rng.normal();
        }
    }
    (s1, s2, t)
}

#[derive(Clone)]
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64() >> 11; // 53 bits
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        // Box–Muller.
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> String {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("pid-exp0-{name}-{stamp}"))
            .display()
            .to_string()
    }

    #[test]
    fn exp0_runlog_export_is_valid_and_summarizable() {
        let summary_path = temp_path("summary.json");
        let runlog_path = temp_path("runlog.jsonl");
        let gates = GateSummary {
            case_results: 1,
            red_zero_checks: 1,
            red_zero_passes: 1,
            ..Default::default()
        };
        let dims = [10usize, 64, 256];
        let seeds = [42u64];
        write_summary_json(&summary_path, &gates, 500, 3, &dims, &seeds, Some(64), None).unwrap();
        write_exp0_runlog(
            &runlog_path,
            Some(&summary_path),
            &gates,
            Exp0RunConfig {
                n: 500,
                k: 3,
                dims: &dims,
                seeds: &seeds,
                hash_project_to: Some(64),
            },
            None,
        )
        .unwrap();

        let events = pid_runlog::read_events_from_path(&runlog_path).unwrap();
        let validation = pid_runlog::validate_events(&events);
        assert!(validation.is_valid(), "{:?}", validation.issues);
        let summary = pid_runlog::summarize_events(&events).unwrap();
        assert_eq!(summary.run_id.as_deref(), Some("exp0-rust-quick-run"));
        assert_eq!(summary.pid_metrics, 7);
        assert_eq!(summary.pid_metric_events, 7);
        assert_eq!(summary.artifacts, 1);
        assert_eq!(summary.errors, 0);

        let _ = std::fs::remove_file(summary_path);
        let _ = std::fs::remove_file(runlog_path);
    }

    #[test]
    fn exp0_runlog_records_non_go_status_as_recoverable_error() {
        let runlog_path = temp_path("nogate.jsonl");
        let gates = GateSummary {
            case_results: 0,
            ..Default::default()
        };
        write_exp0_runlog(
            &runlog_path,
            None,
            &gates,
            Exp0RunConfig {
                n: 500,
                k: 3,
                dims: &[10],
                seeds: &[42],
                hash_project_to: Some(64),
            },
            None,
        )
        .unwrap();

        let events = pid_runlog::read_events_from_path(&runlog_path).unwrap();
        let validation = pid_runlog::validate_events(&events);
        assert!(validation.is_valid(), "{:?}", validation.issues);
        let summary = pid_runlog::summarize_events(&events).unwrap();
        assert_eq!(summary.errors, 1);
        assert!(events.iter().any(|event| matches!(
            event,
            RunLogEvent::ErrorLogged {
                recoverable: true,
                ..
            }
        )));

        let _ = std::fs::remove_file(runlog_path);
    }

    fn ksg_cfg_for_test() -> KsgConfig {
        KsgConfig {
            k: 3,
            metric: Metric::Chebyshev,
            tie_epsilon: 0.0,
            negative_handling: NegativeHandling::ClampToZero,
        }
    }

    #[test]
    fn uncertainty_recovers_marginal_truth_table() {
        // The preregistered, ground-truth-derived contract: the permutation null
        // test must call a source significant iff it is marginally informative by
        // construction. On healthy small-d data this should be recovered exactly.
        // Small counts keep this fast under `cargo test` (debug).
        let cfg = UncertaintyConfig {
            n_boot: 24,
            n_perm: 60,
            block_size: 1,
            alpha: 0.05,
            seed: 0xC0FFEE,
        };
        let u = compute_uncertainty(240, &ksg_cfg_for_test(), cfg).unwrap();
        // All four scenarios, both sources → 8 checks; all should agree.
        assert_eq!(u.permutation_checks, 8);
        assert_eq!(
            u.permutation_agreements, 8,
            "permutation null failed to recover marginal-informativeness truth"
        );
        // Subsampling is KSG-safe, so joint-MI bootstrap should be stable.
        assert_eq!(u.bootstrap_instabilities, 0);
        // Per-scenario sanity: unique_s1 → S1 significant, S2 not.
        let unique = u
            .scenarios
            .iter()
            .find(|s| s.name == "unique_s1")
            .expect("unique_s1 present");
        assert!(unique.perm_s1_p.unwrap() < cfg.alpha);
        assert!(unique.perm_s2_p.unwrap() >= cfg.alpha);
    }

    #[test]
    fn uncertainty_violations_block_go_but_not_when_clean() {
        // A clean uncertainty run (agreements == checks, no instabilities) must
        // contribute zero violations and therefore not change the verdict.
        let mut gates = GateSummary {
            case_results: 1,
            red_zero_checks: 1,
            red_zero_passes: 1,
            ..Default::default()
        };
        assert_eq!(gates.status(), "GO");
        let clean = UncertaintySummary {
            enabled: true,
            permutation_checks: 8,
            permutation_agreements: 8,
            bootstrap_instabilities: 0,
            ..Default::default()
        };
        gates.observe_uncertainty(&clean);
        assert_eq!(gates.uncertainty_violations(), 0);
        assert_eq!(gates.status(), "GO");

        // A disagreement (e.g. a pure-noise source flagged significant) must block GO.
        let dirty = UncertaintySummary {
            enabled: true,
            permutation_checks: 8,
            permutation_agreements: 6,
            bootstrap_instabilities: 1,
            ..Default::default()
        };
        gates.observe_uncertainty(&dirty);
        assert_eq!(gates.uncertainty_violations(), 3);
        assert_ne!(gates.status(), "GO");
    }

    #[test]
    fn uncertainty_runlog_is_valid_and_keeps_pid_metrics_at_eight() {
        let runlog_path = temp_path("unc-runlog.jsonl");
        let mut gates = GateSummary {
            case_results: 12,
            red_zero_checks: 3,
            red_zero_passes: 2,
            invariant_violations: 7,
            ..Default::default()
        };
        let cfg = UncertaintyConfig {
            n_boot: 16,
            n_perm: 40,
            block_size: 1,
            alpha: 0.05,
            seed: 7,
        };
        let u = compute_uncertainty(200, &ksg_cfg_for_test(), cfg).unwrap();
        gates.observe_uncertainty(&u);
        write_exp0_runlog(
            &runlog_path,
            None,
            &gates,
            Exp0RunConfig {
                n: 200,
                k: 3,
                dims: &[10],
                seeds: &[42],
                hash_project_to: Some(64),
            },
            Some(&u),
        )
        .unwrap();

        let events = pid_runlog::read_events_from_path(&runlog_path).unwrap();
        let validation = pid_runlog::validate_events(&events);
        assert!(validation.is_valid(), "{:?}", validation.issues);
        let summary = pid_runlog::summarize_events(&events).unwrap();
        // Uncertainty events are EvaluationMetric, so the 7 PidMetric gate events
        // are unchanged; the CI smoke greps rely on this invariant.
        assert_eq!(summary.pid_metrics, 7);
        assert!(summary.evaluation_metrics >= 4);

        let _ = std::fs::remove_file(runlog_path);
    }

    #[test]
    fn gaussian_atom_truth_matches_closed_form() {
        // Independent jointly-Gaussian system T = a*S1 + b*S2 + c*Z; verify the closed-form
        // MI terms (Cover & Thomas) and the Barrett-2015 MMI atom identities, NOT against the
        // estimator but against hand-checked algebra.
        let (a, b, c) = (1.0, 1.0, 1.0);
        let truth = gaussian_atom_truth(a, b, c);
        // var_t = 3, Var(T|S1,S2) = 1 => I12 = 0.5 ln 3.
        assert!((truth.i12 - 0.5 * 3.0_f64.ln()).abs() < 1e-12);
        // I(S1;T) = 0.5 ln(3/2) (b^2+c^2 = 2); symmetric in a<->b so I1 == I2 here.
        assert!((truth.i1 - 0.5 * (3.0_f64 / 2.0).ln()).abs() < 1e-12);
        assert!((truth.i2 - truth.i1).abs() < 1e-12);
        // Barrett MMI: Red = min(I1,I2) = I1; Unq = 0; identity Red+Unq1+Unq2+Syn = I12.
        assert!((truth.red_mmi - truth.i1).abs() < 1e-12);
        assert!(truth.unq1_mmi.abs() < 1e-12 && truth.unq2_mmi.abs() < 1e-12);
        let sum = truth.red_mmi + truth.unq1_mmi + truth.unq2_mmi + truth.syn_mmi;
        assert!((sum - truth.i12).abs() < 1e-12, "MMI atoms must sum to I12");
        // Asymmetric system: a > b => I1 > I2, Red = I2, Unq1 = I1 - I2 > 0.
        let asym = gaussian_atom_truth(1.0, 0.3, 1.0);
        assert!(asym.i1 > asym.i2);
        assert!((asym.red_mmi - asym.i2).abs() < 1e-12);
        assert!(asym.unq1_mmi > 0.0 && asym.unq2_mmi.abs() < 1e-12);
    }

    #[test]
    fn strict_band_gate_is_go_in_validated_regime() {
        // The curated analytic band (d=1 Gaussian grid at STRICT_BAND_GATE_N) must return GO:
        // this is the regime where the KSG estimator recovers the closed-form MI terms within
        // the documented scale-aware noise floor, so a regression here is a genuine signal.
        // This is the only sweep `--strict-gate` enforces — the default high-d sweep's
        // PIVOT/NO-GO stays informative and ungated.
        let mut sink = Vec::new();
        let band = strict_band_gate(&mut sink, true, &ksg_cfg_for_test()).unwrap();
        assert_eq!(
            band.status(),
            "GO",
            "curated analytic band must be GO in the validated regime; invariant_violations={}",
            band.invariant_violations
        );
        // Three grid systems, each contributing one case result; all MI checks within tol.
        assert_eq!(band.case_results, STRICT_BAND_GAUSS_GRID.len());
        assert_eq!(band.invariant_violations, 0);
    }
}
