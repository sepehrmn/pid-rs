//! Pipeline functions that compose PLS projection, PID decomposition, and bootstrap
//! uncertainty quantification.
//!
//! These are convenience entry points for the common VLA analysis workflow:
//!
//! 1. `pls_project_then_pid3` — fit PLS on high-dimensional embeddings, project into a
//!    low-dimensional task-relevant subspace, then run full 3-source SxPID.
//! 2. `bootstrap_pid3` — block-bootstrap resample rows of (V,L,D,A) jointly, recompute PID
//!    on each resample, and return percentile CIs on every PID atom.

use crate::bootstrap::BootstrapConfig;
use crate::concat_horiz;
use crate::discrete_pid::discrete_pid3;
use crate::error::{PidError, PidResult};
use crate::matrix::{MatOwned, MatRef};
use crate::pid2::{pid2_isx, Pid2Config, Pid2Result};
use crate::pid3::{pid3_isx, Antichain3, Pid3Config, Pid3Result};
use crate::pls::PlsProjector;
use crate::preprocess::SplitMix64;
use crate::sxpid::discrete_sxpid2;

// ── PLS → PID3 ─────────────────────────────────────────────────────────────

/// Configuration for [`pls_project_then_pid3`].
#[derive(Debug, Clone)]
pub struct PlsPid3Config {
    /// Number of PLS latent components to extract (applied to each source and target).
    pub pls_components: usize,
    /// PID3 estimator configuration (k, metric, tie_epsilon).
    pub pid_cfg: Pid3Config,
}

/// Output of [`pls_project_then_pid3`].
#[derive(Debug, Clone)]
pub struct PlsPid3Result {
    /// PID decomposition on the PLS-projected embeddings.
    pub pid: Pid3Result,
    /// Number of PLS components used.
    pub pls_components: usize,
    /// Input column counts for V, L, D, A before projection.
    pub input_dims: [usize; 4],
    /// Output column count after projection (= pls_components).
    pub projected_dim: usize,
}

/// Fit per-source PLS projectors (each source → A) to reduce dimensionality, then
/// run 3-source SxPID on the projected embeddings.
///
/// Each of V, L, D is projected through its own PLS model fitted with A as target.
/// A is projected through a PLS fitted with the concatenated VLD as target.
/// All four projections yield `pls_components`-dimensional representations.
///
/// The three sources (V, L, D) must share the same row count `n`, and A must also have `n` rows.
///
/// # Leakage warning
///
/// This function fits PLS on **all** provided data. For proper train/test separation,
/// call [`PlsProjector::fit`] on training data only, then [`PlsProjector::transform`]
/// on each split, and finally [`pid3_isx`] on the projected matrices.
pub fn pls_project_then_pid3(
    v: MatRef<'_>,
    l: MatRef<'_>,
    d: MatRef<'_>,
    a: MatRef<'_>,
    cfg: &PlsPid3Config,
) -> PidResult<PlsPid3Result> {
    let n = v.nrows();
    if l.nrows() != n || d.nrows() != n || a.nrows() != n {
        return Err(PidError::RowCountMismatch {
            context: "pls_project_then_pid3",
            left_rows: n,
            right_rows: l.nrows().min(d.nrows()).min(a.nrows()),
        });
    }

    // Fit a per-source PLS projector: each source S_i → A.
    // This gives each source its own low-d task-relevant representation.
    let v_proj = PlsProjector::fit(v, a, cfg.pls_components)?.transform(v)?;
    let l_proj = PlsProjector::fit(l, a, cfg.pls_components)?.transform(l)?;
    let d_proj = PlsProjector::fit(d, a, cfg.pls_components)?.transform(d)?;
    // For A, fit a PLS using the concatenated VLD as target so that the
    // projected target captures task-relevant variance from the sources.
    let vld = concat_horiz(concat_horiz(v, l)?.as_ref(), d)?;
    let a_proj = PlsProjector::fit(a, vld.as_ref(), cfg.pls_components)?.transform(a)?;

    let pid = pid3_isx(
        v_proj.as_ref(),
        l_proj.as_ref(),
        d_proj.as_ref(),
        a_proj.as_ref(),
        &cfg.pid_cfg,
    )?;

    Ok(PlsPid3Result {
        pid,
        pls_components: cfg.pls_components,
        input_dims: [v.ncols(), l.ncols(), d.ncols(), a.ncols()],
        projected_dim: cfg.pls_components,
    })
}

// ── Bootstrap PID3 ─────────────────────────────────────────────────────────

/// Per-atom bootstrap confidence interval for a 3-source PID decomposition.
#[derive(Debug, Clone)]
pub struct Pid3BootstrapAtom {
    /// The antichain identifying this atom on the PID lattice.
    pub antichain: Antichain3,
    /// Point estimate on the original (un-resampled) data.
    pub point_estimate: f64,
    /// Mean of the bootstrap distribution.
    pub boot_mean: f64,
    /// Standard error (std of bootstrap distribution).
    pub boot_se: f64,
    /// Lower percentile CI bound.
    pub ci_low: f64,
    /// Upper percentile CI bound.
    pub ci_high: f64,
}

/// Result of [`bootstrap_pid3`].
#[derive(Debug, Clone)]
pub struct BootstrapPid3Result {
    /// Point estimate PID result on the original data.
    pub point_estimate: Pid3Result,
    /// Bootstrap CIs for each atom (same canonical order as `point_estimate.atoms`).
    pub atoms: Vec<Pid3BootstrapAtom>,
    /// Number of bootstrap resamples used.
    pub n_boot: usize,
    /// Block size used.
    pub block_size: usize,
}

/// Block-bootstrap confidence intervals on every atom of a 3-source PID decomposition.
///
/// Rows of (V, L, D, A) are resampled jointly (same block indices across all four matrices),
/// preserving any cross-variable dependence. `pid3_isx` is recomputed on each resample, and
/// percentile CIs are extracted for each of the 18 atoms.
///
/// # Errors
///
/// Returns [`PidError::RowCountMismatch`] if V, L, D, A do not share a row count, and
/// [`PidError::InvalidConfig`] if `block_size` is not in `1..=n`, `n_boot == 0`, or
/// `alpha` is not in the open interval `(0, 1)`.
pub fn bootstrap_pid3(
    v: MatRef<'_>,
    l: MatRef<'_>,
    d: MatRef<'_>,
    a: MatRef<'_>,
    pid_cfg: &Pid3Config,
    boot_cfg: &BootstrapConfig,
) -> PidResult<BootstrapPid3Result> {
    let n = v.nrows();
    if l.nrows() != n || d.nrows() != n || a.nrows() != n {
        return Err(PidError::RowCountMismatch {
            context: "bootstrap_pid3",
            left_rows: n,
            right_rows: l.nrows().min(d.nrows()).min(a.nrows()),
        });
    }
    if boot_cfg.block_size == 0 || boot_cfg.block_size > n {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_pid3",
            message: "block_size must be in 1..=n",
        });
    }
    if boot_cfg.n_boot == 0 {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_pid3",
            message: "n_boot must be > 0",
        });
    }
    // `alpha` indexes percentile bounds below; outside (0,1) it yields an out-of-range
    // index (alpha >= 2 panics) or an inverted CI (alpha in (1,2)). Reject up front.
    if !(boot_cfg.alpha > 0.0 && boot_cfg.alpha < 1.0) {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_pid3",
            message: "alpha must be in the open interval (0, 1)",
        });
    }

    let dv = v.ncols();
    let dl = l.ncols();
    let dd = d.ncols();
    let da = a.ncols();
    // `block_size` is in `1..=n`, so `n_blocks >= 1`.
    let n_blocks = n / boot_cfg.block_size;

    // Point estimate on original data.
    let point_estimate = pid3_isx(v, l, d, a, pid_cfg)?;
    let n_atoms = point_estimate.atoms.len();

    // Draw every resample's row-index set serially so the RNG stream is unchanged regardless of
    // whether the (expensive) `pid3_isx` evaluations later run in parallel.
    let mut rng = SplitMix64::new(boot_cfg.seed);
    let resample_indices: Vec<Vec<usize>> = (0..boot_cfg.n_boot)
        .map(|_| {
            let mut indices = Vec::with_capacity(n_blocks * boot_cfg.block_size);
            for _ in 0..n_blocks {
                let block_start = (rng.next_u64() as usize % n_blocks) * boot_cfg.block_size;
                for j in 0..boot_cfg.block_size {
                    indices.push(block_start + j);
                }
            }
            indices
        })
        .collect();

    // Evaluate PID on each resample, collected **in resample order**. Each closure reads the
    // shared (immutable) inputs and allocates its own owned resample matrices, so it is pure;
    // collecting by index and only then reducing keeps the parallel path bit-identical.
    let per_resample: Vec<Vec<f64>> =
        crate::par::slice_map_index_ordered(&resample_indices, |indices| {
            let resample = |mat: MatRef<'_>, dim: usize| -> MatOwned {
                let mut data = Vec::with_capacity(indices.len() * dim);
                for &i in indices {
                    data.extend_from_slice(mat.row(i));
                }
                MatOwned::new(data, indices.len(), dim).expect("resample data should be finite")
            };

            let vr = resample(v, dv);
            let lr = resample(l, dl);
            let dr = resample(d, dd);
            let ar = resample(a, da);

            match pid3_isx(vr.as_ref(), lr.as_ref(), dr.as_ref(), ar.as_ref(), pid_cfg) {
                Ok(result) => result.atoms.iter().map(|atom| atom.value).collect(),
                // PID failed on this resample (e.g. degenerate geometry); record NaN per atom.
                Err(_) => vec![f64::NAN; n_atoms],
            }
        });

    // boot_values[atom_idx][boot_idx], filled in resample order (identical to the serial push
    // order), so all downstream summaries are bit-identical.
    let mut boot_values: Vec<Vec<f64>> = vec![Vec::with_capacity(boot_cfg.n_boot); n_atoms];
    for atom_vals in &per_resample {
        for (idx, &val) in atom_vals.iter().enumerate() {
            boot_values[idx].push(val);
        }
    }

    // Build per-atom bootstrap summaries.
    let alpha = boot_cfg.alpha;
    let atoms: Vec<Pid3BootstrapAtom> = point_estimate
        .atoms
        .iter()
        .enumerate()
        .map(|(idx, atom)| {
            let vals = &boot_values[idx];
            // Filter out NaN entries from failed resamples.
            let mut finite: Vec<f64> = vals.iter().copied().filter(|x| x.is_finite()).collect();
            if finite.is_empty() {
                return Pid3BootstrapAtom {
                    antichain: atom.antichain,
                    point_estimate: atom.value,
                    boot_mean: f64::NAN,
                    boot_se: f64::NAN,
                    ci_low: f64::NAN,
                    ci_high: f64::NAN,
                };
            }
            finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = finite.len();
            let mean = finite.iter().sum::<f64>() / m as f64;
            let var = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / m as f64;
            let se = var.sqrt();
            let lo_idx = (((alpha / 2.0) * m as f64).floor() as usize).min(m - 1);
            let hi_idx = (((1.0 - alpha / 2.0) * m as f64).ceil() as usize)
                .saturating_sub(1)
                .min(m - 1);
            Pid3BootstrapAtom {
                antichain: atom.antichain,
                point_estimate: atom.value,
                boot_mean: mean,
                boot_se: se,
                ci_low: finite[lo_idx],
                ci_high: finite[hi_idx],
            }
        })
        .collect();

    Ok(BootstrapPid3Result {
        point_estimate,
        atoms,
        n_boot: boot_cfg.n_boot,
        block_size: boot_cfg.block_size,
    })
}

// ── Permutation test ───────────────────────────────────────────────────────────

/// Result of a permutation test on PID atoms.
#[derive(Debug, Clone)]
pub struct PermutationPid3Atom {
    pub antichain: Antichain3,
    pub observed: f64,
    pub p_value: f64,
    /// Number of permutations that yielded a *finite* atom value and therefore actually
    /// entered this atom's null distribution and p-value. This can be smaller than
    /// [`PermutationPid3Result::n_perm`] (the requested count) when some resamples fail
    /// (e.g. degenerate kNN geometry); it is the denominator's `n_valid` in the
    /// add-one-corrected p-value.
    pub n_perm: usize,
}

/// Result of [`permutation_pid3`].
#[derive(Debug, Clone)]
pub struct PermutationPid3Result {
    pub atoms: Vec<PermutationPid3Atom>,
    /// The number of permutations *requested* (the loop count), not necessarily the number
    /// that produced a finite value for any given atom — see
    /// [`PermutationPid3Atom::n_perm`].
    pub n_perm: usize,
    pub source_shuffled: usize,
}

/// Permutation test for PID atoms: shuffles rows of a single source to build a null
/// distribution, then computes one-sided p-values for each atom.
///
/// `source_idx` selects which source to shuffle (0=V, 1=L, 2=D). Under H0 (source carries
/// no information about target), the shuffled PID atoms should be ~0.
#[allow(clippy::too_many_arguments)]
pub fn permutation_pid3(
    v: MatRef<'_>,
    l: MatRef<'_>,
    d: MatRef<'_>,
    a: MatRef<'_>,
    pid_cfg: &Pid3Config,
    n_perm: usize,
    source_idx: usize,
    seed: u64,
) -> PidResult<PermutationPid3Result> {
    if source_idx > 2 {
        return Err(PidError::InvalidConfig {
            context: "permutation_pid3",
            message: "source_idx must be 0, 1, or 2",
        });
    }
    let n = v.nrows();
    if n_perm == 0 {
        return Err(PidError::InvalidConfig {
            context: "permutation_pid3",
            message: "n_perm must be > 0",
        });
    }

    // Observed PID on real data.
    let observed = pid3_isx(v, l, d, a, pid_cfg)?;

    let mut rng = SplitMix64::new(seed);
    let n_atoms = observed.atoms.len();
    // perm_values[atom_idx][perm_idx]
    let mut perm_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_perm); n_atoms];

    let dv = v.ncols();
    let dl = l.ncols();
    let dd = d.ncols();

    for _ in 0..n_perm {
        // Build a permutation of row indices.
        let mut perm: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle.
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }

        let shuffle = |mat: MatRef<'_>, dim: usize| -> MatOwned {
            let mut data = Vec::with_capacity(n * dim);
            for &i in &perm {
                data.extend_from_slice(mat.row(i));
            }
            MatOwned::new(data, n, dim).expect("shuffle data should be finite")
        };

        let copy_mat = |mat: MatRef<'_>, dim: usize| -> MatOwned {
            let mut data = Vec::with_capacity(n * dim);
            for i in 0..n {
                data.extend_from_slice(mat.row(i));
            }
            MatOwned::new(data, n, dim).expect("copy data should be finite")
        };

        // Only shuffle the selected source; keep others and target intact.
        let vp = if source_idx == 0 {
            shuffle(v, dv)
        } else {
            copy_mat(v, dv)
        };
        let lp = if source_idx == 1 {
            shuffle(l, dl)
        } else {
            copy_mat(l, dl)
        };
        let dp = if source_idx == 2 {
            shuffle(d, dd)
        } else {
            copy_mat(d, dd)
        };

        match pid3_isx(vp.as_ref(), lp.as_ref(), dp.as_ref(), a, pid_cfg) {
            Ok(result) => {
                for (idx, atom) in result.atoms.iter().enumerate() {
                    perm_values[idx].push(atom.value);
                }
            }
            Err(_) => {
                for pv in &mut perm_values {
                    pv.push(f64::NAN);
                }
            }
        }
    }

    let atoms: Vec<PermutationPid3Atom> = observed
        .atoms
        .iter()
        .enumerate()
        .map(|(idx, atom)| {
            let vals = &perm_values[idx];
            let finite: Vec<f64> = vals.iter().copied().filter(|x| x.is_finite()).collect();
            let n_valid = finite.len();
            // One-sided p-value with the add-one correction (Phipson & Smyth 2010), so a
            // finite permutation set can never yield an invalid p-value of exactly 0:
            // p = (1 + #{perm >= observed}) / (1 + n_valid).
            let p_value = if n_valid == 0 {
                f64::NAN
            } else {
                let n_ge = finite.iter().filter(|&&x| x >= atom.value).count();
                (1 + n_ge) as f64 / (1 + n_valid) as f64
            };
            PermutationPid3Atom {
                antichain: atom.antichain,
                observed: atom.value,
                p_value,
                n_perm: n_valid,
            }
        })
        .collect();

    Ok(PermutationPid3Result {
        atoms,
        n_perm,
        source_shuffled: source_idx,
    })
}

// ── PLS cross-validation ───────────────────────────────────────────────────────

/// Result of PLS cross-validation for component selection.
#[derive(Debug, Clone)]
pub struct PlsCvResult {
    /// Predictive power Q² for each candidate component count.
    pub q2: Vec<f64>,
    /// Optimal number of components (maximizing Q²).
    pub best_components: usize,
    /// Total number of candidate components tested.
    pub max_components: usize,
}

/// Leave-one-out cross-validation to select the optimal number of PLS components.
///
/// For each candidate `k` in 1..=max_components, this computes Q² = 1 - PRESS/SS_total,
/// where PRESS is the sum of squared prediction errors from LOO-CV and SS_total is the
/// total sum of squares of the target.
///
/// `x` is the source matrix (n×d_x) and `y` is the target (n×d_y).
pub fn pls_cv_select_components(
    x: MatRef<'_>,
    y: MatRef<'_>,
    max_components: usize,
) -> PidResult<PlsCvResult> {
    let n = x.nrows();
    let d_x = x.ncols();
    let d_y = y.ncols();
    if y.nrows() != n {
        return Err(PidError::RowCountMismatch {
            context: "pls_cv_select_components",
            left_rows: n,
            right_rows: y.nrows(),
        });
    }
    let max_out = d_x.min(n.saturating_sub(1));
    let max_components = max_components.min(max_out);
    if max_components == 0 {
        return Err(PidError::InvalidConfig {
            context: "pls_cv_select_components",
            message: "max_components must be >= 1 after clipping",
        });
    }

    // Compute SS_total.
    let mut y_mean = vec![0.0f64; d_y];
    for i in 0..n {
        let row = y.row(i);
        for (j, ym) in y_mean.iter_mut().enumerate() {
            *ym += row[j];
        }
    }
    for m in &mut y_mean {
        *m /= n as f64;
    }
    let ss_total: f64 = {
        let ym = &y_mean;
        (0..n)
            .flat_map(|i| (0..d_y).map(move |j| (y.row(i)[j] - ym[j]).powi(2)))
            .sum()
    };

    let mut q2 = Vec::with_capacity(max_components);
    for k in 1..=max_components {
        let mut press = 0.0f64;
        // LOO-CV: for each held-out sample, fit PLS on the rest and predict.
        for held_out in 0..n {
            // Build train set (n-1 samples).
            let train_n = n - 1;
            let mut x_train_data = Vec::with_capacity(train_n * d_x);
            let mut y_train_data = Vec::with_capacity(train_n * d_y);
            for i in 0..n {
                if i == held_out {
                    continue;
                }
                x_train_data.extend_from_slice(x.row(i));
                y_train_data.extend_from_slice(y.row(i));
            }
            let x_train =
                MatOwned::new(x_train_data, train_n, d_x).expect("train data should be finite");
            let y_train =
                MatOwned::new(y_train_data, train_n, d_y).expect("train data should be finite");

            match PlsProjector::fit(x_train.as_ref(), y_train.as_ref(), k) {
                Ok(pls) => {
                    // Predict the held-out sample with the model's OWN PLS regression
                    // (`Ŷ = (x−x̄_train)·B + ȳ_train`). This uses the training-fold
                    // target mean as the intercept — never the full-data mean — so the
                    // held-out target does not leak into its own prediction, and the
                    // `W(PᵀW)⁻¹` rotation makes it exact for any number of components.
                    let x_ho =
                        MatRef::new(x.row(held_out), 1, d_x).expect("held-out row should be valid");
                    match pls.predict(x_ho) {
                        Ok(y_hat) => {
                            let pred = y_hat.as_ref().row(0);
                            let ho_row = y.row(held_out);
                            for j in 0..d_y {
                                press += (ho_row[j] - pred[j]).powi(2);
                            }
                        }
                        Err(_) => {
                            press += f64::NAN;
                        }
                    }
                }
                Err(_) => {
                    press += f64::NAN;
                }
            }
        }
        let q2_k = if ss_total > 0.0 && press.is_finite() {
            1.0 - press / ss_total
        } else {
            f64::NEG_INFINITY
        };
        q2.push(q2_k);
    }

    // Select the most parsimonious k achieving the best Q². `max_by` returns the LAST
    // maximum, which biases toward more components on ties; and if every fold failed (all
    // Q² are -inf) it would silently return the largest k. Pick the first k within a small
    // tolerance of the maximum, and error out when no fold produced a finite Q².
    let best = q2.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !best.is_finite() {
        return Err(PidError::NumericalInstability {
            context: "pls_cv_select_components: all CV folds failed (no finite Q²)",
        });
    }
    let best_idx = q2.iter().position(|&v| v >= best - 1e-9).unwrap_or(0);
    let best_components = best_idx + 1;

    Ok(PlsCvResult {
        q2,
        best_components,
        max_components,
    })
}

// ── PLS → Discrete PID3 ──────────────────────────────────────────────────────

/// Configuration for [`pls_project_then_discrete_pid3`].
#[derive(Debug, Clone)]
pub struct PlsDiscretePid3Config {
    /// Number of PLS latent components to extract.
    pub pls_components: usize,
    /// Number of equal-width bins for discrete PID.
    pub num_bins: usize,
}

/// Result of [`pls_project_then_discrete_pid3`].
#[derive(Debug, Clone)]
pub struct PlsDiscretePid3Result {
    pub pid: crate::discrete_pid::DiscretePid3Result,
    pub pls_components: usize,
    pub num_bins: usize,
    pub input_dims: [usize; 4],
    pub projected_dim: usize,
}

/// Fit per-source PLS projectors, project all four matrices into a low-dimensional
/// task-relevant subspace, then run discrete PID3 on the quantized projections.
///
/// This is the recommended escape hatch when continuous kNN-based PID fails due to
/// high ambient dimension or distance concentration.
pub fn pls_project_then_discrete_pid3(
    v: MatRef<'_>,
    l: MatRef<'_>,
    d: MatRef<'_>,
    a: MatRef<'_>,
    cfg: &PlsDiscretePid3Config,
) -> PidResult<PlsDiscretePid3Result> {
    let n = v.nrows();
    if l.nrows() != n || d.nrows() != n || a.nrows() != n {
        return Err(PidError::RowCountMismatch {
            context: "pls_project_then_discrete_pid3",
            left_rows: n,
            right_rows: l.nrows().min(d.nrows()).min(a.nrows()),
        });
    }

    // Per-source PLS projectors.
    let v_proj = PlsProjector::fit(v, a, cfg.pls_components)?.transform(v)?;
    let l_proj = PlsProjector::fit(l, a, cfg.pls_components)?.transform(l)?;
    let d_proj = PlsProjector::fit(d, a, cfg.pls_components)?.transform(d)?;
    let vld = concat_horiz(concat_horiz(v, l)?.as_ref(), d)?;
    let a_proj = PlsProjector::fit(a, vld.as_ref(), cfg.pls_components)?.transform(a)?;

    let pid = discrete_pid3(
        v_proj.as_ref(),
        l_proj.as_ref(),
        d_proj.as_ref(),
        a_proj.as_ref(),
        cfg.num_bins,
    )?;

    Ok(PlsDiscretePid3Result {
        pid,
        pls_components: cfg.pls_components,
        num_bins: cfg.num_bins,
        input_dims: [v.ncols(), l.ncols(), d.ncols(), a.ncols()],
        projected_dim: cfg.pls_components,
    })
}

// ── Multi-pair PID2 screening ──────────────────────────────────────────────────

/// A single PID2 screening result for a pair of sources.
#[derive(Debug, Clone)]
pub struct Pid2ScreenEntry {
    /// Source pair indices (i, j) into the sources list.
    pub source_i: usize,
    pub source_j: usize,
    pub result: Pid2Result,
}

/// Screen all pairs of sources with PID2, returning one entry per pair.
///
/// `sources` is a slice of matrices, each n×d_i. `target` is the target matrix.
/// This computes PID2 for all C(n_sources, 2) pairs and sorts them by descending
/// synergy.
pub fn screen_pid2_pairs(
    sources: &[MatRef<'_>],
    target: MatRef<'_>,
    cfg: &Pid2Config,
) -> PidResult<Vec<Pid2ScreenEntry>> {
    let n = target.nrows();
    let n_src = sources.len();
    let mut entries = Vec::with_capacity(n_src * (n_src.saturating_sub(1)) / 2);

    // Validate every source up front. A per-pair `continue` for source `j` could only
    // mask a row-count mismatch until the outer loop reached that index and hard-errored,
    // so the outcome was identical — validate once, then keep the pair loop clean.
    for s in sources {
        if s.nrows() != n {
            return Err(PidError::RowCountMismatch {
                context: "screen_pid2_pairs",
                left_rows: n,
                right_rows: s.nrows(),
            });
        }
    }

    for i in 0..n_src {
        for j in (i + 1)..n_src {
            match pid2_isx(sources[i], sources[j], target, cfg) {
                Ok(result) => {
                    entries.push(Pid2ScreenEntry {
                        source_i: i,
                        source_j: j,
                        result,
                    });
                }
                Err(_) => {
                    // Skip pairs that fail (e.g. degenerate geometry).
                }
            }
        }
    }

    // Sort by descending synergy.
    entries.sort_by(|a, b| {
        b.result
            .synergy
            .partial_cmp(&a.result.synergy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(entries)
}

// ── Generic row-resampling uncertainty helpers ─────────────────────────────

/// Bootstrap summary for one scalar statistic from [`bootstrap_rows_stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct RowBootstrapStat {
    /// Statistic evaluated on the original (un-resampled, un-jittered) data.
    pub point_estimate: f64,
    /// Mean of the bootstrap distribution (finite resamples only).
    pub boot_mean: f64,
    /// Standard error (std of bootstrap distribution; finite resamples only).
    pub boot_se: f64,
    /// Lower percentile CI bound.
    pub ci_low: f64,
    /// Upper percentile CI bound.
    pub ci_high: f64,
    /// Number of resamples attempted.
    pub n_attempted: usize,
    /// Number of resamples on which this statistic evaluated to a finite value.
    pub n_valid: usize,
}

/// Row-resampling scheme for [`bootstrap_rows_stats`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowResampleScheme {
    /// Moving-block bootstrap **with replacement** plus deterministic tie-breaking
    /// jitter on the resampled rows.
    ///
    /// With-replacement resampling guarantees duplicate rows. The KSG-family
    /// estimators in this crate intentionally reject zero kNN radii caused by
    /// duplicates (`PidError::NumericalInstability`), so `jitter_rel` must be > 0
    /// for kNN statistics (each resampled element gets an additive uniform
    /// perturbation of amplitude `jitter_rel × column_std`, column std measured on
    /// the original data — the tie-breaking noise recommended by Kraskov et al.
    /// 2004 §II).
    ///
    /// **Caveat (empirically pinned by a test):** even with jitter, duplicated
    /// points distort kNN local-density statistics, shifting the bootstrap mean of
    /// KSG MI by a non-negligible amount. Naive with-replacement bootstrap is known
    /// to be unreliable for kNN information estimators; prefer
    /// [`RowResampleScheme::Subsample`] for KSG-based statistics.
    BlockBootstrapJitter {
        /// Relative jitter amplitude (e.g. `1e-9`); 0 disables jitter.
        jitter_rel: f64,
    },
    /// Politis–Romano-style subsampling **without replacement**: each resample
    /// draws `subsample_len / block_size` *distinct* contiguous blocks, yielding a
    /// duplicate-free subsample of (approximately) `subsample_len` rows.
    ///
    /// Duplicate-free resamples are safe for kNN estimators with no jitter. The
    /// resulting percentile interval describes the sampling variability of the
    /// statistic at sample size `m = subsample_len`; for `m < n` its width
    /// overstates the n-sample uncertainty (roughly by `sqrt(n/m)` for
    /// root-n-rate statistics), so treat it as a conservative interval and report
    /// `m` alongside it.
    Subsample {
        /// Subsample length `m` (rows; rounded down to a multiple of `block_size`).
        subsample_len: usize,
    },
}

/// Result of [`bootstrap_rows_stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct RowBootstrapResult {
    /// Per-statistic summaries, in the order returned by the statistic closure.
    pub stats: Vec<RowBootstrapStat>,
    /// Number of resamples attempted.
    pub n_boot: usize,
    /// Block size used for the block-level resampling.
    pub block_size: usize,
    /// Resampling scheme used.
    pub scheme: RowResampleScheme,
}

/// Joint block-level row resampling over several aligned matrices, for a
/// vector-valued statistic, under a configurable [`RowResampleScheme`].
///
/// All matrices must share the same row count `n`. Each resample draws contiguous
/// blocks of row indices (the same indices are applied to every matrix, preserving
/// cross-variable alignment), then evaluates `stat` on the resampled matrices.
/// Percentile intervals are computed from the finite resample statistics.
///
/// For KSG/kNN-based statistics use [`RowResampleScheme::Subsample`]; see the
/// scheme docs for why with-replacement bootstrap is problematic there.
///
/// Failed resamples (statistic returns `Err`, or a non-finite entry) are recorded
/// via `n_valid`, not silently dropped: callers should treat a low
/// `n_valid / n_attempted` ratio as an estimator-regime warning.
///
/// # Errors
///
/// Returns an error if the matrices are empty or misaligned, if the configuration is
/// invalid, or if the statistic fails **on the original data** (a failed point
/// estimate makes the resampling distribution meaningless).
pub fn bootstrap_rows_stats<F>(
    mats: &[MatRef<'_>],
    cfg: &BootstrapConfig,
    scheme: RowResampleScheme,
    stat: F,
) -> PidResult<RowBootstrapResult>
where
    F: Fn(&[MatRef<'_>]) -> PidResult<Vec<f64>>,
{
    if mats.is_empty() {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_rows_stats",
            message: "mats must not be empty",
        });
    }
    let n = mats[0].nrows();
    for m in mats {
        if m.nrows() != n {
            return Err(PidError::RowCountMismatch {
                context: "bootstrap_rows_stats",
                left_rows: n,
                right_rows: m.nrows(),
            });
        }
    }
    if cfg.n_boot == 0 {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_rows_stats",
            message: "n_boot must be > 0",
        });
    }
    if cfg.block_size == 0 || cfg.block_size > n {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_rows_stats",
            message: "block_size must be in 1..=n",
        });
    }
    if !(cfg.alpha > 0.0 && cfg.alpha < 1.0) {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_rows_stats",
            message: "alpha must be in (0, 1)",
        });
    }
    let n_blocks = n / cfg.block_size;
    // Number of distinct blocks to draw per resample, and whether to draw with
    // replacement, depend on the scheme.
    let (blocks_per_resample, with_replacement, jitter_rel) = match scheme {
        RowResampleScheme::BlockBootstrapJitter { jitter_rel } => {
            if !jitter_rel.is_finite() || jitter_rel < 0.0 {
                return Err(PidError::InvalidConfig {
                    context: "bootstrap_rows_stats",
                    message: "jitter_rel must be finite and >= 0",
                });
            }
            (n_blocks, true, jitter_rel)
        }
        RowResampleScheme::Subsample { subsample_len } => {
            let blocks = subsample_len / cfg.block_size;
            if blocks == 0 {
                return Err(PidError::InvalidConfig {
                    context: "bootstrap_rows_stats",
                    message: "subsample_len must be >= block_size",
                });
            }
            if blocks > n_blocks {
                return Err(PidError::InvalidConfig {
                    context: "bootstrap_rows_stats",
                    message: "subsample_len must be <= n (in whole blocks)",
                });
            }
            (blocks, false, 0.0)
        }
    };

    let point = stat(mats)?;
    if point.is_empty() {
        return Err(PidError::InvalidConfig {
            context: "bootstrap_rows_stats",
            message: "stat must return at least one value",
        });
    }
    let n_stats = point.len();

    // Per-matrix, per-column population std on the original data (jitter scale).
    let col_stds: Vec<Vec<f64>> = mats
        .iter()
        .map(|m| {
            let d = m.ncols();
            let mut mean = vec![0.0f64; d];
            for i in 0..n {
                for (j, v) in m.row(i).iter().enumerate() {
                    mean[j] += v;
                }
            }
            for v in &mut mean {
                *v /= n as f64;
            }
            let mut var = vec![0.0f64; d];
            for i in 0..n {
                for (j, v) in m.row(i).iter().enumerate() {
                    var[j] += (v - mean[j]) * (v - mean[j]);
                }
            }
            var.iter().map(|v| (v / n as f64).sqrt()).collect()
        })
        .collect();

    let mut rng = SplitMix64::new(cfg.seed);
    // boot_values[stat_idx][resample_idx], only finite values are pushed.
    let mut boot_values: Vec<Vec<f64>> = vec![Vec::with_capacity(cfg.n_boot); n_stats];

    for _ in 0..cfg.n_boot {
        // Draw `blocks_per_resample` block starts, with or without replacement.
        let mut block_ids: Vec<usize> = Vec::with_capacity(blocks_per_resample);
        if with_replacement {
            for _ in 0..blocks_per_resample {
                block_ids.push(rng.next_u64() as usize % n_blocks);
            }
        } else {
            // Partial Fisher–Yates over [0, n_blocks): draw distinct block ids.
            let mut pool: Vec<usize> = (0..n_blocks).collect();
            for k in 0..blocks_per_resample {
                let j = k + (rng.next_u64() as usize) % (n_blocks - k);
                pool.swap(k, j);
                block_ids.push(pool[k]);
            }
            // Keep temporal order of subsampled blocks for block-structure fidelity.
            block_ids.sort_unstable();
        }

        let mut indices = Vec::with_capacity(block_ids.len() * cfg.block_size);
        for &b in &block_ids {
            let block_start = b * cfg.block_size;
            for j in 0..cfg.block_size {
                indices.push(block_start + j);
            }
        }

        let mut owned: Vec<MatOwned> = Vec::with_capacity(mats.len());
        for (m_idx, m) in mats.iter().enumerate() {
            let d = m.ncols();
            let mut data = Vec::with_capacity(indices.len() * d);
            for &i in &indices {
                data.extend_from_slice(m.row(i));
            }
            if jitter_rel > 0.0 {
                for (flat_idx, v) in data.iter_mut().enumerate() {
                    let col = flat_idx % d;
                    let scale = jitter_rel * col_stds[m_idx][col];
                    if scale > 0.0 {
                        // Uniform in [-scale, scale]: tie-breaking only, shape irrelevant.
                        let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                        *v += scale * (2.0 * u - 1.0);
                    }
                }
            }
            owned.push(MatOwned::new(data, indices.len(), d).map_err(|_| {
                PidError::InvalidConfig {
                    context: "bootstrap_rows_stats",
                    message: "resampled data must be finite",
                }
            })?);
        }
        let refs: Vec<MatRef<'_>> = owned.iter().map(|m| m.as_ref()).collect();
        if let Ok(values) = stat(&refs) {
            if values.len() != n_stats {
                return Err(PidError::InvalidConfig {
                    context: "bootstrap_rows_stats",
                    message: "stat returned an inconsistent number of values",
                });
            }
            for (idx, value) in values.into_iter().enumerate() {
                if value.is_finite() {
                    boot_values[idx].push(value);
                }
            }
        }
    }

    let stats = point
        .iter()
        .enumerate()
        .map(|(idx, &point_estimate)| {
            let vals = &mut boot_values[idx];
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = vals.len();
            if m == 0 {
                return RowBootstrapStat {
                    point_estimate,
                    boot_mean: f64::NAN,
                    boot_se: f64::NAN,
                    ci_low: f64::NAN,
                    ci_high: f64::NAN,
                    n_attempted: cfg.n_boot,
                    n_valid: 0,
                };
            }
            let mean = vals.iter().sum::<f64>() / m as f64;
            let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / m as f64;
            let lo_idx = ((cfg.alpha / 2.0) * m as f64).floor() as usize;
            let hi_idx = (((1.0 - cfg.alpha / 2.0) * m as f64).ceil() as usize)
                .saturating_sub(1)
                .min(m - 1);
            RowBootstrapStat {
                point_estimate,
                boot_mean: mean,
                boot_se: var.sqrt(),
                ci_low: vals[lo_idx],
                ci_high: vals[hi_idx],
                n_attempted: cfg.n_boot,
                n_valid: m,
            }
        })
        .collect();

    Ok(RowBootstrapResult {
        stats,
        n_boot: cfg.n_boot,
        block_size: cfg.block_size,
        scheme,
    })
}

/// Bootstrap confidence intervals for the averaged 2-source discrete SxPID (`i^sx_∩`) atoms.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteSxPid2BootstrapResult {
    pub redundancy: RowBootstrapStat,
    pub unique_s1: RowBootstrapStat,
    pub unique_s2: RowBootstrapStat,
    pub synergy: RowBootstrapStat,
    pub n_boot: usize,
    pub block_size: usize,
}

/// Dependence-aware bootstrap confidence intervals for the averaged discrete SxPID atoms
/// (`discrete_sxpid2`).
///
/// Resampling uses a moving-block bootstrap **with replacement and no jitter**
/// ([`RowResampleScheme::BlockBootstrapJitter`] with `jitter_rel = 0`): unlike the kNN/KSG
/// estimators, the discrete (counting-based) SxPID is unaffected by duplicate rows, and jitter
/// would corrupt the discrete labels. Set `cfg.block_size = 1` for i.i.d. data, or a larger block
/// for autocorrelated (e.g. time-series) data. The percentile interval is the
/// `(1 − cfg.alpha)` two-sided CI of each atom over the resamples.
///
/// This mirrors the uncertainty story IDTxl provides for PID via its surrogate framework.
pub fn bootstrap_discrete_sxpid2(
    s1: MatRef<'_>,
    s2: MatRef<'_>,
    t: MatRef<'_>,
    num_bins: usize,
    cfg: &BootstrapConfig,
) -> PidResult<DiscreteSxPid2BootstrapResult> {
    let stat = |mats: &[MatRef<'_>]| -> PidResult<Vec<f64>> {
        let r = discrete_sxpid2(mats[0], mats[1], mats[2], num_bins)?;
        Ok(vec![r.red.net, r.unq1.net, r.unq2.net, r.syn.net])
    };
    let res = bootstrap_rows_stats(
        &[s1, s2, t],
        cfg,
        RowResampleScheme::BlockBootstrapJitter { jitter_rel: 0.0 },
        stat,
    )?;
    let mut it = res.stats.into_iter();
    let mut next = || {
        it.next().ok_or(PidError::InvalidConfig {
            context: "bootstrap_discrete_sxpid2",
            message: "missing bootstrap statistic",
        })
    };
    Ok(DiscreteSxPid2BootstrapResult {
        redundancy: next()?,
        unique_s1: next()?,
        unique_s2: next()?,
        synergy: next()?,
        n_boot: res.n_boot,
        block_size: res.block_size,
    })
}

/// Result of [`permutation_rows_pvalue`].
#[derive(Debug, Clone, PartialEq)]
pub struct RowPermutationStat {
    /// Statistic on the original data.
    pub observed: f64,
    /// Add-one one-sided p-value: `(1 + #{perm >= observed}) / (1 + n_valid)`.
    pub p_value: f64,
    /// Number of permutations attempted.
    pub n_attempted: usize,
    /// Number of permutations on which the statistic evaluated to a finite value.
    pub n_valid: usize,
    /// Index (into `mats`) of the matrix whose rows were shuffled.
    pub shuffled_index: usize,
}

/// One-sided permutation test on a scalar statistic of several aligned matrices.
///
/// Shuffles the rows of `mats[shuffled_index]` (Fisher–Yates, seeded) while keeping
/// every other matrix fixed, re-evaluating `stat` on each permuted dataset. Under the
/// null hypothesis that the shuffled variable carries no information about the rest,
/// the observed statistic should be exchangeable with the permuted ones.
///
/// Unlike [`permutation_pid3`] (kept as-is for backward compatibility), this helper
/// uses the add-one Monte Carlo p-value `(b + 1) / (m + 1)` (Phipson & Smyth 2010),
/// which is a valid p-value (never exactly zero) and is the convention the
/// Experiment 0 gate relies on.
///
/// Permuting rows introduces no duplicate rows, so no jitter is needed here.
///
/// # Errors
///
/// Returns an error on misaligned/empty inputs, an out-of-range `shuffled_index`,
/// `n_perm == 0`, or if the statistic fails or is non-finite on the original data.
pub fn permutation_rows_pvalue<F>(
    mats: &[MatRef<'_>],
    shuffled_index: usize,
    n_perm: usize,
    seed: u64,
    stat: F,
) -> PidResult<RowPermutationStat>
where
    F: Fn(&[MatRef<'_>]) -> PidResult<f64>,
{
    if mats.is_empty() {
        return Err(PidError::InvalidConfig {
            context: "permutation_rows_pvalue",
            message: "mats must not be empty",
        });
    }
    let n = mats[0].nrows();
    for m in mats {
        if m.nrows() != n {
            return Err(PidError::RowCountMismatch {
                context: "permutation_rows_pvalue",
                left_rows: n,
                right_rows: m.nrows(),
            });
        }
    }
    if shuffled_index >= mats.len() {
        return Err(PidError::InvalidConfig {
            context: "permutation_rows_pvalue",
            message: "shuffled_index out of range",
        });
    }
    if n_perm == 0 {
        return Err(PidError::InvalidConfig {
            context: "permutation_rows_pvalue",
            message: "n_perm must be > 0",
        });
    }

    let observed = stat(mats)?;
    if !observed.is_finite() {
        return Err(PidError::InvalidConfig {
            context: "permutation_rows_pvalue",
            message: "observed statistic must be finite",
        });
    }

    let mut rng = SplitMix64::new(seed);
    let shuffled_dim = mats[shuffled_index].ncols();
    let mut n_valid = 0usize;
    let mut n_geq = 0usize;

    for _ in 0..n_perm {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let mut data = Vec::with_capacity(n * shuffled_dim);
        for &i in &perm {
            data.extend_from_slice(mats[shuffled_index].row(i));
        }
        let shuffled =
            MatOwned::new(data, n, shuffled_dim).map_err(|_| PidError::InvalidConfig {
                context: "permutation_rows_pvalue",
                message: "shuffled data must be finite",
            })?;
        let mut refs: Vec<MatRef<'_>> = mats.to_vec();
        refs[shuffled_index] = shuffled.as_ref();
        if let Ok(value) = stat(&refs) {
            if value.is_finite() {
                n_valid += 1;
                if value >= observed {
                    n_geq += 1;
                }
            }
        }
    }

    let p_value = if n_valid == 0 {
        f64::NAN
    } else {
        (1.0 + n_geq as f64) / (1.0 + n_valid as f64)
    };

    Ok(RowPermutationStat {
        observed,
        p_value,
        n_attempted: n_perm,
        n_valid,
        shuffled_index,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::SplitMix64;

    /// Helper: generate synthetic (V, L, D, A) data where V and L share signal about A,
    /// D is pure noise.
    fn make_vlda(n: usize, seed: u64) -> (MatOwned, MatOwned, MatOwned, MatOwned) {
        let mut rng = SplitMix64::new(seed);
        let mut v_data = Vec::with_capacity(n * 3);
        let mut l_data = Vec::with_capacity(n * 3);
        let mut d_data = Vec::with_capacity(n * 2);
        let mut a_data = Vec::with_capacity(n);
        for _ in 0..n {
            let signal = rng.normal();
            // V carries signal in dim 0, noise in dims 1,2
            v_data.push(signal + 0.1 * rng.normal());
            v_data.push(rng.normal());
            v_data.push(rng.normal());
            // L carries signal in dim 0, noise in dims 1,2
            l_data.push(signal + 0.1 * rng.normal());
            l_data.push(rng.normal());
            l_data.push(rng.normal());
            // D is pure noise
            d_data.push(rng.normal());
            d_data.push(rng.normal());
            // A = signal + small noise
            a_data.push(signal + 0.05 * rng.normal());
        }
        let v = MatOwned::new(v_data, n, 3).unwrap();
        let l = MatOwned::new(l_data, n, 3).unwrap();
        let d = MatOwned::new(d_data, n, 2).unwrap();
        let a = MatOwned::new(a_data, n, 1).unwrap();
        (v, l, d, a)
    }

    #[test]
    fn pls_project_then_pid3_runs_and_returns_18_atoms() {
        let (v, l, d, a) = make_vlda(60, 42);
        let cfg = PlsPid3Config {
            pls_components: 1,
            pid_cfg: Pid3Config::default(),
        };
        let result =
            pls_project_then_pid3(v.as_ref(), l.as_ref(), d.as_ref(), a.as_ref(), &cfg).unwrap();
        // The PID result has 18 atoms for 3 sources.
        assert_eq!(result.pid.atoms.len(), 18);
        assert_eq!(result.pls_components, 1);
        assert_eq!(result.projected_dim, 1);
        assert_eq!(result.input_dims, [3, 3, 2, 1]);
    }

    #[test]
    fn pls_project_then_pid3_rejects_mismatched_rows() {
        let v = MatOwned::new(vec![0.0; 30], 10, 3).unwrap();
        let l = MatOwned::new(vec![0.0; 15], 5, 3).unwrap(); // Wrong row count
        let d = MatOwned::new(vec![0.0; 20], 10, 2).unwrap();
        let a = MatOwned::new(vec![0.0; 10], 10, 1).unwrap();
        let cfg = PlsPid3Config {
            pls_components: 1,
            pid_cfg: Pid3Config::default(),
        };
        assert!(
            pls_project_then_pid3(v.as_ref(), l.as_ref(), d.as_ref(), a.as_ref(), &cfg,).is_err()
        );
    }

    #[test]
    fn bootstrap_pid3_returns_ci_for_each_atom() {
        let (v, l, d, a) = make_vlda(80, 77);
        let pid_cfg = Pid3Config::default();
        let boot_cfg = BootstrapConfig {
            n_boot: 20, // Small for test speed
            block_size: 10,
            seed: 42,
            alpha: 0.1,
        };
        let result = bootstrap_pid3(
            v.as_ref(),
            l.as_ref(),
            d.as_ref(),
            a.as_ref(),
            &pid_cfg,
            &boot_cfg,
        )
        .unwrap();

        // 18 atoms for 3-source PID.
        assert_eq!(result.atoms.len(), 18);
        assert_eq!(result.point_estimate.atoms.len(), 18);
        assert_eq!(result.n_boot, 20);
        assert_eq!(result.block_size, 10);

        // Each atom's CI should bracket the point estimate (for most atoms with finite CIs).
        let finite_atoms: Vec<_> = result
            .atoms
            .iter()
            .filter(|a| a.ci_low.is_finite() && a.ci_high.is_finite())
            .collect();
        assert!(
            !finite_atoms.is_empty(),
            "at least some bootstrap atoms should have finite CIs"
        );
        for atom in &finite_atoms {
            assert!(
                atom.ci_low <= atom.ci_high,
                "CI low ({}) must be <= CI high ({})",
                atom.ci_low,
                atom.ci_high
            );
        }
    }

    #[test]
    fn bootstrap_pid3_rejects_out_of_range_alpha() {
        let (v, l, d, a) = make_vlda(40, 5);
        let pid_cfg = Pid3Config::default();
        // alpha >= 1 previously produced an out-of-range percentile index (alpha >= 2 panicked);
        // every alpha outside the open interval (0, 1) must now be a clean Err.
        for bad_alpha in [0.0, 1.0, 2.0, -0.1] {
            let boot_cfg = BootstrapConfig {
                n_boot: 5,
                block_size: 8,
                seed: 1,
                alpha: bad_alpha,
            };
            let res = bootstrap_pid3(
                v.as_ref(),
                l.as_ref(),
                d.as_ref(),
                a.as_ref(),
                &pid_cfg,
                &boot_cfg,
            );
            assert!(
                res.is_err(),
                "alpha={bad_alpha} must be rejected (require 0 < alpha < 1)"
            );
        }
        // A valid alpha still succeeds.
        let ok_cfg = BootstrapConfig {
            n_boot: 5,
            block_size: 8,
            seed: 1,
            alpha: 0.05,
        };
        assert!(bootstrap_pid3(
            v.as_ref(),
            l.as_ref(),
            d.as_ref(),
            a.as_ref(),
            &pid_cfg,
            &ok_cfg
        )
        .is_ok());
    }

    #[test]
    fn bootstrap_pid3_is_deterministic() {
        let (v, l, d, a) = make_vlda(60, 123);
        let pid_cfg = Pid3Config::default();
        let boot_cfg = BootstrapConfig {
            n_boot: 10,
            block_size: 10,
            seed: 99,
            alpha: 0.05,
        };
        let r1 = bootstrap_pid3(
            v.as_ref(),
            l.as_ref(),
            d.as_ref(),
            a.as_ref(),
            &pid_cfg,
            &boot_cfg,
        )
        .unwrap();
        let r2 = bootstrap_pid3(
            v.as_ref(),
            l.as_ref(),
            d.as_ref(),
            a.as_ref(),
            &pid_cfg,
            &boot_cfg,
        )
        .unwrap();

        // Same seed → same bootstrap results.
        for (a1, a2) in r1.atoms.iter().zip(r2.atoms.iter()) {
            assert_eq!(a1.point_estimate, a2.point_estimate);
            if a1.boot_mean.is_finite() {
                assert!(
                    (a1.boot_mean - a2.boot_mean).abs() < 1e-12,
                    "bootstrap must be deterministic"
                );
            }
        }
    }

    #[test]
    fn bootstrap_pid3_point_estimate_matches_direct() {
        let (v, l, d, a) = make_vlda(60, 55);
        let pid_cfg = Pid3Config::default();
        let boot_cfg = BootstrapConfig {
            n_boot: 5,
            block_size: 10,
            seed: 0,
            alpha: 0.05,
        };
        let result = bootstrap_pid3(
            v.as_ref(),
            l.as_ref(),
            d.as_ref(),
            a.as_ref(),
            &pid_cfg,
            &boot_cfg,
        )
        .unwrap();

        // Point estimate should match a direct pid3_isx call.
        let direct = pid3_isx(v.as_ref(), l.as_ref(), d.as_ref(), a.as_ref(), &pid_cfg).unwrap();
        for (boot_atom, direct_atom) in result.point_estimate.atoms.iter().zip(direct.atoms.iter())
        {
            assert_eq!(boot_atom.antichain, direct_atom.antichain);
            assert!(
                (boot_atom.value - direct_atom.value).abs() < 1e-12,
                "point estimate must match direct pid3_isx"
            );
        }
    }

    #[test]
    fn permutation_pid3_produces_p_values() {
        let (v, l, d, a) = make_vlda(60, 42);
        let pid_cfg = Pid3Config::default();
        let result = permutation_pid3(
            v.as_ref(),
            l.as_ref(),
            d.as_ref(),
            a.as_ref(),
            &pid_cfg,
            10, // Small for test speed.
            2,  // Shuffle D (noise source → p-values should be high).
            42,
        )
        .unwrap();
        assert_eq!(result.atoms.len(), 18);
        assert_eq!(result.n_perm, 10);
        assert_eq!(result.source_shuffled, 2);
        // D is pure noise, so permuting it should yield high p-values.
        let finite_atoms: Vec<_> = result
            .atoms
            .iter()
            .filter(|a| a.p_value.is_finite())
            .collect();
        assert!(!finite_atoms.is_empty());
    }

    #[test]
    fn permutation_pid3_rejects_bad_source_idx() {
        let (v, l, d, a) = make_vlda(60, 42);
        let pid_cfg = Pid3Config::default();
        assert!(permutation_pid3(
            v.as_ref(),
            l.as_ref(),
            d.as_ref(),
            a.as_ref(),
            &pid_cfg,
            5,
            3, // Invalid source index.
            0
        )
        .is_err());
    }

    #[test]
    fn pls_cv_selects_at_least_one_component() {
        let n = 50;
        let mut rng = SplitMix64::new(77);
        let mut x_data = Vec::with_capacity(n * 5);
        let mut y_data = Vec::with_capacity(n);
        for _ in 0..n {
            let sig = rng.normal();
            x_data.push(sig + 0.1 * rng.normal());
            for _ in 1..5 {
                x_data.push(rng.normal());
            }
            y_data.push(sig);
        }
        let x = MatRef::new(&x_data, n, 5).unwrap();
        let y = MatRef::new(&y_data, n, 1).unwrap();
        let result = pls_cv_select_components(x, y, 3).unwrap();
        assert_eq!(result.q2.len(), 3);
        assert!(result.best_components >= 1);
        assert!(result.best_components <= 3);
    }

    #[test]
    fn pls_project_then_discrete_pid3_runs() {
        let (v, l, d, a) = make_vlda(60, 42);
        let cfg = PlsDiscretePid3Config {
            pls_components: 1,
            num_bins: 8,
        };
        let result =
            pls_project_then_discrete_pid3(v.as_ref(), l.as_ref(), d.as_ref(), a.as_ref(), &cfg)
                .unwrap();
        assert_eq!(result.pid.atoms.len(), 18);
        assert_eq!(result.pls_components, 1);
        assert_eq!(result.num_bins, 8);
        assert_eq!(result.projected_dim, 1);
        assert_eq!(result.input_dims, [3, 3, 2, 1]);
    }

    #[test]
    fn screen_pid2_pairs_returns_all_pairs() {
        let n = 60;
        let mut rng = SplitMix64::new(42);
        let mut s0_data = Vec::with_capacity(n * 2);
        let mut s1_data = Vec::with_capacity(n * 2);
        let mut s2_data = Vec::with_capacity(n);
        let mut t_data = Vec::with_capacity(n);
        for _ in 0..n {
            let sig = rng.normal();
            s0_data.push(sig + 0.1 * rng.normal());
            s0_data.push(rng.normal());
            s1_data.push(sig + 0.1 * rng.normal());
            s1_data.push(rng.normal());
            s2_data.push(rng.normal());
            t_data.push(sig + 0.05 * rng.normal());
        }
        let s0 = MatOwned::new(s0_data, n, 2).unwrap();
        let s1 = MatOwned::new(s1_data, n, 2).unwrap();
        let s2 = MatOwned::new(s2_data, n, 1).unwrap();
        let t = MatOwned::new(t_data, n, 1).unwrap();
        let sources: Vec<MatRef<'_>> = vec![s0.as_ref(), s1.as_ref(), s2.as_ref()];
        let cfg = Pid2Config::default();
        let entries = screen_pid2_pairs(&sources, t.as_ref(), &cfg).unwrap();
        // 3 sources → C(3,2) = 3 pairs.
        assert_eq!(entries.len(), 3);
        // Sorted by descending synergy.
        for w in entries.windows(2) {
            assert!(w[0].result.synergy >= w[1].result.synergy);
        }
    }

    /// Helper: paired (x, y) columns with y = x + noise, returned as 1-col matrices.
    fn make_linear_pair(n: usize, noise: f64, seed: u64) -> (MatOwned, MatOwned) {
        let mut rng = SplitMix64::new(seed);
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.normal();
            x_data.push(x);
            y_data.push(x + noise * rng.normal());
        }
        (
            MatOwned::new(x_data, n, 1).unwrap(),
            MatOwned::new(y_data, n, 1).unwrap(),
        )
    }

    fn pearson_stat(mats: &[MatRef<'_>]) -> PidResult<Vec<f64>> {
        let x = mats[0];
        let y = mats[1];
        let n = x.nrows() as f64;
        let mx: f64 = (0..x.nrows()).map(|i| x.row(i)[0]).sum::<f64>() / n;
        let my: f64 = (0..y.nrows()).map(|i| y.row(i)[0]).sum::<f64>() / n;
        let mut cov = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for i in 0..x.nrows() {
            let a = x.row(i)[0] - mx;
            let b = y.row(i)[0] - my;
            cov += a * b;
            vx += a * a;
            vy += b * b;
        }
        Ok(vec![cov / (vx.sqrt() * vy.sqrt())])
    }

    #[test]
    fn bootstrap_rows_stats_is_deterministic_and_brackets_point() {
        let (x, y) = make_linear_pair(200, 0.5, 7);
        let cfg = BootstrapConfig {
            n_boot: 64,
            block_size: 1,
            seed: 11,
            alpha: 0.05,
        };
        let mats = [x.as_ref(), y.as_ref()];
        let scheme = RowResampleScheme::Subsample { subsample_len: 150 };
        let a = bootstrap_rows_stats(&mats, &cfg, scheme, pearson_stat).unwrap();
        let b = bootstrap_rows_stats(&mats, &cfg, scheme, pearson_stat).unwrap();
        assert_eq!(a, b);
        let s = &a.stats[0];
        assert_eq!(s.n_attempted, 64);
        assert_eq!(s.n_valid, 64);
        assert!(s.ci_low <= s.point_estimate + 0.05);
        assert!(s.ci_high >= s.point_estimate - 0.05);
        assert!(s.boot_se > 0.0 && s.boot_se < 0.2, "se={}", s.boot_se);
    }

    #[test]
    fn bootstrap_rows_stats_subsample_is_duplicate_free_for_ksg() {
        // Subsampling draws *distinct* rows, so KSG (which rejects duplicate-induced
        // zero kNN radii) succeeds with no jitter on every resample.
        let (x, y) = make_linear_pair(200, 0.5, 13);
        let ksg_cfg = crate::KsgConfig::default();
        let stat = |mats: &[MatRef<'_>]| -> PidResult<Vec<f64>> {
            Ok(vec![crate::ksg_mi(mats[0], mats[1], &ksg_cfg)?])
        };
        let cfg = BootstrapConfig {
            n_boot: 32,
            block_size: 1,
            seed: 3,
            alpha: 0.05,
        };
        let mats = [x.as_ref(), y.as_ref()];
        let sub = bootstrap_rows_stats(
            &mats,
            &cfg,
            RowResampleScheme::Subsample { subsample_len: 120 },
            stat,
        )
        .unwrap();
        assert_eq!(sub.stats[0].n_valid, 32);
        assert!(sub.stats[0].ci_low.is_finite());
        assert!(sub.stats[0].ci_high >= sub.stats[0].ci_low);
        assert!(sub.stats[0].ci_low <= sub.stats[0].point_estimate);
        assert!(sub.stats[0].ci_high >= sub.stats[0].point_estimate);
    }

    #[test]
    fn bootstrap_rows_stats_with_replacement_needs_jitter_for_ksg() {
        // With-replacement bootstrap without jitter produces duplicate rows that
        // make KSG fail on every resample (n_valid == 0); a tiny jitter rescues
        // validity. This pins the failure mode documented on the scheme enum.
        let (x, y) = make_linear_pair(150, 0.5, 17);
        let ksg_cfg = crate::KsgConfig::default();
        let stat = |mats: &[MatRef<'_>]| -> PidResult<Vec<f64>> {
            Ok(vec![crate::ksg_mi(mats[0], mats[1], &ksg_cfg)?])
        };
        let cfg = BootstrapConfig {
            n_boot: 16,
            block_size: 1,
            seed: 3,
            alpha: 0.05,
        };
        let mats = [x.as_ref(), y.as_ref()];
        let without = bootstrap_rows_stats(
            &mats,
            &cfg,
            RowResampleScheme::BlockBootstrapJitter { jitter_rel: 0.0 },
            stat,
        )
        .unwrap();
        assert_eq!(without.stats[0].n_valid, 0);
        assert!(without.stats[0].ci_low.is_nan());

        let with = bootstrap_rows_stats(
            &mats,
            &cfg,
            RowResampleScheme::BlockBootstrapJitter { jitter_rel: 1e-9 },
            stat,
        )
        .unwrap();
        assert_eq!(with.stats[0].n_valid, 16);
        assert!(with.stats[0].ci_low.is_finite());
        assert!(with.stats[0].ci_high >= with.stats[0].ci_low);
    }

    #[test]
    fn bootstrap_rows_stats_rejects_bad_config() {
        let (x, y) = make_linear_pair(50, 0.5, 1);
        let mats = [x.as_ref(), y.as_ref()];
        let jit = RowResampleScheme::BlockBootstrapJitter { jitter_rel: 0.0 };
        let mut cfg = BootstrapConfig {
            n_boot: 0,
            block_size: 1,
            seed: 0,
            alpha: 0.05,
        };
        assert!(bootstrap_rows_stats(&mats, &cfg, jit, pearson_stat).is_err());
        cfg.n_boot = 8;
        cfg.block_size = 0;
        assert!(bootstrap_rows_stats(&mats, &cfg, jit, pearson_stat).is_err());
        cfg.block_size = 1;
        cfg.alpha = 1.5;
        assert!(bootstrap_rows_stats(&mats, &cfg, jit, pearson_stat).is_err());
        cfg.alpha = 0.05;
        assert!(bootstrap_rows_stats(
            &mats,
            &cfg,
            RowResampleScheme::BlockBootstrapJitter { jitter_rel: -1.0 },
            pearson_stat
        )
        .is_err());
        // Subsample longer than n must be rejected.
        assert!(bootstrap_rows_stats(
            &mats,
            &cfg,
            RowResampleScheme::Subsample {
                subsample_len: 1000
            },
            pearson_stat
        )
        .is_err());
        // Subsample shorter than block_size must be rejected.
        cfg.block_size = 10;
        assert!(bootstrap_rows_stats(
            &mats,
            &cfg,
            RowResampleScheme::Subsample { subsample_len: 5 },
            pearson_stat
        )
        .is_err());
    }

    #[test]
    fn permutation_rows_pvalue_detects_signal_and_respects_null() {
        let ksg_cfg = crate::KsgConfig::default();
        let stat =
            |mats: &[MatRef<'_>]| -> PidResult<f64> { crate::ksg_mi(mats[0], mats[1], &ksg_cfg) };

        // Strong linear signal: p should be at the add-one floor 1/(M+1).
        let (x, y) = make_linear_pair(150, 0.3, 21);
        let mats = [x.as_ref(), y.as_ref()];
        let signal = permutation_rows_pvalue(&mats, 0, 99, 5, stat).unwrap();
        assert_eq!(signal.n_valid, 99);
        assert!(
            (signal.p_value - 1.0 / 100.0).abs() < 1e-12,
            "p={}",
            signal.p_value
        );

        // Independent pair: p should be large (deterministic for this seed; the
        // statistical claim is uniformity, this is a regression pin).
        let (x_a, _) = make_linear_pair(150, 0.3, 100);
        let (x_b, _) = make_linear_pair(150, 0.3, 200);
        let mats_null = [x_a.as_ref(), x_b.as_ref()];
        let null = permutation_rows_pvalue(&mats_null, 0, 99, 5, stat).unwrap();
        assert!(null.p_value > 0.1, "p={}", null.p_value);
    }

    #[test]
    fn permutation_rows_pvalue_is_deterministic_and_validates_input() {
        let (x, y) = make_linear_pair(80, 0.5, 33);
        let mats = [x.as_ref(), y.as_ref()];
        let stat = pearson_stat;
        let scalar = |m: &[MatRef<'_>]| -> PidResult<f64> { Ok(stat(m)?[0]) };
        let a = permutation_rows_pvalue(&mats, 0, 49, 9, scalar).unwrap();
        let b = permutation_rows_pvalue(&mats, 0, 49, 9, scalar).unwrap();
        assert_eq!(a, b);
        assert!(permutation_rows_pvalue(&mats, 2, 49, 9, scalar).is_err());
        assert!(permutation_rows_pvalue(&mats, 0, 0, 9, scalar).is_err());
        let empty: [MatRef<'_>; 0] = [];
        assert!(permutation_rows_pvalue(&empty, 0, 9, 9, scalar).is_err());
    }
}
