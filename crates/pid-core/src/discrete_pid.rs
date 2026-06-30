//! Discrete PID via quantization: an escape hatch for high-dimensional continuous data
//! where kNN-based MI estimation fails due to distance concentration.
//!
//! # Strategy
//!
//! 1. Quantize each continuous variable into `num_bins` equal-width bins per dimension.
//! 2. Compute discrete entropies by counting bin occupancies.
//! 3. Derive MI, co-information, and a Williams–Beer-style `I_min` redundancy
//!    (minimum specific information per target outcome) from discrete counts.
//! 4. Produce PID atoms (Red, Unq1, Unq2, Syn) via the standard Möbius identities,
//!    but with counting-based estimation.
//!
//! # Measure identity (discrete `I_min` vs continuous `I^sx_∩` — do not blur)
//!
//! The redundancy implemented here is the Williams & Beer (2010, arXiv:1004.2515)
//! `I_min` functional, **not** the discrete shared-exclusions `i^sx_∩` of
//! Makkeh et al. (2021). In the *exact* (population) limit the `I_min` redundancy is
//! non-negative and the resulting Red/Unq atoms are non-negative; however, **this
//! module computes a plug-in estimate from finite, quantized samples**, and that
//! property does **not** carry over. With finite data the empirical bin
//! frequencies are noisy, so derived atoms — the uniques `MI - Red`, and especially
//! the synergy/higher-order atoms obtained by Möbius inversion — can come out
//! negative even though the population values are not. Treat negative atoms here as
//! sampling/quantization artifacts (and as a signal that more samples or fewer bins
//! are needed), not as the genuine negative-synergy features that `I^sx_∩` admits by
//! construction. Comparing this module's output against the continuous `I^sx_∩` path
//! is a cross-measure comparison (Warning 6), valid only as a robustness check.
//!
//! This bypasses the kNN geometry problems entirely: discrete PID counts mass in
//! joint/marginal bins rather than measuring exclusion-ball volumes.
//!
//! # When to use
//!
//! - When the Experiment 0 geometry gate flags distance concentration or high intrinsic
//!   dimension in the continuous data.
//! - When `v̄ < 0` (monotonicity violation) blocks continuous PID interpretation.
//! - As a robustness check: compare discrete and continuous PID on the same data.
//!
//! # Limitations
//!
//! - Quantization destroys fine-grained information; results depend on `num_bins`.
//! - High-dimensional quantization is combinatorial (curse of dimensionality in bin counts).
//! - This module is designed for **low effective dimension** targets (after PLS/PCA reduction)
//!   or for scalar/low-d action spaces.

use crate::error::{PidError, PidResult};
use crate::matrix::MatRef;
use std::collections::BTreeMap;

/// Result of a discrete 2-source PID decomposition.
#[derive(Debug, Clone)]
pub struct DiscretePid2Result {
    pub redundancy: f64,
    pub unique_s1: f64,
    pub unique_s2: f64,
    pub synergy: f64,
    pub mi_s1_t: f64,
    pub mi_s2_t: f64,
    pub mi_s1s2_t: f64,
    pub num_bins: usize,
}

/// Quantize a continuous matrix into equal-width bins per dimension.
///
/// Each column is independently binned into `num_bins` equal-width bins spanning
/// the column's [min, max] range. Values exactly at `max` are placed in the last bin.
///
/// Returns a matrix of bin indices (nrows × ncols), stored row-major.
pub fn quantize_equal_width(x: MatRef<'_>, num_bins: usize) -> PidResult<Vec<Vec<usize>>> {
    if num_bins < 2 {
        return Err(PidError::InvalidConfig {
            context: "quantize_equal_width",
            message: "num_bins must be >= 2",
        });
    }
    let n = x.nrows();
    let d = x.ncols();

    // Compute column min/max.
    let mut col_min = vec![f64::INFINITY; d];
    let mut col_max = vec![f64::NEG_INFINITY; d];
    for i in 0..n {
        let row = x.row(i);
        for j in 0..d {
            if row[j] < col_min[j] {
                col_min[j] = row[j];
            }
            if row[j] > col_max[j] {
                col_max[j] = row[j];
            }
        }
    }

    let mut out = vec![vec![0usize; d]; n];
    for (i, out_row) in out.iter_mut().enumerate() {
        let row = x.row(i);
        for j in 0..d {
            let range = col_max[j] - col_min[j];
            // Treat a column as constant only relative to its own magnitude, so a genuinely
            // varying but tiny-scale column (e.g. values ~1e-9 apart) is not collapsed into
            // bin 0 by an absolute threshold.
            let scale = col_max[j].abs().max(col_min[j].abs()).max(1.0);
            let bin = if range <= 1e-12 * scale {
                0 // Constant column → all in bin 0.
            } else {
                let frac = (row[j] - col_min[j]) / range;
                let b = (frac * num_bins as f64) as usize;
                b.min(num_bins - 1) // Clamp max value into last bin.
            };
            out_row[j] = bin;
        }
    }
    Ok(out)
}

/// Compute discrete Shannon entropy H(X) from bin assignments.
///
/// `bins` is n×d_x; entropy is computed over the joint distribution of all columns.
/// Units: nats (natural logarithm).
///
/// Occupancy is counted per **distinct bin vector** (the row slice is the histogram
/// key), so there is no packed-integer key and therefore no overflow/collision in
/// high dimension — distinct joint states never alias. `num_bins` is accepted for
/// interface symmetry with the quantize-based callers; the count is independent of
/// it.
pub fn discrete_entropy(bins: &[Vec<usize>], num_bins: usize) -> f64 {
    let _ = num_bins;
    let n = bins.len();
    if n == 0 {
        return 0.0;
    }
    let counts = count_dist(bins);
    let inv_n = 1.0 / n as f64;
    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 * inv_n;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Compute discrete mutual information I(X;Y) from quantized data.
///
/// `x_bins` is n×d_x, `y_bins` is n×d_y.
/// I(X;Y) = H(X) + H(Y) - H(X,Y).
pub fn discrete_mi(
    x_bins: &[Vec<usize>],
    y_bins: &[Vec<usize>],
    num_bins: usize,
) -> PidResult<f64> {
    if x_bins.len() != y_bins.len() {
        return Err(PidError::RowCountMismatch {
            context: "discrete_mi",
            left_rows: x_bins.len(),
            right_rows: y_bins.len(),
        });
    }
    let n = x_bins.len();

    let h_x = discrete_entropy(x_bins, num_bins);
    let h_y = discrete_entropy(y_bins, num_bins);

    // Joint entropy H(X,Y).
    let mut joint = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = x_bins[i].clone();
        row.extend_from_slice(&y_bins[i]);
        joint.push(row);
    }
    let h_xy = discrete_entropy(&joint, num_bins);

    Ok(h_x + h_y - h_xy)
}

/// Compute discrete 2-source PID atoms via quantization + a Williams–Beer-style
/// `I_min` redundancy (not discrete `i^sx_∩`; see the module docs).
///
/// Sources S1, S2 and target T are each quantized into `num_bins` equal-width bins.
/// Redundancy uses the minimum-specific-information (`I_min`) formula:
///
/// `Red(S1,S2;T) = Σ_t p(t) min(i_spec(S1;t), i_spec(S2;t))`
///
/// where `i_spec(S;t) = Σ_s p(s|t) log(p(t|s)/p(t))` is the specific information.
pub fn discrete_pid2(
    s1: MatRef<'_>,
    s2: MatRef<'_>,
    target: MatRef<'_>,
    num_bins: usize,
) -> PidResult<DiscretePid2Result> {
    if num_bins < 2 {
        return Err(PidError::InvalidConfig {
            context: "discrete_pid2",
            message: "num_bins must be >= 2",
        });
    }
    let n = s1.nrows();
    if s2.nrows() != n || target.nrows() != n {
        return Err(PidError::RowCountMismatch {
            context: "discrete_pid2",
            left_rows: n,
            right_rows: if s2.nrows() != n {
                s2.nrows()
            } else {
                target.nrows()
            },
        });
    }

    // 1. Quantize all three variables.
    let s1_bins = quantize_equal_width(s1, num_bins)?;
    let s2_bins = quantize_equal_width(s2, num_bins)?;
    let t_bins = quantize_equal_width(target, num_bins)?;

    // 2. Compute MI terms.
    let mi_s1_t = discrete_mi(&s1_bins, &t_bins, num_bins)?;
    let mi_s2_t = discrete_mi(&s2_bins, &t_bins, num_bins)?;

    // For joint MI: concatenate S1 and S2 bins.
    let mut s1s2_bins = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = s1_bins[i].clone();
        row.extend_from_slice(&s2_bins[i]);
        s1s2_bins.push(row);
    }
    let mi_s1s2_t = discrete_mi(&s1s2_bins, &t_bins, num_bins)?;

    // 3. Compute the I_min redundancy via per-target-outcome specific information.
    let redundancy = discrete_imin_redundancy(&s1_bins, &s2_bins, &t_bins);

    // 4. Derive PID atoms.
    let unique_s1 = mi_s1_t - redundancy;
    let unique_s2 = mi_s2_t - redundancy;
    let synergy = mi_s1s2_t - mi_s1_t - mi_s2_t + redundancy;

    Ok(DiscretePid2Result {
        redundancy,
        unique_s1,
        unique_s2,
        synergy,
        mi_s1_t,
        mi_s2_t,
        mi_s1s2_t,
        num_bins,
    })
}

/// Discrete Williams–Beer-style `I_min` redundancy.
///
/// `Red(S1,S2;T) = Σ_t p(t) min(i_spec(S1;t), i_spec(S2;t))`
///
/// where `i_spec(S;t) = Σ_s p(s|t) log(p(t|s)/p(t))`.
fn discrete_imin_redundancy(
    s1_bins: &[Vec<usize>],
    s2_bins: &[Vec<usize>],
    t_bins: &[Vec<usize>],
) -> f64 {
    let n = s1_bins.len();
    if n == 0 {
        return 0.0;
    }
    let inv_n = 1.0 / n as f64;

    // Build marginal distributions and conditional distributions.
    // For each source S, compute p(s) and p(s|t) and p(t|s).
    let t_counts = count_dist(t_bins);
    let s1_counts = count_dist(s1_bins);
    let s2_counts = count_dist(s2_bins);

    // Joint counts: (s, t) for each source.
    let s1t_counts = count_joint_dist(s1_bins, t_bins);
    let s2t_counts = count_joint_dist(s2_bins, t_bins);

    // Compute specific information for each (source, t) pair:
    // i_spec(S;t) = Σ_s p(s|t) log(p(t|s) / p(t))
    //             = Σ_s [p(s,t)/p(t)] log[p(s,t) * n / (p(s) * p(t) * n)]
    //             = Σ_s [count(s,t)/count(t)] log[count(s,t) * n / (count(s) * count(t))]
    let i_spec_s1 = specific_information(&s1t_counts, &s1_counts, &t_counts, n);
    let i_spec_s2 = specific_information(&s2t_counts, &s2_counts, &t_counts, n);

    // Red = Σ_t p(t) min(i_spec(S1;t), i_spec(S2;t))
    let mut red = 0.0;
    for (t_key, &ct) in &t_counts {
        let p_t = ct as f64 * inv_n;
        let is1 = i_spec_s1.get(t_key).copied().unwrap_or(0.0);
        let is2 = i_spec_s2.get(t_key).copied().unwrap_or(0.0);
        red += p_t * is1.min(is2);
    }

    red
}

/// Count the frequency of each distinct bin vector.
///
/// The histogram key is the bin vector itself, so distinct joint states can never
/// collide (unlike a packed base-`num_bins` integer, which overflows `usize` once
/// `num_bins`^d exceeds 2^64).
fn count_dist(bins: &[Vec<usize>]) -> BTreeMap<Vec<usize>, usize> {
    let mut counts = BTreeMap::new();
    for row in bins {
        *counts.entry(row.clone()).or_insert(0) += 1;
    }
    counts
}

/// Count the joint frequency of (x_bins, y_bins) pairs, keyed on the bin vectors.
fn count_joint_dist(
    x_bins: &[Vec<usize>],
    y_bins: &[Vec<usize>],
) -> BTreeMap<(Vec<usize>, Vec<usize>), usize> {
    let mut counts = BTreeMap::new();
    for (xr, yr) in x_bins.iter().zip(y_bins) {
        *counts.entry((xr.clone(), yr.clone())).or_insert(0) += 1;
    }
    counts
}

/// Compute specific information `i(S; t)` for each target bin `t`.
///
/// `i(S; t) = Σ_s p(s|t) log(p(s,t) * n / (p(s) * p(t) * n))`
///           = Σ_s [count(s,t)/count(t)] * log[count(s,t) * n / (count(s) * count(t))]
fn specific_information(
    st_counts: &BTreeMap<(Vec<usize>, Vec<usize>), usize>,
    s_counts: &BTreeMap<Vec<usize>, usize>,
    t_counts: &BTreeMap<Vec<usize>, usize>,
    n: usize,
) -> BTreeMap<Vec<usize>, f64> {
    let mut result = BTreeMap::new();

    // Group joint counts by t.
    let mut by_t: BTreeMap<&[usize], Vec<(&[usize], usize)>> = BTreeMap::new();
    for ((sk, tk), &cst) in st_counts {
        by_t.entry(tk).or_default().push((sk, cst));
    }

    for (&tk, entries) in &by_t {
        let ct = t_counts.get(tk).copied().unwrap_or(0);
        if ct == 0 {
            continue;
        }
        let mut is = 0.0;
        for &(sk, cst) in entries {
            let cs = s_counts.get(sk).copied().unwrap_or(0);
            if cs == 0 || cst == 0 {
                continue;
            }
            // p(s|t) = cst / ct
            // log(p(s,t) / (p(s) * p(t))) = log(cst * n / (cs * ct))
            let log_ratio = ((cst as f64) * (n as f64) / ((cs as f64) * (ct as f64))).ln();
            is += (cst as f64 / ct as f64) * log_ratio;
        }
        result.insert(tk.to_vec(), is);
    }

    result
}

/// Result of a discrete 3-source PID decomposition (18 atoms on the redundancy lattice).
#[derive(Debug, Clone)]
pub struct DiscretePid3Result {
    /// PID atoms in canonical antichain order (same 18 antichains as continuous pid3_isx).
    pub atoms: Vec<DiscretePid3Atom>,
    /// Per-antichain redundancy values.
    pub redundancies: Vec<f64>,
    /// MI terms: I(S0;T), I(S1;T), I(S2;T).
    pub mi_s0_t: f64,
    pub mi_s1_t: f64,
    pub mi_s2_t: f64,
    /// Pairwise joint MIs: I(S0,S1;T), I(S0,S2;T), I(S1,S2;T).
    pub mi_s0s1_t: f64,
    pub mi_s0s2_t: f64,
    pub mi_s1s2_t: f64,
    /// Triple joint MI: I(S0,S1,S2;T).
    pub mi_s0s1s2_t: f64,
    pub num_bins: usize,
}

/// A single PID atom for discrete 3-source decomposition.
#[derive(Debug, Clone)]
pub struct DiscretePid3Atom {
    /// Antichain identifying this atom (as a bitmask array, same encoding as pid3_isx).
    pub antichain_sets: Vec<u8>,
    pub value: f64,
}

/// Compute discrete 3-source PID atoms via quantization + a Williams–Beer-style
/// `I_min` redundancy over the full 18-antichain lattice (not discrete `i^sx_∩`;
/// see the module docs).
///
/// Sources S0, S1, S2 and target T are each quantized into `num_bins` equal-width bins.
/// All 18 antichains on the redundancy lattice are evaluated, and Möbius inversion
/// yields the PID atoms.
///
/// Units: nats (natural logarithm).
pub fn discrete_pid3(
    s0: MatRef<'_>,
    s1: MatRef<'_>,
    s2: MatRef<'_>,
    target: MatRef<'_>,
    num_bins: usize,
) -> PidResult<DiscretePid3Result> {
    if num_bins < 2 {
        return Err(PidError::InvalidConfig {
            context: "discrete_pid3",
            message: "num_bins must be >= 2",
        });
    }
    let n = s0.nrows();
    if s1.nrows() != n || s2.nrows() != n || target.nrows() != n {
        // Report the first operand that actually mismatches, not min() (which can equal n
        // and hide the real culprit).
        let right_rows = if s1.nrows() != n {
            s1.nrows()
        } else if s2.nrows() != n {
            s2.nrows()
        } else {
            target.nrows()
        };
        return Err(PidError::RowCountMismatch {
            context: "discrete_pid3",
            left_rows: n,
            right_rows,
        });
    }

    // Quantize all variables.
    let s0_bins = quantize_equal_width(s0, num_bins)?;
    let s1_bins = quantize_equal_width(s1, num_bins)?;
    let s2_bins = quantize_equal_width(s2, num_bins)?;
    let t_bins = quantize_equal_width(target, num_bins)?;
    let sources: [&[Vec<usize>]; 3] = [&s0_bins, &s1_bins, &s2_bins];

    // Compute MI terms.
    let mi_s0_t = discrete_mi(&s0_bins, &t_bins, num_bins)?;
    let mi_s1_t = discrete_mi(&s1_bins, &t_bins, num_bins)?;
    let mi_s2_t = discrete_mi(&s2_bins, &t_bins, num_bins)?;
    let mi_s0s1_t = discrete_mi(&join_bins_pair(&s0_bins, &s1_bins), &t_bins, num_bins)?;
    let mi_s0s2_t = discrete_mi(&join_bins_pair(&s0_bins, &s2_bins), &t_bins, num_bins)?;
    let mi_s1s2_t = discrete_mi(&join_bins_pair(&s1_bins, &s2_bins), &t_bins, num_bins)?;
    let mi_s0s1s2_t = discrete_mi(
        &join_bins_triple(&s0_bins, &s1_bins, &s2_bins),
        &t_bins,
        num_bins,
    )?;

    // Compute 18 antichain redundancies.
    let antichains = discrete_antichains_3();
    let mut redundancies = Vec::with_capacity(18);
    for &ac in &antichains {
        let val = discrete_imin_redundancy_3way(&sources, &t_bins, ac);
        redundancies.push(val);
    }

    // Möbius inversion to get atoms.
    let atoms = discrete_mobius_inversion_3(&antichains, &redundancies);

    Ok(DiscretePid3Result {
        atoms,
        redundancies,
        mi_s0_t,
        mi_s1_t,
        mi_s2_t,
        mi_s0s1_t,
        mi_s0s2_t,
        mi_s1s2_t,
        mi_s0s1s2_t,
        num_bins,
    })
}

/// Build joint bins for a pair of sources (for subset mask with 2 bits set).
fn join_bins_pair(a: &[Vec<usize>], b: &[Vec<usize>]) -> Vec<Vec<usize>> {
    a.iter()
        .zip(b)
        .map(|(ar, br)| {
            let mut row = ar.clone();
            row.extend_from_slice(br);
            row
        })
        .collect()
}

/// Build joint bins for three sources.
fn join_bins_triple(a: &[Vec<usize>], b: &[Vec<usize>], c: &[Vec<usize>]) -> Vec<Vec<usize>> {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((ar, br), cr)| {
            let mut row = ar.clone();
            row.extend_from_slice(br);
            row.extend_from_slice(cr);
            row
        })
        .collect()
}

/// 18 canonical antichains on {0,1,2}, encoded as bitmask arrays.
pub(crate) fn discrete_antichains_3() -> [[u8; 3]; 18] {
    [
        [0b001, 0, 0],
        [0b010, 0, 0],
        [0b100, 0, 0],
        [0b011, 0, 0],
        [0b101, 0, 0],
        [0b110, 0, 0],
        [0b111, 0, 0],
        [0b001, 0b010, 0],
        [0b001, 0b100, 0],
        [0b001, 0b110, 0],
        [0b010, 0b100, 0],
        [0b010, 0b101, 0],
        [0b011, 0b100, 0],
        [0b011, 0b101, 0],
        [0b011, 0b110, 0],
        [0b101, 0b110, 0],
        [0b001, 0b010, 0b100],
        [0b011, 0b101, 0b110],
    ]
}

/// Compute specific information i(S;t) for an arbitrary source subset mask.
///
/// The source subset is the joint distribution of the sources indicated by `mask`.
fn i_spec_for_mask(
    sources: &[&[Vec<usize>]; 3],
    t_bins: &[Vec<usize>],
    mask: u8,
    n: usize,
) -> BTreeMap<Vec<usize>, f64> {
    let joint = match mask {
        0b001 => sources[0].to_vec(),
        0b010 => sources[1].to_vec(),
        0b100 => sources[2].to_vec(),
        m => {
            let mut j = vec![Vec::new(); n];
            for i in 0..n {
                if (m & 0b001) != 0 {
                    j[i].extend_from_slice(&sources[0][i]);
                }
                if (m & 0b010) != 0 {
                    j[i].extend_from_slice(&sources[1][i]);
                }
                if (m & 0b100) != 0 {
                    j[i].extend_from_slice(&sources[2][i]);
                }
            }
            j
        }
    };
    let s_counts = count_dist(&joint);
    let st_counts = count_joint_dist(&joint, t_bins);
    let t_counts = count_dist(t_bins);
    specific_information(&st_counts, &s_counts, &t_counts, n)
}

/// 3-source discrete Williams–Beer-style `I_min` redundancy for a single antichain.
fn discrete_imin_redundancy_3way(
    sources: &[&[Vec<usize>]; 3],
    t_bins: &[Vec<usize>],
    antichain: [u8; 3],
) -> f64 {
    let n = t_bins.len();
    if n == 0 {
        return 0.0;
    }
    let inv_n = 1.0 / n as f64;

    // Determine how many sets are in this antichain.
    let n_sets = if antichain[2] != 0 {
        3
    } else if antichain[1] != 0 {
        2
    } else {
        1
    };

    // Compute i_spec for each set in the antichain.
    let mut i_specs: Vec<BTreeMap<Vec<usize>, f64>> = Vec::with_capacity(n_sets);
    for &mask in antichain.iter().take(n_sets) {
        i_specs.push(i_spec_for_mask(sources, t_bins, mask, n));
    }

    // Red = Σ_t p(t) min_s i_spec(S_s; t)
    let t_counts = count_dist(t_bins);
    let mut red = 0.0;
    for (t_key, &ct) in &t_counts {
        let p_t = ct as f64 * inv_n;
        let mut min_is = f64::INFINITY;
        for is in &i_specs {
            min_is = min_is.min(is.get(t_key).copied().unwrap_or(0.0));
        }
        if min_is.is_finite() {
            red += p_t * min_is;
        }
    }
    red
}

/// Möbius inversion on the 3-source redundancy lattice to obtain PID atoms.
///
/// Measure-agnostic: it inverts any per-antichain *cumulative* functional that obeys
/// `cumulative(α) = Σ_{β ⪯ α} atom(β)`. Reused by both the `I_min` path here and the
/// shared-exclusions `i^sx_∩` path in the `sxpid` module.
pub(crate) fn discrete_mobius_inversion_3(
    antichains: &[[u8; 3]],
    redundancies: &[f64],
) -> Vec<DiscretePid3Atom> {
    let n = antichains.len();
    let mut atoms = vec![0.0f64; n];

    // Topological order: start from minimal antichains (fewest sets, smallest masks).
    let topo = discrete_topo_order_3(antichains);

    for (pos, &idx) in topo.iter().enumerate() {
        let mut val = redundancies[idx];
        for &j in &topo[..pos] {
            if discrete_leq_3(antichains[j], antichains[idx]) {
                val -= atoms[j];
            }
        }
        atoms[idx] = val;
    }

    antichains
        .iter()
        .enumerate()
        .map(|(idx, ac)| {
            let sets: Vec<u8> = ac.iter().copied().filter(|&m| m != 0).collect();
            DiscretePid3Atom {
                antichain_sets: sets,
                value: atoms[idx],
            }
        })
        .collect()
}

/// Check if antichain a ⪯ b in the redundancy lattice ordering.
fn discrete_leq_3(a: [u8; 3], b: [u8; 3]) -> bool {
    let n_b = if b[2] != 0 {
        3
    } else if b[1] != 0 {
        2
    } else {
        1
    };
    let n_a = if a[2] != 0 {
        3
    } else if a[1] != 0 {
        2
    } else {
        1
    };
    for &b_j in b.iter().take(n_b) {
        let mut found = false;
        for &a_i in a.iter().take(n_a) {
            if (a_i & b_j) == a_i {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

/// Topological sort for the 18-antichain lattice.
fn discrete_topo_order_3(antichains: &[[u8; 3]]) -> Vec<usize> {
    let n = antichains.len();
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n);
    while !remaining.is_empty() {
        let mut mins = Vec::new();
        'outer: for &i in &remaining {
            for &j in &remaining {
                if i == j {
                    continue;
                }
                if discrete_leq_3(antichains[j], antichains[i]) {
                    continue 'outer;
                }
            }
            mins.push(i);
        }
        mins.sort();
        let chosen = mins[0];
        out.push(chosen);
        remaining.retain(|&x| x != chosen);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatRef;

    #[test]
    fn discrete_entropy_uniform() {
        // 4 equally likely bins → H = ln(4) ≈ 1.386
        let bins: Vec<Vec<usize>> = (0..400).map(|i| vec![i % 4]).collect();
        let h = discrete_entropy(&bins, 4);
        assert!(
            (h - 4.0f64.ln()).abs() < 0.05,
            "H(uniform 4 bins) should be ≈ ln(4); got {h}"
        );
    }

    #[test]
    fn discrete_mi_independent() {
        // Independent X and Y → I(X;Y) ≈ 0
        let n = 1000;
        let mut rng = crate::preprocess::SplitMix64::new(42);
        let mut x_bins = Vec::with_capacity(n);
        let mut y_bins = Vec::with_capacity(n);
        for _ in 0..n {
            x_bins.push(vec![(rng.next_u64() as usize) % 4]);
            y_bins.push(vec![(rng.next_u64() as usize) % 4]);
        }
        let mi = discrete_mi(&x_bins, &y_bins, 4).unwrap();
        assert!(
            mi.abs() < 0.05,
            "MI of independent vars should be ≈ 0; got {mi}"
        );
    }

    #[test]
    fn discrete_mi_copy() {
        // Y = X → I(X;Y) = H(X)
        let n = 500;
        let bins: Vec<Vec<usize>> = (0..n).map(|i| vec![i % 8]).collect();
        let mi = discrete_mi(&bins, &bins, 8).unwrap();
        let h = discrete_entropy(&bins, 8);
        assert!(
            (mi - h).abs() < 0.01,
            "MI(X;X) should equal H(X); MI={mi}, H={h}"
        );
    }

    #[test]
    fn discrete_pid2_redundant_copy() {
        // S1 = S2 = signal → Red ≈ MI, Unq ≈ 0, Syn ≈ 0
        let n = 500;
        let d = 1;
        let mut rng = crate::preprocess::SplitMix64::new(99);
        let mut s1_data = Vec::with_capacity(n * d);
        let mut s2_data = Vec::with_capacity(n * d);
        let mut t_data = Vec::with_capacity(n * d);
        for _ in 0..n {
            let sig = rng.normal();
            s1_data.push(sig);
            s2_data.push(sig + 0.01 * rng.normal()); // Near-copy
            t_data.push(sig + 0.1 * rng.normal());
        }
        let s1 = MatRef::new(&s1_data, n, d).unwrap();
        let s2 = MatRef::new(&s2_data, n, d).unwrap();
        let t = MatRef::new(&t_data, n, d).unwrap();

        let result = discrete_pid2(s1, s2, t, 10).unwrap();

        // Redundancy should dominate; unique should be small.
        assert!(
            result.redundancy > 0.5 * result.mi_s1_t,
            "Redundancy should be > 50% of MI for near-copies; Red={}, MI={}",
            result.redundancy,
            result.mi_s1_t
        );
        assert!(
            result.unique_s1.abs() < 0.3 * result.mi_s1_t,
            "Unique S1 should be small for near-copies; Unq1={}",
            result.unique_s1
        );
    }

    /// Build an exactly-enumerated 2-input binary gate dataset: every (s1,s2) ∈ {0,1}²
    /// combination repeated `reps` times, with `t = gate(s1,s2)`. Because each of the four
    /// joint states appears equally often, the empirical distribution is *exact* (no sampling
    /// error), so the binned `I_min` PID equals its analytic closed form to machine precision.
    fn binary_gate_dataset(
        reps: usize,
        gate: impl Fn(u8, u8) -> u8,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        let mut t = Vec::new();
        for _ in 0..reps {
            for a in 0u8..2 {
                for b in 0u8..2 {
                    s1.push(a as f64);
                    s2.push(b as f64);
                    t.push(gate(a, b) as f64);
                }
            }
        }
        let n = 4 * reps;
        (s1, s2, t, n)
    }

    #[test]
    fn discrete_pid2_xor_is_pure_synergy() {
        // Williams & Beer (2010), canonical XOR gate, uniform independent inputs:
        //   I(S1;T) = I(S2;T) = 0,  I(S1,S2;T) = ln 2  (1 bit)
        //   Red = 0, Unq1 = Unq2 = 0, Syn = ln 2.
        let (s1, s2, t, n) = binary_gate_dataset(64, |a, b| a ^ b);
        let s1 = MatRef::new(&s1, n, 1).unwrap();
        let s2 = MatRef::new(&s2, n, 1).unwrap();
        let t = MatRef::new(&t, n, 1).unwrap();

        let r = discrete_pid2(s1, s2, t, 2).unwrap();
        let ln2 = 2.0f64.ln();
        let tol = 1e-9;
        assert!(
            r.mi_s1_t.abs() < tol,
            "I(S1;T) should be 0; got {}",
            r.mi_s1_t
        );
        assert!(
            r.mi_s2_t.abs() < tol,
            "I(S2;T) should be 0; got {}",
            r.mi_s2_t
        );
        assert!(
            (r.mi_s1s2_t - ln2).abs() < tol,
            "I(S1,S2;T) should be ln 2; got {}",
            r.mi_s1s2_t
        );
        assert!(
            r.redundancy.abs() < tol,
            "Red should be 0; got {}",
            r.redundancy
        );
        assert!(
            r.unique_s1.abs() < tol,
            "Unq1 should be 0; got {}",
            r.unique_s1
        );
        assert!(
            r.unique_s2.abs() < tol,
            "Unq2 should be 0; got {}",
            r.unique_s2
        );
        assert!(
            (r.synergy - ln2).abs() < tol,
            "Syn should be ln 2; got {}",
            r.synergy
        );
        // Identity must hold exactly.
        assert!((r.redundancy + r.unique_s1 + r.unique_s2 + r.synergy - r.mi_s1s2_t).abs() < tol);
    }

    #[test]
    fn discrete_pid2_and_matches_williams_beer() {
        // Williams & Beer (2010), canonical AND gate, uniform independent inputs.
        // p(T=1) = 1/4. Analytic atoms (nats):
        //   H(T) = 0.25·ln4 + 0.75·ln(4/3) = 0.5623351446...
        //   I(S1;T) = I(S2;T) = H(T) − 0.5·ln2 = 0.2157615680...
        //   Red = I(S1;T) (symmetric sources), Unq1 = Unq2 = 0,
        //   Syn = I(S1,S2;T) − I(S1;T) = H(T) − I(S1;T) = 0.3465735903... = 0.5·ln4·... (= ln2/2·... )
        let (s1, s2, t, n) = binary_gate_dataset(64, |a, b| a & b);
        let s1 = MatRef::new(&s1, n, 1).unwrap();
        let s2 = MatRef::new(&s2, n, 1).unwrap();
        let t = MatRef::new(&t, n, 1).unwrap();

        let r = discrete_pid2(s1, s2, t, 2).unwrap();

        let h_t = 0.25 * 4.0f64.ln() + 0.75 * (4.0f64 / 3.0).ln();
        let i_single = h_t - 0.5 * 2.0f64.ln();
        let syn = h_t - i_single;
        let tol = 1e-9;

        assert!(
            (r.mi_s1_t - i_single).abs() < tol,
            "I(S1;T)={} want {i_single}",
            r.mi_s1_t
        );
        assert!(
            (r.mi_s2_t - i_single).abs() < tol,
            "I(S2;T)={} want {i_single}",
            r.mi_s2_t
        );
        assert!(
            (r.mi_s1s2_t - h_t).abs() < tol,
            "I(S1,S2;T)={} want {h_t}",
            r.mi_s1s2_t
        );
        assert!(
            (r.redundancy - i_single).abs() < tol,
            "Red={} want {i_single}",
            r.redundancy
        );
        assert!(
            r.unique_s1.abs() < tol,
            "Unq1 should be 0; got {}",
            r.unique_s1
        );
        assert!(
            r.unique_s2.abs() < tol,
            "Unq2 should be 0; got {}",
            r.unique_s2
        );
        assert!(
            (r.synergy - syn).abs() < tol,
            "Syn={} want {syn}",
            r.synergy
        );
        assert!((r.redundancy + r.unique_s1 + r.unique_s2 + r.synergy - r.mi_s1s2_t).abs() < tol);
    }

    #[test]
    fn quantize_rejects_bad_bins() {
        let data = vec![0.0f64; 10];
        let m = MatRef::new(&data, 5, 2).unwrap();
        assert!(quantize_equal_width(m, 0).is_err());
        assert!(quantize_equal_width(m, 1).is_err());
    }

    #[test]
    fn discrete_pid3_produces_18_atoms() {
        let n = 80;
        let mut rng = crate::preprocess::SplitMix64::new(42);
        let mut s0_data = Vec::with_capacity(n * 2);
        let mut s1_data = Vec::with_capacity(n * 2);
        let mut s2_data = Vec::with_capacity(n);
        let mut t_data = Vec::with_capacity(n);
        for _ in 0..n {
            let signal = rng.normal();
            // S0 carries signal in dim 0.
            s0_data.push(signal + 0.1 * rng.normal());
            s0_data.push(rng.normal());
            // S1 carries signal in dim 0 (redundant with S0).
            s1_data.push(signal + 0.1 * rng.normal());
            s1_data.push(rng.normal());
            // S2 is pure noise.
            s2_data.push(rng.normal());
            // T = signal + small noise.
            t_data.push(signal + 0.05 * rng.normal());
        }
        let s0 = MatRef::new(&s0_data, n, 2).unwrap();
        let s1 = MatRef::new(&s1_data, n, 2).unwrap();
        let s2 = MatRef::new(&s2_data, n, 1).unwrap();
        let t = MatRef::new(&t_data, n, 1).unwrap();

        let result = discrete_pid3(s0, s1, s2, t, 8).unwrap();
        assert_eq!(result.atoms.len(), 18, "should produce 18 atoms");
        assert_eq!(result.redundancies.len(), 18);
        assert_eq!(result.num_bins, 8);
    }

    #[test]
    fn discrete_pid3_redundant_sources_dominant() {
        // S0 ≈ S1 (near-copy), S2 is noise → redundancy should dominate.
        let n = 200;
        let mut rng = crate::preprocess::SplitMix64::new(99);
        let mut s0 = Vec::with_capacity(n);
        let mut s1 = Vec::with_capacity(n);
        let mut s2 = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            let sig = rng.normal();
            s0.push(sig);
            s1.push(sig + 0.01 * rng.normal());
            s2.push(rng.normal());
            t.push(sig + 0.1 * rng.normal());
        }
        let s0_m = MatRef::new(&s0, n, 1).unwrap();
        let s1_m = MatRef::new(&s1, n, 1).unwrap();
        let s2_m = MatRef::new(&s2, n, 1).unwrap();
        let t_m = MatRef::new(&t, n, 1).unwrap();

        let result = discrete_pid3(s0_m, s1_m, s2_m, t_m, 10).unwrap();

        // Lattice landmarks (see `discrete_antichains_3()`), redundancies in antichain order:
        //   index 6  = {{0,1,2}}        — the single full collection = lattice TOP, whose
        //              I_min equals the joint MI I(S0,S1,S2;T) (NOT a redundancy);
        //   index 7  = {{0},{1}}        — pairwise redundancy of the two near-copies S0,S1;
        //   index 16 = {{0},{1},{2}}    — global redundancy shared by *all three* sources,
        //              hence diluted by the noise source S2.
        let joint_top = result.redundancies[6]; // {0b111} — joint MI, not redundancy
        let red_s0_s1 = result.redundancies[7]; // {{0},{1}} — pairwise redundancy
        let red_all = result.redundancies[16]; // {{0},{1},{2}} — global redundancy

        // The TOP node {{0,1,2}} is a *single* collection, so by the self-redundancy axiom its
        // I_min equals the joint MI I(S0,S1,S2;T) exactly — a strong invariant, not just a bound.
        assert!(
            (joint_top - result.mi_s0s1s2_t).abs() < 1e-9,
            "TOP node {{0,1,2}} must equal the joint MI exactly; top={joint_top}, joint MI={}",
            result.mi_s0s1s2_t
        );

        // Because S0 and S1 are near-copies, the information they *share* about T is sizable —
        // close to I(S0;T). This is the genuine redundancy-dominance claim.
        assert!(
            red_s0_s1 > 0.3 * result.mi_s0_t,
            "Pairwise redundancy of near-copies should be > 30% of I(S0;T); red_s0_s1={red_s0_s1}, MI={}",
            result.mi_s0_t
        );

        // Adding the pure-noise source S2 can only shrink the shared information: the global
        // (all-three) redundancy must not exceed the pairwise redundancy of S0,S1.
        assert!(
            red_all <= red_s0_s1 + 1e-9,
            "Global redundancy (incl. noise S2) must not exceed pairwise S0,S1 redundancy; \
             red_all={red_all}, red_s0_s1={red_s0_s1}"
        );
    }

    #[test]
    fn discrete_pid3_rejects_mismatched_rows() {
        let s0_data = vec![0.0; 10];
        let s1_data = vec![0.0; 5];
        let s2_data = vec![0.0; 10];
        let t_data = vec![0.0; 10];
        let s0 = MatRef::new(&s0_data, 10, 1).unwrap();
        let s1 = MatRef::new(&s1_data, 5, 1).unwrap();
        let s2 = MatRef::new(&s2_data, 10, 1).unwrap();
        let t = MatRef::new(&t_data, 10, 1).unwrap();
        assert!(discrete_pid3(s0, s1, s2, t, 5).is_err());
    }
}
