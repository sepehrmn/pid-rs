//! Discrete **shared-exclusions** PID — the genuine `i^sx_∩` of Makkeh, Gutknecht & Wibral
//! (2021, Phys. Rev. E 103, 032149; arXiv:2002.03356), with the part-whole / formal-logic
//! foundation of Gutknecht, Wibral & Makkeh (2021, arXiv:2008.09535).
//!
//! # Why this exists (and how it differs from the `discrete_pid` module)
//!
//! The `discrete_pid` module computes the Williams & Beer (2010) `I_min` redundancy. `I_min` is
//! precisely the measure SxPID was introduced to replace: on the two-bit COPY of *independent*
//! sources it attributes the **maximal** 1 bit of redundancy (the Harder et al. (2013) identity
//! axiom says the answer should be 0), and it is not differentiable. SxPID instead defines
//! redundancy through **shared exclusions**: the information that source realizations *jointly
//! exclude* about the target, combined by logical **disjunction** over a redundancy lattice.
//! This is the discrete sibling of the continuous `I^sx_∩` estimator (the `isx` / `pid2` modules)
//! — so the library now decomposes information with **one** measure across the discrete and
//! continuous regimes.
//!
//! # The measure (exact)
//!
//! For a realization `(s_1,…,s_n,t)`, a *collection* `a ⊆ {1..n}` denotes the event
//! `𝔞 = ⋂_{i∈a}{S_i = s_i}`; write `𝔱 = {T = t}`. A lattice node is an **antichain**
//! `α = {a_1,…,a_k}` (no collection a subset of another). Define
//!
//! ```text
//! i⁺(t:α) = −log P(⋃_j 𝔞_j)                      (informative; sources only)
//! i⁻(t:α) =  log[ P(t) / P(𝔱 ∩ ⋃_j 𝔞_j) ]        (misinformative)
//! i^sx_∩(t:α) = i⁺ − i⁻ = log[ P(𝔱 ∩ ⋃_j 𝔞_j) / (P(t)·P(⋃_j 𝔞_j)) ]
//! ```
//!
//! `P(⋃_j 𝔞_j)` is obtained by **inclusion–exclusion** over the collections (an intersection of
//! collection-events fixes the *union* of their source indices). The **pointwise atoms**
//! `π^sx(t:α)` are the Möbius inverse on the redundancy lattice
//! (`i^sx_∩(t:α) = Σ_{β ⪯ α} π^sx(t:β)`); **averaged atoms** are `Π(α) = Σ_rlz p(rlz) π(rlz,α)`
//! (inversion and averaging commute). A single-collection node gives `i^sx_∩(t:{a}) = i(t:s_a)`
//! (pointwise MI), i.e. the **self-redundancy** axiom.
//!
//! # Conventions (match the rest of the crate)
//!
//! - **Units: nats** (natural log). The reference fixtures (Abzinger/SxPID, IDTxl) are in bits;
//!   the regression tests convert with `× ln 2`.
//! - **Atoms can be negative** — pointwise *and* averaged (e.g. XOR redundancy `= log(2/3) < 0`,
//!   COPY unique `< 0`). This is the deliberate "misinformation" content of SxPID; it is never
//!   clamped. Do not assert non-negativity.
//! - **Determinism**: the joint pmf is built over a `BTreeMap`, so realization order — and hence
//!   every floating-point accumulation — is fixed.
//!
//! # Complexity
//!
//! Brute-force over the empirical distribution: with `D` distinct realizations the cost is
//! `O(D² · #nodes · 2^{max collections})`. This mirrors the reference implementation and is meant
//! for low-effective-dimension discrete data (gates, or PLS/PCA-reduced targets).

use crate::discrete_pid::{
    discrete_antichains_3, discrete_mi, discrete_mobius_inversion_3, quantize_equal_width,
};
use crate::error::{PidError, PidResult};
use crate::matrix::MatRef;
use std::collections::BTreeMap;

/// A single shared-exclusions PID atom: the informative (`π⁺`) and misinformative (`π⁻`) parts
/// and their net `π = π⁺ − π⁻`. All in nats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SxAtom {
    pub informative: f64,
    pub misinformative: f64,
    pub net: f64,
}

/// One pointwise (per-realization) decomposition for the 2-source lattice.
///
/// `s1`, `s2`, `t` are the (binned) realization labels; `prob` its empirical probability. Atoms
/// are ordered as the 2-source lattice: unique-1 `{{1}}`, unique-2 `{{2}}`, synergy `{{1,2}}`,
/// redundancy `{{1},{2}}`.
#[derive(Debug, Clone)]
pub struct SxPointwise2 {
    pub s1: Vec<usize>,
    pub s2: Vec<usize>,
    pub t: Vec<usize>,
    pub prob: f64,
    pub unq1: SxAtom,
    pub unq2: SxAtom,
    pub syn: SxAtom,
    pub red: SxAtom,
}

/// Result of a discrete 2-source shared-exclusions PID.
#[derive(Debug, Clone)]
pub struct DiscreteSxPid2Result {
    /// One entry per distinct realization (the signature pointwise output of SxPID).
    pub pointwise: Vec<SxPointwise2>,
    /// Probability-weighted (averaged) atoms.
    pub unq1: SxAtom,
    pub unq2: SxAtom,
    pub syn: SxAtom,
    pub red: SxAtom,
    /// MI terms (nats), for the reconstruction / self-redundancy identities.
    pub mi_s1_t: f64,
    pub mi_s2_t: f64,
    pub mi_s1s2_t: f64,
    pub num_bins: usize,
}

/// One pointwise decomposition for the 3-source lattice (18 antichains, in the canonical
/// `discrete_antichains_3` order).
#[derive(Debug, Clone)]
pub struct SxPointwise3 {
    pub s0: Vec<usize>,
    pub s1: Vec<usize>,
    pub s2: Vec<usize>,
    pub t: Vec<usize>,
    pub prob: f64,
    pub atoms: Vec<SxAtom>,
}

/// Result of a discrete 3-source shared-exclusions PID.
#[derive(Debug, Clone)]
pub struct DiscreteSxPid3Result {
    pub pointwise: Vec<SxPointwise3>,
    /// The 18 antichains (as set-lists of bitmasks), aligned with `atoms`.
    pub antichains: Vec<Vec<u8>>,
    /// Averaged atoms, aligned with `antichains`.
    pub atoms: Vec<SxAtom>,
    pub mi_s0_t: f64,
    pub mi_s1_t: f64,
    pub mi_s2_t: f64,
    pub mi_s0s1s2_t: f64,
    pub num_bins: usize,
}

impl DiscreteSxPid3Result {
    /// Look up the averaged atom for an antichain given as a slice of bitmasks (e.g. `&[0b001,
    /// 0b010, 0b100]` for `{{0},{1},{2}}`). Order-insensitive.
    pub fn atom(&self, sets: &[u8]) -> Option<SxAtom> {
        let mut want = sets.to_vec();
        want.sort_unstable();
        self.antichains
            .iter()
            .position(|ac| {
                let mut a = ac.clone();
                a.sort_unstable();
                a == want
            })
            .map(|i| self.atoms[i])
    }
}

// ----------------------------------------------------------------------------------------------
// Core primitives
// ----------------------------------------------------------------------------------------------

/// Empirical joint pmf over distinct realizations. `var_bins[v]` is variable `v`'s per-sample
/// binned label vector; the last variable is the target. Returns `(realization, probability)`
/// pairs in a deterministic (`BTreeMap`) order.
fn build_pmf(var_bins: &[&[Vec<usize>]]) -> Vec<(Vec<Vec<usize>>, f64)> {
    let n = var_bins[0].len();
    let mut counts: BTreeMap<Vec<Vec<usize>>, usize> = BTreeMap::new();
    for i in 0..n {
        let rlz: Vec<Vec<usize>> = var_bins.iter().map(|vb| vb[i].clone()).collect();
        *counts.entry(rlz).or_insert(0) += 1;
    }
    let inv_n = 1.0 / n as f64;
    counts
        .into_iter()
        .map(|(k, c)| (k, c as f64 * inv_n))
        .collect()
}

/// Marginal probability of the event "agrees with `rlz` on the source indices in `source_mask`
/// (and on the target if `with_target`)".
fn marg(
    pmf: &[(Vec<Vec<usize>>, f64)],
    rlz: &[Vec<usize>],
    source_mask: u32,
    n_sources: usize,
    with_target: bool,
) -> f64 {
    let mut s = 0.0;
    for (cand, p) in pmf {
        let mut ok = true;
        for src in 0..n_sources {
            if source_mask & (1 << src) != 0 && cand[src] != rlz[src] {
                ok = false;
                break;
            }
        }
        if ok && with_target && cand[n_sources] != rlz[n_sources] {
            ok = false;
        }
        if ok {
            s += p;
        }
    }
    s
}

/// `P(⋃_j 𝔞_j)` (optionally intersected with the target event) via inclusion–exclusion over the
/// antichain's `collections` (each a source bitmask).
fn union_prob(
    pmf: &[(Vec<Vec<usize>>, f64)],
    rlz: &[Vec<usize>],
    collections: &[u8],
    n_sources: usize,
    with_target: bool,
) -> f64 {
    let k = collections.len();
    let mut total = 0.0;
    // Non-empty subsets I of the collections; a term fixes the UNION of their source indices.
    for subset in 1u32..(1u32 << k) {
        let mut idx_mask = 0u32;
        for (j, &c) in collections.iter().enumerate() {
            if subset & (1 << j) != 0 {
                idx_mask |= c as u32;
            }
        }
        let sign = if subset.count_ones() % 2 == 1 {
            1.0
        } else {
            -1.0
        };
        total += sign * marg(pmf, rlz, idx_mask, n_sources, with_target);
    }
    total
}

/// The three cumulative terms `(i⁺, i⁻, i_cap)` for one antichain node at one realization.
fn node_terms(
    pmf: &[(Vec<Vec<usize>>, f64)],
    rlz: &[Vec<usize>],
    collections: &[u8],
    n_sources: usize,
) -> PidResult<(f64, f64, f64)> {
    let p_t = marg(pmf, rlz, 0, n_sources, true);
    let p_union = union_prob(pmf, rlz, collections, n_sources, false);
    let p_t_union = union_prob(pmf, rlz, collections, n_sources, true);
    // The realization itself lies in every collection-event, so all three probabilities are >0
    // for any positive-mass realization. Guard defensively against accumulated round-off anyway.
    if !(p_t > 0.0 && p_union > 0.0 && p_t_union > 0.0) {
        return Err(PidError::NumericalInstability {
            context: "sxpid: degenerate union/target probability (non-positive)",
        });
    }
    let i_plus = -p_union.ln();
    let i_minus = (p_t / p_t_union).ln();
    Ok((i_plus, i_minus, i_plus - i_minus))
}

// ----------------------------------------------------------------------------------------------
// 2-source
// ----------------------------------------------------------------------------------------------

/// 2-source lattice nodes in the canonical order `[unq1, unq2, syn, red]`, each a list of source
/// collections (bitmasks over `{0,1}`).
const NODES2: [&[u8]; 4] = [&[0b01], &[0b10], &[0b11], &[0b01, 0b10]];

/// Explicit Möbius inversion of a length-4 cumulative vector (`[unq1, unq2, syn, red]` order) into
/// atoms. The lattice: `red` is the bottom; `unq1`, `unq2` cover `red`; `syn` is the top.
#[inline]
fn invert2(cum: [f64; 4]) -> [f64; 4] {
    let red = cum[3];
    let unq1 = cum[0] - red;
    let unq2 = cum[1] - red;
    let syn = cum[2] - unq1 - unq2 - red;
    [unq1, unq2, syn, red]
}

/// Discrete 2-source shared-exclusions PID (`i^sx_∩`).
pub fn discrete_sxpid2(
    s1: MatRef<'_>,
    s2: MatRef<'_>,
    target: MatRef<'_>,
    num_bins: usize,
) -> PidResult<DiscreteSxPid2Result> {
    if num_bins < 2 {
        return Err(PidError::InvalidConfig {
            context: "discrete_sxpid2",
            message: "num_bins must be >= 2",
        });
    }
    let n = s1.nrows();
    if s2.nrows() != n || target.nrows() != n {
        return Err(PidError::RowCountMismatch {
            context: "discrete_sxpid2",
            left_rows: n,
            right_rows: if s2.nrows() != n {
                s2.nrows()
            } else {
                target.nrows()
            },
        });
    }

    let s1_bins = quantize_equal_width(s1, num_bins)?;
    let s2_bins = quantize_equal_width(s2, num_bins)?;
    let t_bins = quantize_equal_width(target, num_bins)?;

    let mi_s1_t = discrete_mi(&s1_bins, &t_bins, num_bins)?;
    let mi_s2_t = discrete_mi(&s2_bins, &t_bins, num_bins)?;
    let mi_s1s2_t = discrete_mi(&join_pair(&s1_bins, &s2_bins), &t_bins, num_bins)?;

    let pmf = build_pmf(&[&s1_bins, &s2_bins, &t_bins]);
    let n_sources = 2;

    let mut pointwise = Vec::with_capacity(pmf.len());
    // Averaged accumulators for [unq1, unq2, syn, red] × (plus, minus, net).
    let mut avg = [[0.0f64; 3]; 4];

    for (rlz, prob) in &pmf {
        let mut cum_plus = [0.0f64; 4];
        let mut cum_minus = [0.0f64; 4];
        let mut cum_cap = [0.0f64; 4];
        for (node_idx, collections) in NODES2.iter().enumerate() {
            let (ip, im, ic) = node_terms(&pmf, rlz, collections, n_sources)?;
            cum_plus[node_idx] = ip;
            cum_minus[node_idx] = im;
            cum_cap[node_idx] = ic;
        }
        let pi_plus = invert2(cum_plus);
        let pi_minus = invert2(cum_minus);
        let pi_net = invert2(cum_cap);

        let atoms: [SxAtom; 4] = std::array::from_fn(|i| SxAtom {
            informative: pi_plus[i],
            misinformative: pi_minus[i],
            net: pi_net[i],
        });
        for i in 0..4 {
            avg[i][0] += prob * pi_plus[i];
            avg[i][1] += prob * pi_minus[i];
            avg[i][2] += prob * pi_net[i];
        }

        pointwise.push(SxPointwise2 {
            s1: rlz[0].clone(),
            s2: rlz[1].clone(),
            t: rlz[2].clone(),
            prob: *prob,
            unq1: atoms[0],
            unq2: atoms[1],
            syn: atoms[2],
            red: atoms[3],
        });
    }

    let mk = |a: [f64; 3]| SxAtom {
        informative: a[0],
        misinformative: a[1],
        net: a[2],
    };
    Ok(DiscreteSxPid2Result {
        pointwise,
        unq1: mk(avg[0]),
        unq2: mk(avg[1]),
        syn: mk(avg[2]),
        red: mk(avg[3]),
        mi_s1_t,
        mi_s2_t,
        mi_s1s2_t,
        num_bins,
    })
}

// ----------------------------------------------------------------------------------------------
// 3-source
// ----------------------------------------------------------------------------------------------

/// Discrete 3-source shared-exclusions PID over the 18-antichain lattice.
pub fn discrete_sxpid3(
    s0: MatRef<'_>,
    s1: MatRef<'_>,
    s2: MatRef<'_>,
    target: MatRef<'_>,
    num_bins: usize,
) -> PidResult<DiscreteSxPid3Result> {
    if num_bins < 2 {
        return Err(PidError::InvalidConfig {
            context: "discrete_sxpid3",
            message: "num_bins must be >= 2",
        });
    }
    let n = s0.nrows();
    if s1.nrows() != n || s2.nrows() != n || target.nrows() != n {
        let right_rows = if s1.nrows() != n {
            s1.nrows()
        } else if s2.nrows() != n {
            s2.nrows()
        } else {
            target.nrows()
        };
        return Err(PidError::RowCountMismatch {
            context: "discrete_sxpid3",
            left_rows: n,
            right_rows,
        });
    }

    let s0_bins = quantize_equal_width(s0, num_bins)?;
    let s1_bins = quantize_equal_width(s1, num_bins)?;
    let s2_bins = quantize_equal_width(s2, num_bins)?;
    let t_bins = quantize_equal_width(target, num_bins)?;

    let mi_s0_t = discrete_mi(&s0_bins, &t_bins, num_bins)?;
    let mi_s1_t = discrete_mi(&s1_bins, &t_bins, num_bins)?;
    let mi_s2_t = discrete_mi(&s2_bins, &t_bins, num_bins)?;
    let mi_s0s1s2_t = discrete_mi(
        &join_triple(&s0_bins, &s1_bins, &s2_bins),
        &t_bins,
        num_bins,
    )?;

    let antichains = discrete_antichains_3();
    // Each antichain's nonzero masks = its list of source collections.
    let node_collections: Vec<Vec<u8>> = antichains
        .iter()
        .map(|ac| ac.iter().copied().filter(|&m| m != 0).collect())
        .collect();

    let pmf = build_pmf(&[&s0_bins, &s1_bins, &s2_bins, &t_bins]);
    let n_sources = 3;
    let m = antichains.len();

    let mut pointwise = Vec::with_capacity(pmf.len());
    let mut avg = vec![[0.0f64; 3]; m];

    for (rlz, prob) in &pmf {
        let mut cum_plus = vec![0.0f64; m];
        let mut cum_minus = vec![0.0f64; m];
        let mut cum_cap = vec![0.0f64; m];
        for (idx, collections) in node_collections.iter().enumerate() {
            let (ip, im, ic) = node_terms(&pmf, rlz, collections, n_sources)?;
            cum_plus[idx] = ip;
            cum_minus[idx] = im;
            cum_cap[idx] = ic;
        }
        // Reuse the measure-agnostic Möbius inversion (returns atoms aligned with `antichains`).
        let pi_plus = discrete_mobius_inversion_3(&antichains, &cum_plus);
        let pi_minus = discrete_mobius_inversion_3(&antichains, &cum_minus);
        let pi_net = discrete_mobius_inversion_3(&antichains, &cum_cap);

        let mut atoms = Vec::with_capacity(m);
        for i in 0..m {
            let a = SxAtom {
                informative: pi_plus[i].value,
                misinformative: pi_minus[i].value,
                net: pi_net[i].value,
            };
            avg[i][0] += prob * a.informative;
            avg[i][1] += prob * a.misinformative;
            avg[i][2] += prob * a.net;
            atoms.push(a);
        }

        pointwise.push(SxPointwise3 {
            s0: rlz[0].clone(),
            s1: rlz[1].clone(),
            s2: rlz[2].clone(),
            t: rlz[3].clone(),
            prob: *prob,
            atoms,
        });
    }

    let atoms_avg: Vec<SxAtom> = avg
        .iter()
        .map(|a| SxAtom {
            informative: a[0],
            misinformative: a[1],
            net: a[2],
        })
        .collect();

    Ok(DiscreteSxPid3Result {
        pointwise,
        antichains: node_collections,
        atoms: atoms_avg,
        mi_s0_t,
        mi_s1_t,
        mi_s2_t,
        mi_s0s1s2_t,
        num_bins,
    })
}

// ----------------------------------------------------------------------------------------------
// Small local join helpers (the `discrete_pid` ones are private; these keep this module
// self-contained without widening that module's surface further).
// ----------------------------------------------------------------------------------------------

fn join_pair(a: &[Vec<usize>], b: &[Vec<usize>]) -> Vec<Vec<usize>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let mut r = x.clone();
            r.extend_from_slice(y);
            r
        })
        .collect()
}

fn join_triple(a: &[Vec<usize>], b: &[Vec<usize>], c: &[Vec<usize>]) -> Vec<Vec<usize>> {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((x, y), z)| {
            let mut r = x.clone();
            r.extend_from_slice(y);
            r.extend_from_slice(z);
            r
        })
        .collect()
}

// ----------------------------------------------------------------------------------------------
// General n-source (n = 2..=4) — same redundancy lattice machinery for arbitrary source count.
// The per-realization probability primitives (`union_prob`, `node_terms`) are already n-general;
// the only n-specific parts are the antichain enumeration and the Möbius inversion below. The
// 2- and 3-source `discrete_sxpid2/3` paths above are kept as the validated reference; a test
// pins this general path to reproduce them exactly.
// ----------------------------------------------------------------------------------------------

/// One pointwise decomposition for the general n-source lattice.
#[derive(Debug, Clone)]
pub struct SxPointwiseN {
    /// The realization as per-variable binned labels: `n_sources` sources then the target.
    pub realization: Vec<Vec<usize>>,
    pub prob: f64,
    /// Atoms aligned with [`DiscreteSxPidNResult::antichains`].
    pub atoms: Vec<SxAtom>,
}

/// Result of a general n-source discrete shared-exclusions PID.
#[derive(Debug, Clone)]
pub struct DiscreteSxPidNResult {
    pub n_sources: usize,
    /// Lattice nodes as set-lists of source bitmasks (canonical: each list sorted ascending).
    pub antichains: Vec<Vec<u8>>,
    /// Averaged atoms, aligned with `antichains`.
    pub atoms: Vec<SxAtom>,
    pub pointwise: Vec<SxPointwiseN>,
    /// Joint MI `I(S_0,…,S_{n-1}; T)` — the sum of all averaged net atoms (reconstruction).
    pub joint_mi: f64,
    pub num_bins: usize,
}

impl DiscreteSxPidNResult {
    /// Averaged atom for an antichain given as a slice of bitmasks (order-insensitive).
    pub fn atom(&self, sets: &[u8]) -> Option<SxAtom> {
        let mut want = sets.to_vec();
        want.sort_unstable();
        self.antichains
            .iter()
            .position(|ac| *ac == want)
            .map(|i| self.atoms[i])
    }
}

/// `a ⪯ b` on the redundancy lattice: every collection in `b` contains some collection in `a`.
/// (`aa ⊆ bb` is tested as `aa & !bb == 0` — no bit of `aa` lies outside `bb`.)
fn leq_n(a: &[u8], b: &[u8]) -> bool {
    b.iter().all(|&bb| a.iter().any(|&aa| aa & !bb == 0))
}

/// All antichains over the non-empty subsets of `{0..n}` (n ≤ 4), each canonicalised to an
/// ascending mask list. Brute-force over the powerset of the `2^n − 1` non-empty masks.
fn antichains_n(n: usize) -> Vec<Vec<u8>> {
    let masks: Vec<u8> = (1u16..(1u16 << n)).map(|m| m as u8).collect();
    let mut out = Vec::new();
    for combo in 1u32..(1u32 << masks.len()) {
        let sel: Vec<u8> = masks
            .iter()
            .enumerate()
            .filter(|(i, _)| combo & (1 << i) != 0)
            .map(|(_, &m)| m)
            .collect();
        // Antichain iff no member is a subset of another.
        let is_antichain =
            (0..sel.len()).all(|i| (0..sel.len()).all(|j| i == j || (sel[i] & sel[j]) != sel[i]));
        if is_antichain {
            out.push(sel); // already ascending: `masks` is ascending and the filter preserves order
        }
    }
    out
}

/// Möbius inversion of a per-antichain cumulative vector into atoms (general n).
fn mobius_n(antichains: &[Vec<u8>], cumulative: &[f64]) -> Vec<f64> {
    let m = antichains.len();
    let topo = topo_order_n(antichains);
    let mut atoms = vec![0.0f64; m];
    for (pos, &idx) in topo.iter().enumerate() {
        let mut val = cumulative[idx];
        for &j in &topo[..pos] {
            if leq_n(&antichains[j], &antichains[idx]) {
                val -= atoms[j];
            }
        }
        atoms[idx] = val;
    }
    atoms
}

/// Topological order (minimal elements first) of the antichain lattice.
fn topo_order_n(antichains: &[Vec<u8>]) -> Vec<usize> {
    let mut remaining: Vec<usize> = (0..antichains.len()).collect();
    let mut out = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let mut mins: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&i| {
                !remaining
                    .iter()
                    .any(|&j| j != i && leq_n(&antichains[j], &antichains[i]))
            })
            .collect();
        mins.sort_unstable();
        let chosen = mins[0];
        out.push(chosen);
        remaining.retain(|&x| x != chosen);
    }
    out
}

/// Discrete shared-exclusions PID for an arbitrary number of sources (`2 ≤ n ≤ 4`).
///
/// Same measure as [`discrete_sxpid2`]/[`discrete_sxpid3`] (which it reproduces exactly), extended
/// to the full antichain lattice for up to four sources — matching the source count IDTxl's SxPID
/// estimator supports. Atoms are keyed by their antichain (a set-list of source bitmasks), e.g.
/// `&[0b0001, 0b0010, 0b0100, 0b1000]` is the all-singletons (global) redundancy for `n = 4`.
pub fn discrete_sxpid_n(
    sources: &[MatRef<'_>],
    target: MatRef<'_>,
    num_bins: usize,
) -> PidResult<DiscreteSxPidNResult> {
    let n_sources = sources.len();
    if !(2..=4).contains(&n_sources) {
        return Err(PidError::NotImplemented {
            feature: "discrete_sxpid_n supports 2..=4 sources",
        });
    }
    if num_bins < 2 {
        return Err(PidError::InvalidConfig {
            context: "discrete_sxpid_n",
            message: "num_bins must be >= 2",
        });
    }
    let n = target.nrows();
    for s in sources {
        if s.nrows() != n {
            return Err(PidError::RowCountMismatch {
                context: "discrete_sxpid_n",
                left_rows: n,
                right_rows: s.nrows(),
            });
        }
    }

    let source_bins: Vec<Vec<Vec<usize>>> = sources
        .iter()
        .map(|s| quantize_equal_width(*s, num_bins))
        .collect::<PidResult<_>>()?;
    let t_bins = quantize_equal_width(target, num_bins)?;

    // Joint MI I(S_0..S_{n-1}; T) for the reconstruction field.
    let mut joined = vec![Vec::new(); n];
    for sb in &source_bins {
        for (i, row) in sb.iter().enumerate() {
            joined[i].extend_from_slice(row);
        }
    }
    let joint_mi = discrete_mi(&joined, &t_bins, num_bins)?;

    // var_bins = sources then target.
    let mut var_bins: Vec<&[Vec<usize>]> = source_bins.iter().map(|v| v.as_slice()).collect();
    var_bins.push(&t_bins);
    let pmf = build_pmf(&var_bins);

    let antichains = antichains_n(n_sources);
    let m = antichains.len();

    let mut pointwise = Vec::with_capacity(pmf.len());
    let mut avg = vec![[0.0f64; 3]; m];

    for (rlz, prob) in &pmf {
        let mut cum_plus = vec![0.0f64; m];
        let mut cum_minus = vec![0.0f64; m];
        let mut cum_cap = vec![0.0f64; m];
        for (idx, collections) in antichains.iter().enumerate() {
            let (ip, im, ic) = node_terms(&pmf, rlz, collections, n_sources)?;
            cum_plus[idx] = ip;
            cum_minus[idx] = im;
            cum_cap[idx] = ic;
        }
        let pi_plus = mobius_n(&antichains, &cum_plus);
        let pi_minus = mobius_n(&antichains, &cum_minus);
        let pi_net = mobius_n(&antichains, &cum_cap);

        let mut atoms = Vec::with_capacity(m);
        for i in 0..m {
            let a = SxAtom {
                informative: pi_plus[i],
                misinformative: pi_minus[i],
                net: pi_net[i],
            };
            avg[i][0] += prob * a.informative;
            avg[i][1] += prob * a.misinformative;
            avg[i][2] += prob * a.net;
            atoms.push(a);
        }
        pointwise.push(SxPointwiseN {
            realization: rlz.clone(),
            prob: *prob,
            atoms,
        });
    }

    let atoms_avg: Vec<SxAtom> = avg
        .iter()
        .map(|a| SxAtom {
            informative: a[0],
            misinformative: a[1],
            net: a[2],
        })
        .collect();

    Ok(DiscreteSxPidNResult {
        n_sources,
        antichains,
        atoms: atoms_avg,
        pointwise,
        joint_mi,
        num_bins,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatRef;
    use std::f64::consts::LN_2;

    /// Exactly-enumerated 2-input gate dataset (no sampling error → exact pmf).
    fn gate2(rows: &[(usize, usize, usize)], reps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        let mut t = Vec::new();
        for _ in 0..reps {
            for &(a, b, c) in rows {
                s1.push(a as f64);
                s2.push(b as f64);
                t.push(c as f64);
            }
        }
        let n = rows.len() * reps;
        (s1, s2, t, n)
    }

    fn run2(rows: &[(usize, usize, usize)], num_bins: usize) -> DiscreteSxPid2Result {
        let (s1, s2, t, n) = gate2(rows, 8);
        let s1 = MatRef::new(&s1, n, 1).unwrap();
        let s2 = MatRef::new(&s2, n, 1).unwrap();
        let t = MatRef::new(&t, n, 1).unwrap();
        discrete_sxpid2(s1, s2, t, num_bins).unwrap()
    }

    #[test]
    fn xor_pointwise_matches_reference() {
        // Reference (bits): every realization is [3/2, 3/2, 4/3, 2/3] in log2; here in nats (ln).
        let r = run2(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], 2);
        let want = [
            1.5_f64.ln(),
            1.5_f64.ln(),
            (4.0_f64 / 3.0).ln(),
            (2.0_f64 / 3.0).ln(),
        ];
        for p in &r.pointwise {
            for (got, w) in [p.unq1.net, p.unq2.net, p.syn.net, p.red.net]
                .iter()
                .zip(want)
            {
                assert!((got - w).abs() < 1e-12, "got {got} want {w}");
            }
            // net == informative − misinformative, always.
            for a in [p.unq1, p.unq2, p.syn, p.red] {
                assert!((a.net - (a.informative - a.misinformative)).abs() < 1e-12);
            }
        }
        // Averaged XOR shared = log2(2/3) bits (IDTxl's value), in nats.
        assert!((r.red.net - (2.0_f64 / 3.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn and_averaged_red_matches_idtxl() {
        // IDTxl: averaged shared(AND) = 0.12255624891826572 bits.
        let r = run2(&[(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], 2);
        let want_nats = 0.12255624891826572 * LN_2;
        assert!(
            (r.red.net - want_nats).abs() < 1e-12,
            "red={} want {want_nats}",
            r.red.net
        );
        // Reconstruction: atoms sum to I(S1,S2;T).
        let sum = r.unq1.net + r.unq2.net + r.syn.net + r.red.net;
        assert!((sum - r.mi_s1s2_t).abs() < 1e-9);
    }

    #[test]
    fn self_redundancy_and_reconstruction() {
        // UNQ gate T = S1: unq1+red = I(S1;T), and atoms sum to I(S1,S2;T).
        let r = run2(&[(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)], 2);
        assert!((r.unq1.net + r.red.net - r.mi_s1_t).abs() < 1e-9);
        assert!((r.unq2.net + r.red.net - r.mi_s2_t).abs() < 1e-9);
        let sum = r.unq1.net + r.unq2.net + r.syn.net + r.red.net;
        assert!((sum - r.mi_s1s2_t).abs() < 1e-9);
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
    fn tri_rnd_3source() {
        // Giant bit over 3 sources: all atoms 0 except {{0},{1},{2}} = log 2.
        let n = 8 * 2;
        let mut s0 = Vec::new();
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        let mut t = Vec::new();
        for _ in 0..8 {
            for b in [0.0, 1.0] {
                s0.push(b);
                s1.push(b);
                s2.push(b);
                t.push(b);
            }
        }
        let s0 = MatRef::new(&s0, n, 1).unwrap();
        let s1 = MatRef::new(&s1, n, 1).unwrap();
        let s2 = MatRef::new(&s2, n, 1).unwrap();
        let t = MatRef::new(&t, n, 1).unwrap();
        let r = discrete_sxpid3(s0, s1, s2, t, 2).unwrap();

        let red_all = r.atom(&[0b001, 0b010, 0b100]).unwrap();
        assert!(
            (red_all.net - 2.0_f64.ln()).abs() < 1e-12,
            "red_all={}",
            red_all.net
        );
        // Reconstruction: all atoms sum to the joint MI (= log 2 here).
        let sum: f64 = r.atoms.iter().map(|a| a.net).sum();
        assert!((sum - r.mi_s0s1s2_t).abs() < 1e-9);
        assert!((sum - 2.0_f64.ln()).abs() < 1e-9);
    }
}
