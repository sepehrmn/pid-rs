use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pid_core::{
    average_degree_of_redundancy, average_degree_of_vulnerability, co_information_pairwise,
    discrete_pid2, discrete_pid3, discrete_sxpid2, discrete_sxpid3, discrete_sxpid_n,
    distance_concentration_stats, gromov_hyperbolicity, intrinsic_dimension_levina_bickel,
    isx_redundancy, ksg_mi, ksg_mi_concat_xy, pid2_isx, pid3_isx, DistanceConcentrationConfig,
    HashProjector, HyperbolicityConfig, IntrinsicDimConfig, IsxConfig, IsxMethod, KsgConfig,
    MatRef, Metric, NegativeHandling, PcaProjector, Pid2Config, Pid3Config, PlsProjector,
    Standardizer,
};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Convert a numpy array to a `MatRef` borrowing its buffer.
///
/// Requires a **C-contiguous** array. `as_slice()` also accepts a Fortran-contiguous buffer and
/// hands back its column-major bytes, which `MatRef` (row-major) would then read as the
/// transpose — silently producing wrong results for any non-square input (e.g. a transposed or
/// `order="F"` array). We reject non-C-contiguous input up front with an actionable error
/// instead. Non-finite values are rejected by `MatRef::new`.
fn array_to_matref<'a>(arr: &'a PyReadonlyArray2<f64>) -> PyResult<MatRef<'a>> {
    if !arr.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Array must be C-contiguous; wrap it in np.ascontiguousarray(x) \
             (e.g. for a transposed or order='F' array) before passing it in",
        ));
    }
    let slice = arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let arr_view = arr.as_array();
    let (nrows, ncols) = (arr_view.shape()[0], arr_view.shape()[1]);

    MatRef::new(slice, nrows, ncols)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid data: {e}")))
}

fn parse_metric(name: &str) -> PyResult<Metric> {
    match name.to_lowercase().as_str() {
        "chebyshev" | "linf" | "max" => Ok(Metric::Chebyshev),
        // Experimental research metrics (MI-only, not validated for ISX):
        "hyperbolic" | "lorentz" => Ok(Metric::HyperbolicLorentz),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown metric: '{}'. Valid metrics are: 'chebyshev' (aliases: 'linf', 'max'), \
             'hyperbolic' (alias: 'lorentz', experimental MI-only)",
            name
        ))),
    }
}

fn parse_negative_handling(name: &str) -> PyResult<NegativeHandling> {
    match name.to_lowercase().as_str() {
        "allow" | "raw" | "none" => Ok(NegativeHandling::Allow),
        "clamp_to_zero" | "clamp" | "zero" => Ok(NegativeHandling::ClampToZero),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown negative_handling: '{}'. Valid values are: 'allow', 'clamp_to_zero'",
            name
        ))),
    }
}

fn parse_isx_method(name: &str) -> PyResult<IsxMethod> {
    match name.to_lowercase().as_str() {
        "ehrlich_ksg" | "continuous" => Ok(IsxMethod::EhrlichKsg),
        "heuristic_sketch" | "sketch" => Ok(IsxMethod::HeuristicSketch),
        "local_min_ksg" | "local_min" => Ok(IsxMethod::LocalMinKsg),
        "disjunction_from_local_mi" | "disjunction" => Ok(IsxMethod::DisjunctionFromLocalMi),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown method: '{}'. Valid methods are: 'ehrlich_ksg', 'heuristic_sketch', \
             'local_min_ksg', 'disjunction_from_local_mi'",
            name
        ))),
    }
}

fn make_ksg_config(
    k: usize,
    metric: &str,
    tie_epsilon: f64,
    negative_handling: &str,
) -> PyResult<KsgConfig> {
    Ok(KsgConfig {
        k,
        metric: parse_metric(metric)?,
        tie_epsilon,
        negative_handling: parse_negative_handling(negative_handling)?,
    })
}

fn make_isx_config(k: usize, metric: &str, tie_epsilon: f64, method: &str) -> PyResult<IsxConfig> {
    Ok(IsxConfig {
        k,
        metric: parse_metric(metric)?,
        tie_epsilon,
        method: parse_isx_method(method)?,
    })
}

fn pid_err(e: pid_core::PidError) -> PyErr {
    use pid_core::PidError as E;
    let msg = e.to_string();
    match e {
        // Caller-supplied bad input / configuration → ValueError (consistent with the
        // contiguity and shape checks in `array_to_matref`).
        E::ShapeMismatch { .. }
        | E::InvalidConfig { .. }
        | E::RowCountMismatch { .. }
        | E::InvalidK { .. }
        | E::NonFiniteInput { .. } => pyo3::exceptions::PyValueError::new_err(msg),
        // Estimator could not produce a result on otherwise-valid input → RuntimeError.
        E::NumericalInstability { .. } => pyo3::exceptions::PyRuntimeError::new_err(msg),
        E::NotImplemented { .. } => pyo3::exceptions::PyNotImplementedError::new_err(msg),
    }
}

/// Compute KSG Mutual Information.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x, y, k=3, metric="chebyshev", tie_epsilon=0.0, negative_handling="clamp_to_zero"))]
fn compute_mi(
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    k: usize,
    metric: &str,
    tie_epsilon: f64,
    negative_handling: &str,
) -> PyResult<f64> {
    let x_mat = array_to_matref(&x)?;
    let y_mat = array_to_matref(&y)?;
    let cfg = make_ksg_config(k, metric, tie_epsilon, negative_handling)?;

    ksg_mi(x_mat, y_mat, &cfg).map_err(pid_err)
}

/// Compute continuous I_sx_intersect redundancy.
#[pyfunction]
#[pyo3(signature = (s1, s2, target, k=3, method="ehrlich_ksg", metric="chebyshev", tie_epsilon=0.0))]
fn compute_redundancy(
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    k: usize,
    method: &str,
    metric: &str,
    tie_epsilon: f64,
) -> PyResult<f64> {
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let cfg = make_isx_config(k, metric, tie_epsilon, method)?;

    isx_redundancy(s1_mat, s2_mat, t_mat, &cfg).map_err(pid_err)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, target, k=3, metric="chebyshev", tie_epsilon=0.0, negative_handling="allow"))]
fn compute_co_information(
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    k: usize,
    metric: &str,
    tie_epsilon: f64,
    negative_handling: &str,
) -> PyResult<f64> {
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let cfg = make_ksg_config(k, metric, tie_epsilon, negative_handling)?;

    co_information_pairwise(s1_mat, s2_mat, t_mat, &cfg).map_err(pid_err)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, target, k=3, method="ehrlich_ksg", metric="chebyshev", tie_epsilon=0.0, negative_handling="allow"))]
#[allow(clippy::too_many_arguments)]
fn compute_pid2(
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    k: usize,
    method: &str,
    metric: &str,
    tie_epsilon: f64,
    negative_handling: &str,
) -> PyResult<HashMap<String, f64>> {
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let cfg = Pid2Config {
        ksg: make_ksg_config(k, metric, tie_epsilon, negative_handling)?,
        isx: make_isx_config(k, metric, tie_epsilon, method)?,
    };
    let out = pid2_isx(s1_mat, s2_mat, t_mat, &cfg).map_err(pid_err)?;

    let mut map = HashMap::new();
    map.insert("redundancy".to_string(), out.redundancy);
    map.insert("unique_s1".to_string(), out.unique_s1);
    map.insert("unique_s2".to_string(), out.unique_s2);
    map.insert("synergy".to_string(), out.synergy);
    Ok(map)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, target, k=3, metric="chebyshev", tie_epsilon=0.0, negative_handling="allow"))]
fn compute_invariants(
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    k: usize,
    metric: &str,
    tie_epsilon: f64,
    negative_handling: &str,
) -> PyResult<HashMap<String, f64>> {
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let cfg = make_ksg_config(k, metric, tie_epsilon, negative_handling)?;
    let mi_s1_t = ksg_mi(s1_mat, t_mat, &cfg).map_err(pid_err)?;
    let mi_s2_t = ksg_mi(s2_mat, t_mat, &cfg).map_err(pid_err)?;
    let mi_s1s2_t = ksg_mi_concat_xy(s1_mat, s2_mat, t_mat, &cfg).map_err(pid_err)?;
    let ci = co_information_pairwise(s1_mat, s2_mat, t_mat, &cfg).map_err(pid_err)?;

    let mut map = HashMap::new();
    map.insert("mi_s1_t".to_string(), mi_s1_t);
    map.insert("mi_s2_t".to_string(), mi_s2_t);
    map.insert("mi_s1s2_t".to_string(), mi_s1s2_t);
    map.insert("co_information".to_string(), ci);
    map.insert(
        "r_bar".to_string(),
        average_degree_of_redundancy(&[mi_s1_t, mi_s2_t], mi_s1s2_t),
    );
    map.insert(
        "v_bar".to_string(),
        average_degree_of_vulnerability(mi_s1s2_t, &[mi_s2_t, mi_s1_t]),
    );
    Ok(map)
}

/// Estimate intrinsic dimension using Levina-Bickel (kNN MLE).
#[pyfunction]
#[pyo3(signature = (x, k=10, metric="chebyshev"))]
fn estimate_intrinsic_dimension(x: PyReadonlyArray2<f64>, k: usize, metric: &str) -> PyResult<f64> {
    let x_mat = array_to_matref(&x)?;
    let metric_enum = parse_metric(metric)?;

    let cfg = IntrinsicDimConfig {
        k,
        metric: metric_enum,
    };

    intrinsic_dimension_levina_bickel(x_mat, &cfg).map_err(pid_err)
}

/// Estimate Gromov delta-hyperbolicity via 4-point sampling.
#[pyfunction]
#[pyo3(signature = (x, n_samples=1000, metric="chebyshev", seed=42))]
fn estimate_gromov_delta(
    x: PyReadonlyArray2<f64>,
    n_samples: usize,
    metric: &str,
    seed: u64,
) -> PyResult<f64> {
    let x_mat = array_to_matref(&x)?;
    let metric_enum = parse_metric(metric)?;

    let cfg = HyperbolicityConfig {
        n_samples,
        metric: metric_enum,
        seed,
    };

    gromov_hyperbolicity(x_mat, &cfg).map_err(pid_err)
}

/// Compute distance concentration statistics.
/// Returns a dict with summary statistics including:
/// - pairwise min/max/mean/std/cv
/// - nearest-neighbor mean/cv and nn_over_pairwise_mean
#[pyfunction]
#[pyo3(signature = (x, metric="chebyshev"))]
fn distance_stats(x: PyReadonlyArray2<f64>, metric: &str) -> PyResult<HashMap<String, f64>> {
    let x_mat = array_to_matref(&x)?;
    let metric_enum = parse_metric(metric)?;

    let cfg = DistanceConcentrationConfig {
        metric: metric_enum,
    };

    let stats = distance_concentration_stats(x_mat, &cfg).map_err(pid_err)?;

    let mut map = HashMap::new();
    map.insert("pairwise_count".to_string(), stats.pairwise_count as f64);
    map.insert("pairwise_min".to_string(), stats.pairwise_min);
    map.insert("pairwise_max".to_string(), stats.pairwise_max);
    map.insert("pairwise_mean".to_string(), stats.pairwise_mean);
    map.insert("pairwise_std".to_string(), stats.pairwise_std);
    map.insert("pairwise_cv".to_string(), stats.pairwise_cv);
    map.insert("nn_min".to_string(), stats.nn_min);
    map.insert("nn_max".to_string(), stats.nn_max);
    map.insert("nn_mean".to_string(), stats.nn_mean);
    map.insert("nn_cv".to_string(), stats.nn_cv);
    map.insert(
        "nn_over_pairwise_mean".to_string(),
        stats.nn_over_pairwise_mean,
    );
    Ok(map)
}

/// Compute 3-source SxPID (18 atoms via Möbius inversion on the redundancy lattice).
#[pyfunction]
#[pyo3(signature = (s1, s2, s3, target, k=3, metric="chebyshev", tie_epsilon=0.0))]
#[allow(clippy::too_many_arguments)]
fn compute_pid3(
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    s3: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    k: usize,
    metric: &str,
    tie_epsilon: f64,
) -> PyResult<HashMap<String, f64>> {
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let s3_mat = array_to_matref(&s3)?;
    let t_mat = array_to_matref(&target)?;
    let cfg = Pid3Config {
        k,
        metric: parse_metric(metric)?,
        tie_epsilon,
    };
    let out = pid3_isx(s1_mat, s2_mat, s3_mat, t_mat, &cfg).map_err(pid_err)?;

    let mut map = HashMap::new();
    for atom in &out.atoms {
        map.insert(format!("{:?}", atom.antichain), atom.value);
    }
    Ok(map)
}

/// Compute discrete 2-source PID via quantization.
///
/// Useful as a fallback when continuous kNN-based estimation fails due to
/// distance concentration (high intrinsic dimension).
#[pyfunction]
#[pyo3(signature = (s1, s2, target, num_bins=10))]
fn compute_discrete_pid2(
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    num_bins: usize,
) -> PyResult<HashMap<String, f64>> {
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let out = discrete_pid2(s1_mat, s2_mat, t_mat, num_bins).map_err(pid_err)?;

    let mut map = HashMap::new();
    map.insert("redundancy".to_string(), out.redundancy);
    map.insert("unique_s1".to_string(), out.unique_s1);
    map.insert("unique_s2".to_string(), out.unique_s2);
    map.insert("synergy".to_string(), out.synergy);
    map.insert("mi_s1_t".to_string(), out.mi_s1_t);
    map.insert("mi_s2_t".to_string(), out.mi_s2_t);
    map.insert("mi_s1s2_t".to_string(), out.mi_s1s2_t);
    Ok(map)
}

/// Compute discrete 3-source PID via quantization (Williams–Beer `I_min` redundancy).
///
/// The discrete counterpart to `compute_pid3`. Note this is a **different PID
/// measure** from the continuous `I^sx_∩` (a different PID measure):
/// do not pool its atoms with continuous-mode atoms. Keys are the antichain set
/// indices of each atom on the 3-source lattice.
#[pyfunction]
#[pyo3(signature = (s0, s1, s2, target, num_bins=10))]
fn compute_discrete_pid3(
    s0: PyReadonlyArray2<f64>,
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    num_bins: usize,
) -> PyResult<HashMap<String, f64>> {
    let s0_mat = array_to_matref(&s0)?;
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let out = discrete_pid3(s0_mat, s1_mat, s2_mat, t_mat, num_bins).map_err(pid_err)?;

    let mut map = HashMap::new();
    for atom in &out.atoms {
        map.insert(format!("{:?}", atom.antichain_sets), atom.value);
    }
    Ok(map)
}

/// Compute discrete 2-source **shared-exclusions** PID (`i^sx_∩`, Makkeh/Gutknecht/Wibral 2021).
///
/// This is the genuine Wibral-group SxPID redundancy — the discrete sibling of the continuous
/// `I^sx_∩` (`compute_pid2`) and a **different measure** from `compute_discrete_pid2` (which is
/// Williams–Beer `I_min`). Returns the probability-weighted (averaged) atoms in **nats**; each
/// atom is reported as net plus its informative/misinformative split. Atoms may be negative.
#[pyfunction]
#[pyo3(signature = (s1, s2, target, num_bins=10))]
fn compute_discrete_sxpid2(
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    num_bins: usize,
) -> PyResult<HashMap<String, f64>> {
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let out = discrete_sxpid2(s1_mat, s2_mat, t_mat, num_bins).map_err(pid_err)?;

    let mut map = HashMap::new();
    for (name, a) in [
        ("redundancy", out.red),
        ("unique_s1", out.unq1),
        ("unique_s2", out.unq2),
        ("synergy", out.syn),
    ] {
        map.insert(name.to_string(), a.net);
        map.insert(format!("{name}_informative"), a.informative);
        map.insert(format!("{name}_misinformative"), a.misinformative);
    }
    map.insert("mi_s1_t".to_string(), out.mi_s1_t);
    map.insert("mi_s2_t".to_string(), out.mi_s2_t);
    map.insert("mi_s1s2_t".to_string(), out.mi_s1s2_t);
    Ok(map)
}

/// Compute discrete 3-source **shared-exclusions** PID (`i^sx_∩`) over the 18-antichain lattice.
///
/// Averaged atoms in **nats**, keyed by the antichain set-list (e.g. `"[1, 2, 4]"` for the
/// all-singletons redundancy `{{0},{1},{2}}`). A different measure from `compute_discrete_pid3`
/// (`I_min`); atoms may be negative.
#[pyfunction]
#[pyo3(signature = (s0, s1, s2, target, num_bins=10))]
fn compute_discrete_sxpid3(
    s0: PyReadonlyArray2<f64>,
    s1: PyReadonlyArray2<f64>,
    s2: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    num_bins: usize,
) -> PyResult<HashMap<String, f64>> {
    let s0_mat = array_to_matref(&s0)?;
    let s1_mat = array_to_matref(&s1)?;
    let s2_mat = array_to_matref(&s2)?;
    let t_mat = array_to_matref(&target)?;
    let out = discrete_sxpid3(s0_mat, s1_mat, s2_mat, t_mat, num_bins).map_err(pid_err)?;

    let mut map = HashMap::new();
    for (sets, atom) in out.antichains.iter().zip(&out.atoms) {
        map.insert(format!("{sets:?}"), atom.net);
    }
    Ok(map)
}

/// Compute discrete **shared-exclusions** PID (`i^sx_∩`) for an arbitrary number of sources
/// (`2 ≤ len(sources) ≤ 4`, the count IDTxl's SxPID supports).
///
/// Averaged net atoms in **nats**, keyed by the antichain set-list of source bitmasks (e.g.
/// `"[1, 2, 4, 8]"` is the all-singletons global redundancy for 4 sources). Same measure as
/// `compute_discrete_sxpid2/3`, extended to the full lattice. Atoms may be negative.
#[pyfunction]
#[pyo3(signature = (sources, target, num_bins=10))]
fn compute_discrete_sxpid_n(
    sources: Vec<PyReadonlyArray2<f64>>,
    target: PyReadonlyArray2<f64>,
    num_bins: usize,
) -> PyResult<HashMap<String, f64>> {
    let src_mats: Vec<MatRef<'_>> = sources
        .iter()
        .map(array_to_matref)
        .collect::<PyResult<_>>()?;
    let t_mat = array_to_matref(&target)?;
    let out = discrete_sxpid_n(&src_mats, t_mat, num_bins).map_err(pid_err)?;

    let mut map = HashMap::new();
    for (sets, atom) in out.antichains.iter().zip(&out.atoms) {
        map.insert(format!("{sets:?}"), atom.net);
    }
    Ok(map)
}

/// Fit PLS (Partial Least Squares) supervised dimensionality reduction and project X.
///
/// Projects high-dimensional X onto directions maximally correlated with target Y.
/// Unlike PCA, PLS uses label information to find the task-relevant subspace.
/// Returns the projected data as a 2D numpy-compatible flat list + (nrows, ncols).
#[pyfunction]
#[pyo3(signature = (x, y, out_dim))]
fn pls_transform(
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    out_dim: usize,
) -> PyResult<HashMap<String, PyObject>> {
    let x_mat = array_to_matref(&x)?;
    let y_mat = array_to_matref(&y)?;
    let (projected, _pls) = PlsProjector::fit_transform(x_mat, y_mat, out_dim).map_err(pid_err)?;

    let ref_view = projected.as_ref();
    let n = ref_view.nrows();
    let d = ref_view.ncols();
    let mut flat = Vec::with_capacity(n * d);
    for i in 0..n {
        flat.extend_from_slice(ref_view.row(i));
    }

    Python::with_gil(|py| {
        let mut map = HashMap::new();
        map.insert(
            "data".to_string(),
            flat.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "nrows".to_string(),
            n.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "ncols".to_string(),
            d.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(map)
    })
}

/// Standardize a matrix (zero mean, unit variance per column).
#[pyfunction]
#[pyo3(signature = (x))]
fn standardize(x: PyReadonlyArray2<f64>) -> PyResult<HashMap<String, PyObject>> {
    let x_mat = array_to_matref(&x)?;
    let (projected, _std) = Standardizer::fit_transform(x_mat).map_err(pid_err)?;

    let ref_view = projected.as_ref();
    let n = ref_view.nrows();
    let d = ref_view.ncols();
    let mut flat = Vec::with_capacity(n * d);
    for i in 0..n {
        flat.extend_from_slice(ref_view.row(i));
    }

    Python::with_gil(|py| {
        let mut map = HashMap::new();
        map.insert(
            "data".to_string(),
            flat.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "nrows".to_string(),
            n.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "ncols".to_string(),
            d.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(map)
    })
}

/// PCA dimensionality reduction.
#[pyfunction]
#[pyo3(signature = (x, out_dim))]
fn pca_transform(x: PyReadonlyArray2<f64>, out_dim: usize) -> PyResult<HashMap<String, PyObject>> {
    let x_mat = array_to_matref(&x)?;
    let (projected, _pca) = PcaProjector::fit_transform(x_mat, out_dim).map_err(pid_err)?;

    let ref_view = projected.as_ref();
    let n = ref_view.nrows();
    let d = ref_view.ncols();
    let mut flat = Vec::with_capacity(n * d);
    for i in 0..n {
        flat.extend_from_slice(ref_view.row(i));
    }

    Python::with_gil(|py| {
        let mut map = HashMap::new();
        map.insert(
            "data".to_string(),
            flat.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "nrows".to_string(),
            n.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "ncols".to_string(),
            d.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(map)
    })
}

/// Hash-based (CountSketch) dimensionality reduction.
#[pyfunction]
#[pyo3(signature = (x, out_dim, seed=42))]
fn hash_project(
    x: PyReadonlyArray2<f64>,
    out_dim: usize,
    seed: u64,
) -> PyResult<HashMap<String, PyObject>> {
    let x_mat = array_to_matref(&x)?;
    let proj = HashProjector::new(x_mat.ncols(), out_dim, seed).map_err(pid_err)?;
    let projected = proj.transform(x_mat).map_err(pid_err)?;

    let ref_view = projected.as_ref();
    let n = ref_view.nrows();
    let d = ref_view.ncols();
    let mut flat = Vec::with_capacity(n * d);
    for i in 0..n {
        flat.extend_from_slice(ref_view.row(i));
    }

    Python::with_gil(|py| {
        let mut map = HashMap::new();
        map.insert(
            "data".to_string(),
            flat.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "nrows".to_string(),
            n.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "ncols".to_string(),
            d.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(map)
    })
}

#[pymodule]
fn pid_core_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_redundancy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_co_information, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pid2, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pid3, m)?)?;
    m.add_function(wrap_pyfunction!(compute_discrete_pid2, m)?)?;
    m.add_function(wrap_pyfunction!(compute_discrete_pid3, m)?)?;
    m.add_function(wrap_pyfunction!(compute_discrete_sxpid2, m)?)?;
    m.add_function(wrap_pyfunction!(compute_discrete_sxpid3, m)?)?;
    m.add_function(wrap_pyfunction!(compute_discrete_sxpid_n, m)?)?;
    m.add_function(wrap_pyfunction!(compute_invariants, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_intrinsic_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_gromov_delta, m)?)?;
    m.add_function(wrap_pyfunction!(distance_stats, m)?)?;
    m.add_function(wrap_pyfunction!(pls_transform, m)?)?;
    m.add_function(wrap_pyfunction!(standardize, m)?)?;
    m.add_function(wrap_pyfunction!(pca_transform, m)?)?;
    m.add_function(wrap_pyfunction!(hash_project, m)?)?;
    Ok(())
}
