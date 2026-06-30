#![doc = include_str!("../README.md")]
//!
//! ---
//!
//! `pid-core`: continuous mutual information + shared-exclusions PID (`I^sx_∩`) estimators.
//!
//! This crate implements:
//! - KSG mutual information (Kraskov et al. 2004) for continuous variables
//! - Wibral-group shared-exclusions redundancy `I^sx_∩(S1,S2;T)` (Makkeh et al. 2021)
//! - Continuous shared-exclusions estimator (Ehrlich et al. 2024)
//! - 2-source PID atoms derived from MI + `I^sx_∩`, and an optional 3-source SxPID
//! - A hierarchical "fast→slow" screening path for many-source settings
//!
//! # Units
//! All information quantities are reported in **nats** (natural logarithm).
//!
//! # Scientific contract
//! The mathematical object of interest is `I^sx_∩` and its derived PID atoms. Estimators are
//! finite-sample algorithms with failure modes; do not interpret results without first passing an
//! estimator-validation gate on synthetic systems with known information quantities (see the
//! `exp0` validation binary and the project README).
//!
//! # Estimator cautions (read before using on VLA embeddings)
//! - kNN estimators assume i.i.d. samples; trajectories violate this unless you subsample.
//! - High ambient/intrinsic dimension can collapse kNN geometry (distance concentration).
//! - Strong dependence (near-deterministic mappings) can require prohibitive samples even at low
//!   dimension.
//! - `I^sx_∩` (and PID atoms) are **not guaranteed non-negative** under all desiderata; negative
//!   values are possible and must be representable.
#![forbid(unsafe_code)]

mod bootstrap;
mod ci;
mod discrete_pid;
mod distance_matrix;
mod error;
mod geometry;
mod hierarchy;
mod hyperbolic;
mod invariants;
mod isx;
mod ksg;
mod logistic;
mod matrix;
mod metric;
mod nn;
mod par;
mod pid2;
mod pid3;
mod pipeline;
mod pls;
mod preprocess;
mod stats;
mod sxpid;

pub use bootstrap::{block_bootstrap, block_bootstrap_paired, BootstrapConfig, BootstrapResult};
pub use ci::{co_information_pairwise, co_information_triplet};
pub use discrete_pid::{
    discrete_entropy, discrete_mi, discrete_pid2, discrete_pid3, quantize_equal_width,
    DiscretePid2Result, DiscretePid3Atom, DiscretePid3Result,
};
pub use distance_matrix::{symmetric_distances, SymmetricDistanceMatrix};
pub use error::{PidError, PidResult};
pub use geometry::{
    distance_concentration_stats, gromov_hyperbolicity, intrinsic_dimension_levina_bickel,
    DistanceConcentrationConfig, DistanceConcentrationStats, HyperbolicityConfig,
    IntrinsicDimConfig,
};
pub use hierarchy::{
    hierarchical_pairwise, hierarchical_triplet, HierarchicalConfig, HierarchicalTriplet,
    PairSelection, PairwiseScreen,
};
pub use hyperbolic::{hyperbolic_distance_lorentz, lorentz_dot, poincare_to_lorentz};
pub use invariants::{
    average_degree_of_redundancy, average_degree_of_vulnerability,
    co_information_pairwise_discrete, entropy_discrete, joint_entropy_discrete,
    o_information_discrete, red_degree_discrete, vul_degree_discrete,
};
pub use isx::{isx_redundancy, IsxConfig, IsxMethod};
pub use ksg::{ksg_local_mi_terms, ksg_mi, ksg_mi_concat_xy, KsgConfig, NegativeHandling};
pub use logistic::{LogisticRegression, LogisticRegressionConfig};
pub use matrix::{concat_horiz, MatOwned, MatRef};
pub use metric::Metric;
pub use pid2::{pid2_isx, pid2_isx_estimate, Pid2Config, Pid2Estimate, Pid2Result};
pub use pid3::{pid3_isx, Antichain3, Pid3Atom, Pid3Config, Pid3Redundancy, Pid3Result};
pub use pipeline::{
    bootstrap_pid3, bootstrap_rows_stats, permutation_pid3, permutation_rows_pvalue,
    pls_cv_select_components, pls_project_then_discrete_pid3, pls_project_then_pid3,
    screen_pid2_pairs, BootstrapPid3Result, PermutationPid3Atom, PermutationPid3Result,
    Pid2ScreenEntry, Pid3BootstrapAtom, PlsCvResult, PlsDiscretePid3Config, PlsDiscretePid3Result,
    PlsPid3Config, PlsPid3Result, RowBootstrapResult, RowBootstrapStat, RowPermutationStat,
    RowResampleScheme,
};
pub use pls::PlsProjector;
pub use preprocess::{HashProjector, Jitter, PcaProjector, Standardizer};
pub use sxpid::{
    discrete_sxpid2, discrete_sxpid3, DiscreteSxPid2Result, DiscreteSxPid3Result, SxAtom,
    SxPointwise2, SxPointwise3,
};
