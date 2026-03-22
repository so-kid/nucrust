#![warn(missing_docs)]
//! R-matrix theory implementation for nuclear reaction cross sections.
//!
//! This crate implements the Lane-Thomas R-matrix formalism, Brune alternative
//! parameterization, and fitting algorithms (Levenberg-Marquardt, MCMC).
//!
//! # Modules
//!
//! - [`types`]: Data structures for R-matrix parameters, channels, levels, and results
//! - [`rmatrix`]: Core R-matrix cross section computation
//! - [`brune`]: Brune alternative parameterization and transformation
//! - [`fitting`]: Parameter optimization (LM) and uncertainty quantification (MCMC)

pub mod brune;
pub mod fitting;
pub mod rmatrix;
pub mod types;

pub use brune::{brune_to_standard, shift_function, standard_to_brune};
pub use fitting::{levenberg_marquardt, mcmc_sample};
pub use rmatrix::{rmatrix_cross_section, shift_penetrability};
pub use types::{
    BoundaryCondition, ExperimentalData, FitResult, LmConfig, McmcConfig, McmcResult, ParamIndex,
    ParticlePair, RMatrixChannel, RMatrixLevel, RMatrixParams, RMatrixResult,
};

pub use nucrust_core;
