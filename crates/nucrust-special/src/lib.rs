//! Special functions for nuclear physics calculations.
//!
//! This crate provides Coulomb wave functions and related special functions
//! used in nuclear reaction theory. The primary entry point is [`coulomb_wave`],
//! which computes the regular and irregular Coulomb wave functions F_l and G_l,
//! their derivatives, and the Coulomb phase shifts using the Thompson-Barnett
//! (Steed) algorithm with continued fractions.
//!
//! # Key functions
//!
//! - [`coulomb_wave`] -- single-point Coulomb wave function computation
//! - [`coulomb_wave_batch`] -- batch computation over multiple (eta, rho) pairs
//! - [`coulomb_wave_batch_simd`] -- SIMD-accelerated batch computation
//! - [`complex_log_gamma`] -- complex log-gamma via Spouge approximation
//! - [`coulomb_phase_shift`] -- Coulomb phase shift sigma_l
//! - [`gamow_factor`] -- Gamow penetration factor C_l(eta)

#![warn(missing_docs)]

/// Shared numerical constants (convergence thresholds, zero guards).
mod consts;
/// Coulomb wave function computation (Thompson-Barnett / Steed algorithm).
pub mod coulomb;
/// Error types for special function computations.
pub mod error;
/// Gamma function and related utilities (Spouge approximation).
pub mod gamma;
/// Modified Lentz-Thompson-Barnett continued fraction evaluators.
pub mod lentz;
/// SIMD-accelerated batch Coulomb wave function computation.
pub mod simd_batch;

pub use coulomb::{coulomb_wave, coulomb_wave_batch, CoulombResult, ScalingMode};
pub use error::SpecialError;
pub use gamma::{complex_log_gamma, coulomb_phase_shift, gamow_factor};
pub use simd_batch::coulomb_wave_batch_simd;
