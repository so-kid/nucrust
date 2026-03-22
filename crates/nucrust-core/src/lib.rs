//! Core types, physical constants, and traits for the nucrust nuclear reaction framework.
//!
//! This crate provides the foundational building blocks shared across all nucrust crates:
//!
//! - **Nuclear identifiers**: [`Nuclide`], [`Projectile`], [`Channel`]
//! - **Quantum numbers**: [`SpinParity`], [`Parity`]
//! - **Computation results**: [`TransmissionCoeffs`], [`CrossSection`], [`CollisionMatrix`], [`ReactionRate`]
//! - **Numerical utilities**: [`EnergyGrid`], [`CubicSpline`]
//! - **Physical constants and kinematics**: [`units`] module
//! - **Angular momentum coupling**: [`wigner`] module (3j, 6j, Clebsch-Gordan)
//! - **Physics model traits**: [`OpticalPotential`], [`LevelDensity`], [`GammaStrength`]
//! - **Backend abstraction**: [`ComputeBackend`] trait for CPU/GPU dispatch

#![warn(missing_docs)]

/// Compute backend abstraction for CPU/GPU dispatch.
pub mod backend;
/// Reaction channel definitions (entrance/exit).
pub mod channel;
/// Collision matrix U_{cc'}(E) for R-matrix calculations.
pub mod collision_matrix;
/// Coupled-channel optical model types (deformation, rotational bands).
pub mod coupled_channel;
/// Cross section data structures.
pub mod cross_section;
/// Energy grid types for tabulated calculations.
pub mod energy;
/// Error types for the nucrust-core crate.
pub mod error;
/// Nuclide (nucleus) identifier.
pub mod nuclide;
/// Projectile and ejectile particle types.
pub mod projectile;
/// Astrophysical reaction rate results.
pub mod reaction_rate;
/// Spin and parity quantum numbers.
pub mod spin;
/// Cubic spline interpolation.
pub mod spline;
/// Physics model traits (optical potential, level density, gamma strength).
pub mod traits;
/// Transmission coefficient tables.
pub mod transmission;
/// Physical constants (CODATA 2022) and kinematic utilities.
pub mod units;
/// Wigner 3j, 6j symbols and Clebsch-Gordan coefficients.
pub mod wigner;

pub use backend::{ComputeBackend, GsfModelParams, NldModelParams, ToDeviceParams};
pub use channel::Channel;
pub use collision_matrix::CollisionMatrix;
pub use coupled_channel::{
    CoupledState, CoupledTransmission, DeformationParams, RotationalBand, TransmissionOutput,
};
pub use cross_section::{CrossSection, PartialCrossSection};
pub use energy::EnergyGrid;
pub use error::CoreError;
pub use nuclide::Nuclide;
pub use projectile::Projectile;
pub use reaction_rate::ReactionRate;
pub use spin::{Parity, SpinParity};
pub use spline::CubicSpline;
pub use traits::{GammaStrength, LevelDensity, Multipole, OpticalPotential};
pub use transmission::TransmissionCoeffs;
