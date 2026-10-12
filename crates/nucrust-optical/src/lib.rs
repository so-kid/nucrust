//! Optical model potentials and Numerov integration for nuclear transmission coefficients.
//!
//! This crate provides spherical and deformed optical model potentials (Koning-Delaroche,
//! McFadden-Satchler, Avrigeanu-2014), Numerov and Johnson log-derivative solvers for the
//! radial Schrodinger equation, and routines to compute transmission coefficients
//! $T_{\ell j}(E)$ for use in Hauser-Feshbach statistical model calculations.

#![warn(missing_docs)]

pub mod cc_transmission;
pub mod deformation;
pub mod johnson_logderiv;
pub mod numerov;
pub mod omp;
pub mod potential;
pub mod transmission;

pub use cc_transmission::compute_transmission_auto;
pub use deformation::{CcChannel, CoupledChannelSystem, DeformedKoningDelaroche};
pub use omp::{
    Avrigeanu2014, CustomOmp, KdParameters, KoningDelaroche, KoningDelarocheLocal, McFaddenSatchler,
};
pub use potential::{coulomb_potential, spin_orbit_factor, woods_saxon, woods_saxon_deriv};
pub use transmission::compute_transmission_coeffs;
