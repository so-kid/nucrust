//! Parsers and I/O utilities for nuclear data formats.
//!
//! This crate provides readers for the standard nuclear data libraries used
//! in reaction-rate calculations:
//!
//! - [`ripl3`] -- RIPL-3 (Reference Input Parameter Library) parsers for
//!   discrete levels, masses, optical model parameters, level densities,
//!   gamma-ray strength functions, resonance parameters, shell corrections,
//!   and fission barriers.
//! - [`reaclib`] -- REACLIB thermonuclear reaction-rate format: parsing,
//!   evaluation, and 7-parameter fitting.
//! - [`config`] -- TOML job-configuration deserialization.
//! - [`fixed_field`] -- Low-level Fortran fixed-width field parser utilities.
//! - `hdf5_io` -- HDF5 read/write for cross sections and reaction rates
//!   (requires the `hdf5_io` feature).

#![warn(missing_docs)]

/// TOML job-configuration types and parser.
pub mod config;
/// Fortran fixed-width field parser utilities.
pub mod fixed_field;
/// HDF5 I/O for cross sections and reaction rates.
#[cfg(feature = "hdf5_io")]
pub mod hdf5_io;
/// REACLIB thermonuclear reaction-rate format support.
pub mod reaclib;
/// RIPL-3 nuclear data parsers.
pub mod ripl3;
