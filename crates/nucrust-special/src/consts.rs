//! Shared numerical constants for special function computations.
//!
//! These convergence and guard parameters are used by the scalar
//! ([`crate::coulomb`]), continued-fraction ([`crate::lentz`]), and SIMD
//! ([`crate::simd_batch`]) implementations and must stay in sync.

/// Maximum iterations for continued fractions.
pub(crate) const MAX_CF_ITER: u32 = 20_000;

/// Convergence threshold for continued fractions.
pub(crate) const CF_EPS: f64 = 1e-15;

/// Zero-guard value for the modified Lentz method.
pub(crate) const CF_ZERO_GUARD: f64 = 1e-50;

/// Threshold below which power series is used for F (Steed used above).
pub(crate) const RHO_SMALL: f64 = 0.5;
