//! GPU-accelerated nuclear reaction rate calculation framework.
//!
//! Re-exports all nucrust sub-crates and provides compute backends
//! (`CpuBackend`, and `GpuBackend` via `nucrust-gpu`).

#![warn(missing_docs)]

pub mod cpu_backend;

pub use cpu_backend::CpuBackend;
pub use nucrust_astro;
pub use nucrust_core;
pub use nucrust_data;
pub use nucrust_hf;
pub use nucrust_optical;
pub use nucrust_rmatrix;
pub use nucrust_special;
