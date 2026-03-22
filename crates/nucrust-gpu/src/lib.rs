//! GPU backend for nucrust using CUDA via cudarc.
//!
//! Requires the `cuda` feature flag and a CUDA-capable GPU.
//!
//! Enable with: `cargo build -p nucrust-gpu --features cuda`

#[cfg(feature = "cuda")]
pub mod backend;
#[cfg(feature = "cuda")]
mod benchmark;
#[cfg(feature = "cuda")]
pub mod binning;
#[cfg(feature = "cuda")]
pub mod coulomb_device;
#[cfg(feature = "cuda")]
pub mod coulomb_hybrid;
#[cfg(feature = "cuda")]
pub mod coulomb_table;
#[cfg(feature = "cuda")]
pub mod hf_kernel;
#[cfg(feature = "cuda")]
pub mod kernel_cache;
#[cfg(feature = "cuda")]
pub mod macs_kernel;
#[cfg(feature = "cuda")]
pub mod memory;
#[cfg(feature = "cuda")]
pub mod multi_stream;
#[cfg(feature = "cuda")]
pub mod numerov;
#[cfg(feature = "cuda")]
pub mod pipeline;
#[cfg(feature = "cuda")]
pub mod precision;
#[cfg(feature = "cuda")]
pub mod rmatrix_kernel;

#[cfg(feature = "cuda")]
pub use backend::GpuBackend;
