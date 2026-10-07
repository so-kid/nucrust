//! Mixed precision strategy (T-3A.4, §10.5).
//!
//! Detects FP64/FP32 throughput ratio on the GPU and selects precision
//! strategy accordingly. For GPUs with poor FP64 (e.g., consumer and
//! workstation cards with a 1:32 or 1:64 ratio), uses FP32 with iterative
//! refinement.
//!
//! The compute-capability → strategy mapping ([`strategy_for_cc`]) is pure
//! and available without the `cuda` feature; [`detect_precision`] queries
//! the device and requires it.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use nucrust_core::CoreError;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Precision strategy for GPU computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionStrategy {
    /// Full FP64 computation (default for HPC GPUs).
    Full64,
    /// FP32 computation with iterative refinement for critical paths.
    Mixed32Refine,
    /// Pure FP32 (fastest, lowest accuracy).
    Pure32,
}

/// Select the precision strategy for a CUDA compute capability.
///
/// Only architectures with a native FP64:FP32 throughput ratio of 1:2 get
/// [`PrecisionStrategy::Full64`]; all others (1:32 or 1:64, including
/// unknown future capabilities) use [`PrecisionStrategy::Mixed32Refine`],
/// which still meets [`MIXED_PRECISION_TOL`].
pub fn strategy_for_cc(cc_major: i32, cc_minor: i32) -> PrecisionStrategy {
    match (cc_major, cc_minor) {
        // HPC GPUs with FP64:FP32 = 1:2
        (6, 0) => PrecisionStrategy::Full64,  // P100 (Pascal)
        (7, 0) => PrecisionStrategy::Full64,  // V100 (Volta)
        (8, 0) => PrecisionStrategy::Full64,  // A100 / A30 (Ampere)
        (9, 0) => PrecisionStrategy::Full64,  // H100 / H200 (Hopper)
        (10, 0) => PrecisionStrategy::Full64, // B200 (Blackwell)
        // Everything else has weak FP64, e.g.:
        // 6.1 Pascal consumer 1:32, 7.2 Xavier 1:32, 7.5 Turing 1:32,
        // 8.6 Ampere A40 / RTX 30xx 1:64, 8.7 Orin 1:64,
        // 8.9 Ada L4 / L40 / RTX 40xx 1:64, 12.x Blackwell RTX 1:64
        _ => PrecisionStrategy::Mixed32Refine,
    }
}

/// Detect the optimal precision strategy for the given GPU.
///
/// Queries the compute capability and maps it via [`strategy_for_cc`].
#[cfg(feature = "cuda")]
pub fn detect_precision(ctx: &Arc<CudaContext>) -> Result<PrecisionStrategy, CoreError> {
    // Query compute capability
    let cc_major = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .map_err(|_| CoreError::InvalidParameter {
            name: "gpu",
            value: 0.0,
            reason: "failed to query compute capability",
        })?;

    let cc_minor = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .map_err(|_| CoreError::InvalidParameter {
            name: "gpu",
            value: 0.0,
            reason: "failed to query compute capability",
        })?;

    let name = ctx.name().unwrap_or_else(|_| "Unknown".to_string());

    let strategy = strategy_for_cc(cc_major, cc_minor);

    eprintln!(
        "GPU: {} (CC {}.{}), precision strategy: {:?}",
        name, cc_major, cc_minor, strategy
    );

    Ok(strategy)
}

/// Maximum acceptable relative error for mixed-precision computations (10^{-5}).
pub const MIXED_PRECISION_TOL: f64 = 1e-5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hpc_gpus_use_full64() {
        for cc in [(6, 0), (7, 0), (8, 0), (9, 0), (10, 0)] {
            assert_eq!(
                strategy_for_cc(cc.0, cc.1),
                PrecisionStrategy::Full64,
                "{cc:?}"
            );
        }
    }

    #[test]
    fn weak_fp64_gpus_use_mixed() {
        for cc in [
            (5, 2),
            (6, 1),
            (7, 2),
            (7, 5),
            (8, 6),
            (8, 7),
            (8, 9),
            (12, 0),
            (99, 0),
        ] {
            assert_eq!(
                strategy_for_cc(cc.0, cc.1),
                PrecisionStrategy::Mixed32Refine,
                "{cc:?}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn detect_precision_runs() {
        let ctx = CudaContext::new(0).unwrap();
        let strategy = detect_precision(&ctx).unwrap();
        // Should be one of the valid strategies
        assert!(
            strategy == PrecisionStrategy::Full64
                || strategy == PrecisionStrategy::Mixed32Refine
                || strategy == PrecisionStrategy::Pure32
        );
    }
}
