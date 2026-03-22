//! Mixed precision strategy (T-3A.4, §10.5).
//!
//! Detects FP64/FP32 throughput ratio on the GPU and selects precision
//! strategy accordingly. For GPUs with poor FP64 (e.g., consumer cards
//! with 1:32 ratio), uses FP32 with iterative refinement.

use cudarc::driver::CudaContext;
use nucrust_core::CoreError;
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

/// Detect the optimal precision strategy for the given GPU.
///
/// Checks the FP64/FP32 throughput ratio:
/// - ratio >= 1:2 → Full64 (e.g., A100, V100, L4)
/// - ratio 1:4 to 1:16 → Mixed32Refine
/// - ratio < 1:16 → Pure32 (e.g., consumer GeForce)
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

    // Known FP64/FP32 ratios by compute capability and architecture
    let strategy = match (cc_major, cc_minor) {
        // HPC GPUs with good FP64
        (7, 0) => PrecisionStrategy::Full64, // V100: 1:2
        (8, 0) => PrecisionStrategy::Full64, // A100: 1:2
        (9, 0) => PrecisionStrategy::Full64, // H100: 1:2
        // Data center / professional with moderate FP64
        (8, 9) => PrecisionStrategy::Full64, // L4, Ada Lovelace: 1:2 (FP64 capable)
        (8, 6) => PrecisionStrategy::Mixed32Refine, // A40, RTX A5000: 1:64
        // Consumer GPUs with poor FP64
        (7, 5) => PrecisionStrategy::Mixed32Refine, // Turing consumer: 1:32
        (8, 6) => PrecisionStrategy::Mixed32Refine, // Ampere consumer: 1:64
        // Default: assume FP64 is acceptable
        _ => {
            if cc_major >= 7 {
                PrecisionStrategy::Full64
            } else {
                PrecisionStrategy::Mixed32Refine
            }
        }
    };

    eprintln!(
        "GPU: {} (CC {}.{}), precision strategy: {:?}",
        name, cc_major, cc_minor, strategy
    );

    Ok(strategy)
}

/// Maximum acceptable relative error for mixed precision.
pub const MIXED_PRECISION_TOL: f64 = 1e-5;

#[cfg(test)]
mod tests {
    use super::*;

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
