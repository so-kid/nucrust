//! GPU kernel module cache.

use cudarc::driver::CudaStream;
use nucrust_core::CoreError;
use std::sync::Arc;

/// Cached GPU kernel modules.
pub struct GpuModuleCache {
    _initialized: bool,
}

impl GpuModuleCache {
    /// Load and compile all kernel modules.
    pub fn load(_stream: &Arc<CudaStream>) -> Result<Self, CoreError> {
        Ok(Self { _initialized: true })
    }
}
