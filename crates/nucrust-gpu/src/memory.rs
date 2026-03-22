//! GPU memory management.

use cudarc::driver::{CudaSlice, CudaStream};
use nucrust_core::CoreError;
use std::sync::Arc;

/// GPU memory pool for efficient sub-allocation.
pub struct GpuMemoryPool {
    stream: Arc<CudaStream>,
    max_bytes: usize,
}

impl GpuMemoryPool {
    /// Create a new memory pool with the given byte budget.
    pub fn new(stream: Arc<CudaStream>, max_bytes: usize) -> Self {
        Self { stream, max_bytes }
    }

    /// Allocate a zero-initialized f64 buffer of `n` elements on GPU.
    pub fn alloc_f64(&self, n: usize) -> Result<CudaSlice<f64>, CoreError> {
        let bytes = n * std::mem::size_of::<f64>();
        if bytes > self.max_bytes {
            return Err(CoreError::InvalidParameter {
                name: "allocation_size",
                value: bytes as f64,
                reason: "exceeds GPU memory pool budget",
            });
        }
        self.stream
            .alloc_zeros::<f64>(n)
            .map_err(|_| CoreError::NumericalOverflow {
                context: "GPU memory allocation",
            })
    }

    /// Return the maximum allocation budget in bytes.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}
