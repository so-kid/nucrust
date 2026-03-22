//! Multi-stream asynchronous execution (T-3A.15, §10.1).
//!
//! Enables concurrent kernel execution on independent tasks using
//! multiple CUDA streams. Each stream can execute a pipeline stage
//! while another stage runs on a different stream.

use cudarc::driver::{CudaContext, CudaStream};
use nucrust_core::CoreError;
use std::sync::Arc;

/// Multi-stream executor for GPU pipelines.
pub struct MultiStreamExecutor {
    ctx: Arc<CudaContext>,
    streams: Vec<Arc<CudaStream>>,
}

impl MultiStreamExecutor {
    /// Create a multi-stream executor with the given number of streams.
    pub fn new(ctx: &Arc<CudaContext>, n_streams: usize) -> Result<Self, CoreError> {
        let mut streams = Vec::with_capacity(n_streams);
        for _ in 0..n_streams {
            let stream = ctx.new_stream().map_err(|_| CoreError::InvalidParameter {
                name: "n_streams",
                value: n_streams as f64,
                reason: "failed to create CUDA stream",
            })?;
            streams.push(stream);
        }

        Ok(Self {
            ctx: Arc::clone(ctx),
            streams,
        })
    }

    /// Get the number of streams.
    pub fn n_streams(&self) -> usize {
        self.streams.len()
    }

    /// Get a reference to a specific stream.
    pub fn stream(&self, idx: usize) -> &Arc<CudaStream> {
        &self.streams[idx % self.streams.len()]
    }

    /// Get the context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Synchronize all streams.
    pub fn synchronize_all(&self) -> Result<(), CoreError> {
        for stream in &self.streams {
            stream
                .synchronize()
                .map_err(|_| CoreError::NumericalOverflow {
                    context: "multi-stream synchronize",
                })?;
        }
        Ok(())
    }

    /// Fork a new stream from the context for independent work.
    pub fn fork_stream(&self) -> Result<Arc<CudaStream>, CoreError> {
        self.ctx
            .new_stream()
            .map_err(|_| CoreError::InvalidParameter {
                name: "fork_stream",
                value: 0.0,
                reason: "failed to fork CUDA stream",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_multi_stream() {
        let ctx = CudaContext::new(0).unwrap();
        let executor = MultiStreamExecutor::new(&ctx, 4).unwrap();
        assert_eq!(executor.n_streams(), 4);
    }

    #[test]
    fn synchronize_all_streams() {
        let ctx = CudaContext::new(0).unwrap();
        let executor = MultiStreamExecutor::new(&ctx, 2).unwrap();
        executor.synchronize_all().unwrap();
    }

    #[test]
    fn fork_additional_stream() {
        let ctx = CudaContext::new(0).unwrap();
        let executor = MultiStreamExecutor::new(&ctx, 1).unwrap();
        let _forked = executor.fork_stream().unwrap();
    }
}
