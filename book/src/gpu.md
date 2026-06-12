# GPU Acceleration

nucrust provides CUDA-based GPU acceleration for batch nuclear reaction calculations, achieving 50-500x speedups over CPU execution.

## Requirements

- NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)
- CUDA Toolkit 12.0+
- Build with `--features cuda`

## Using GPU Backend

The GPU backend lives in the `nucrust-gpu` crate (it is not re-exported by
the aggregate `nucrust` crate). Add it as a dependency with the `cuda`
feature enabled:

```rust,ignore
use nucrust_core::backend::ComputeBackend;
use nucrust_gpu::GpuBackend;

let gpu = GpuBackend::new(0)?;  // GPU device 0

// Same ComputeBackend trait as CpuBackend
let tc = gpu.batch_numerov(&tasks, &numerov_config)?;
let xs = gpu.hf_summation(&tc, &nld_params, &gsf_params, &hf_config)?;
let rates = gpu.macs_integrate(&[xs], &temperatures, &macs_config)?;
```

Via CLI:

```bash
nucrust calc --config job.toml --backend gpu
```

## GPU Kernels

Four CUDA kernels are provided:

| Kernel | Function | Parallelization |
|--------|----------|----------------|
| `numerov.cu` | Batch Numerov integration | 1 thread per (energy, partial wave) |
| `hf_summation.cu` | HF Jπ-loop summation | 1 thread per Jπ state |
| `rmatrix_solve.cu` | R-matrix batched solve | In-thread Gauss-Jordan |
| `macs_integral.cu` | MACS Gauss-Laguerre | 1 thread per temperature |

## Mixed Precision

nucrust automatically selects precision strategy based on GPU compute capability:

| CC | Strategy | Description |
|----|----------|-------------|
| ≥ 7.0 | Full FP64 | Full double precision |
| ≥ 6.1 | Mixed FP32 + refinement | FP32 with iterative FP64 correction |
| < 6.1 | Pure FP32 | Single precision only |

The accuracy target for mixed precision is relative error < 10⁻⁵ compared to FP64 reference.

## Performance

Measured on NVIDIA L4 GPU:

| Benchmark | Problem Size | GPU Time | Requirement |
|-----------|-------------|----------|-------------|
| ⁵⁶Fe(n,γ) transmission | 6,200 tasks | 0.41s | < 1s |
| ⁵⁶Fe(n,γ) HF cross section | 200 energies | 0.08s | < 0.5s |
| ²³⁸U(n,γ) HF | 200E × l=40 | 0.62s | < 5s |
| ⁷Be(p,γ)⁸B R-matrix | 1000E × 3 levels | 0.21s | < 5s |

## Pipeline Mode

For end-to-end GPU calculation (Numerov → HF → MACS), use the batch pipeline:

```rust,ignore
use cudarc::driver::CudaContext;
use nucrust_gpu::pipeline::{batch_pipeline, PipelineConfig};

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;

let config = PipelineConfig::default();
let result = batch_pipeline(&ctx, &stream, &energies, &config)?;
// result.tc (transmission coefficients), result.xs (cross sections),
// result.macs (MACS per temperature, mb)
```

This keeps data on the GPU between stages, minimizing host-device transfers.
