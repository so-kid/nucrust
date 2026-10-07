//! GPU-accelerated R-matrix batched solve.
//!
//! Each CUDA thread computes the R-matrix cross section for one energy point
//! using in-thread Gauss-Jordan matrix inversion (efficient for N <= 10 channels).

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use nucrust_core::CoreError;
use std::sync::Arc;

const RMATRIX_KERNEL_SRC: &str = include_str!("../../../kernels/rmatrix_solve.cu");

/// Parameters for GPU R-matrix computation (max 10 channels).
pub struct GpuRMatrixParams {
    /// Reduced width amplitudes gamma_{lambda,c}, flattened [n_levels * n_ch].
    pub gamma_widths: Vec<f64>,
    /// Pole energies E_lambda (MeV), length n_levels.
    pub level_energies: Vec<f64>,
    /// Channel penetrabilities P_c, length n_ch.
    pub penetrabilities: Vec<f64>,
    /// Channel shift functions S_c, length n_ch.
    pub shift_functions: Vec<f64>,
    /// Number of channels (max 10).
    pub n_channels: usize,
    /// Number of R-matrix levels.
    pub n_levels: usize,
    /// Entrance channel wave number squared k^2 (fm^{-2}).
    pub k_sq: f64,
}

/// Run batched R-matrix solve on GPU.
///
/// Returns reaction cross section sigma_r(E) for each energy point.
pub fn gpu_rmatrix_solve(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    energies: &[f64],
    params: &GpuRMatrixParams,
) -> Result<Vec<f64>, CoreError> {
    let n_e = energies.len();
    let n_ch = params.n_channels;
    let n_levels = params.n_levels;

    if n_ch > 10 {
        return Err(CoreError::InvalidParameter {
            name: "n_channels",
            value: n_ch as f64,
            reason: "GPU R-matrix kernel supports max 10 channels",
        });
    }

    let ptx = cudarc::nvrtc::compile_ptx(RMATRIX_KERNEL_SRC).map_err(|_| {
        CoreError::InvalidParameter {
            name: "nvrtc",
            value: 0.0,
            reason: "failed to compile R-matrix kernel",
        }
    })?;

    let module = CudaContext::load_module(ctx, ptx).map_err(|_| CoreError::InvalidParameter {
        name: "load_module",
        value: 0.0,
        reason: "failed to load R-matrix module",
    })?;

    let func = module
        .load_function("rmatrix_solve")
        .map_err(|_| CoreError::DataNotFound {
            description: "rmatrix_solve function not found".to_string(),
        })?;

    // Upload data
    let mut d_energies = stream.alloc_zeros::<f64>(n_e).map_err(gpu_err)?;
    let mut d_gamma = stream
        .alloc_zeros::<f64>(n_levels * n_ch)
        .map_err(gpu_err)?;
    let mut d_levels = stream.alloc_zeros::<f64>(n_levels).map_err(gpu_err)?;
    let mut d_pen = stream.alloc_zeros::<f64>(n_ch).map_err(gpu_err)?;
    let mut d_shift = stream.alloc_zeros::<f64>(n_ch).map_err(gpu_err)?;
    let mut d_sigma = stream.alloc_zeros::<f64>(n_e).map_err(gpu_err)?;

    stream
        .memcpy_htod(energies, &mut d_energies)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(&params.gamma_widths, &mut d_gamma)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(&params.level_energies, &mut d_levels)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(&params.penetrabilities, &mut d_pen)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(&params.shift_functions, &mut d_shift)
        .map_err(gpu_err)?;

    let block_size = 256u32;
    let grid_size = (n_e as u32).div_ceil(block_size);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&d_energies)
            .arg(&d_gamma)
            .arg(&d_levels)
            .arg(&d_pen)
            .arg(&d_shift)
            .arg(&(n_ch as i32))
            .arg(&(n_levels as i32))
            .arg(&params.k_sq)
            .arg(&mut d_sigma)
            .arg(&(n_e as i32))
            .launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|_| CoreError::NumericalOverflow {
                context: "R-matrix kernel launch",
            })?;
    }

    let mut sigma = vec![0.0f64; n_e];
    stream.memcpy_dtoh(&d_sigma, &mut sigma).map_err(gpu_err)?;
    stream.synchronize().map_err(gpu_err)?;

    Ok(sigma)
}

fn gpu_err<E>(_: E) -> CoreError {
    CoreError::NumericalOverflow {
        context: "GPU operation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_rmatrix_single_level() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        // Single-level, single-channel Breit-Wigner
        // E_lambda = 0.5 MeV, gamma = 0.1 MeV^{1/2}
        let energies: Vec<f64> = (0..50).map(|i| 0.01 + i as f64 * 0.02).collect();

        let params = GpuRMatrixParams {
            gamma_widths: vec![0.1], // one level, one channel
            level_energies: vec![0.5],
            penetrabilities: vec![0.5],
            shift_functions: vec![0.0],
            n_channels: 1,
            n_levels: 1,
            k_sq: 0.05, // k^2 in fm^{-2}
        };

        let sigma = gpu_rmatrix_solve(&ctx, &stream, &energies, &params).unwrap();
        assert_eq!(sigma.len(), 50);

        // Should show a resonance peak near E = 0.5 MeV
        let mut max_sigma = 0.0_f64;
        for (i, &s) in sigma.iter().enumerate() {
            assert!(s >= 0.0 && s.is_finite(), "sigma[{}] = {} invalid", i, s);
            if s > max_sigma {
                max_sigma = s;
            }
        }

        // Should have a visible resonance peak somewhere in the energy range
        assert!(max_sigma > 0.0, "peak sigma should be > 0");
        // Cross section should vary across energies (resonance structure)
        let min_sigma = sigma.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            max_sigma > min_sigma * 1.1 || min_sigma < 1e-10,
            "cross section should show resonance structure"
        );
    }

    #[test]
    fn gpu_rmatrix_two_channel() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let energies: Vec<f64> = (0..20).map(|i| 0.1 + i as f64 * 0.1).collect();

        let params = GpuRMatrixParams {
            gamma_widths: vec![0.1, 0.05], // one level, two channels
            level_energies: vec![1.0],
            penetrabilities: vec![0.5, 0.3],
            shift_functions: vec![0.0, 0.0],
            n_channels: 2,
            n_levels: 1,
            k_sq: 0.05,
        };

        let sigma = gpu_rmatrix_solve(&ctx, &stream, &energies, &params).unwrap();
        assert_eq!(sigma.len(), 20);
        for &s in &sigma {
            assert!(s >= 0.0 && s.is_finite());
        }
    }
}
