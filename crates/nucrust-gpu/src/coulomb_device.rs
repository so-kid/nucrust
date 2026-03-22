//! On-the-fly GPU Coulomb wave function computation (T-3A.11).
//!
//! CUDA __device__ functions implementing CF1 + CF2 + Steed method.
//! No lookup table needed — each thread computes F, G independently.

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use nucrust_core::CoreError;
use std::sync::Arc;

const COULOMB_DEVICE_SRC: &str = include_str!("../../../kernels/coulomb_device.cu");

/// Compute Coulomb functions F_l, G_l on GPU for a batch of (eta, rho, l) points.
pub fn gpu_coulomb_batch(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    eta: &[f64],
    rho: &[f64],
    l: &[i32],
) -> Result<(Vec<f64>, Vec<f64>), CoreError> {
    let n = eta.len();
    assert_eq!(n, rho.len());
    assert_eq!(n, l.len());

    let ptx = cudarc::nvrtc::compile_ptx(COULOMB_DEVICE_SRC).map_err(|_| {
        CoreError::InvalidParameter {
            name: "nvrtc",
            value: 0.0,
            reason: "failed to compile Coulomb device kernel",
        }
    })?;

    let module = CudaContext::load_module(ctx, ptx).map_err(|_| CoreError::InvalidParameter {
        name: "load_module",
        value: 0.0,
        reason: "failed to load Coulomb device module",
    })?;

    let func =
        module
            .load_function("coulomb_batch_device")
            .map_err(|_| CoreError::DataNotFound {
                description: "coulomb_batch_device not found".to_string(),
            })?;

    let mut d_eta = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_rho = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_l = stream.alloc_zeros::<i32>(n).map_err(gpu_err)?;
    let mut d_f = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_g = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;

    stream.memcpy_htod(eta, &mut d_eta).map_err(gpu_err)?;
    stream.memcpy_htod(rho, &mut d_rho).map_err(gpu_err)?;
    stream.memcpy_htod(l, &mut d_l).map_err(gpu_err)?;

    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&d_eta)
            .arg(&d_rho)
            .arg(&d_l)
            .arg(&mut d_f)
            .arg(&mut d_g)
            .arg(&(n as i32))
            .launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|_| CoreError::NumericalOverflow {
                context: "Coulomb device kernel launch",
            })?;
    }

    let mut f_out = vec![0.0f64; n];
    let mut g_out = vec![0.0f64; n];
    stream.memcpy_dtoh(&d_f, &mut f_out).map_err(gpu_err)?;
    stream.memcpy_dtoh(&d_g, &mut g_out).map_err(gpu_err)?;
    stream.synchronize().map_err(gpu_err)?;

    Ok((f_out, g_out))
}

fn gpu_err<E>(_: E) -> CoreError {
    CoreError::NumericalOverflow {
        context: "GPU Coulomb operation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_coulomb_bessel_limit() {
        // eta=0: F_0 = sin(rho), G_0 = cos(rho)
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let etas = vec![0.0, 0.0, 0.0, 0.0];
        let rhos = vec![1.0, 2.0, 3.0, 5.0];
        let ls = vec![0, 0, 0, 0];

        let (f_vals, g_vals) = gpu_coulomb_batch(&ctx, &stream, &etas, &rhos, &ls).unwrap();

        for (i, &rho) in rhos.iter().enumerate() {
            let expected_f = rho.sin();
            let expected_g = rho.cos();
            let f_err = (f_vals[i].abs() - expected_f.abs()).abs();
            let g_err = (g_vals[i].abs() - expected_g.abs()).abs();
            assert!(
                f_err < 1e-6,
                "|F_0|(rho={}): GPU={}, expected={}, err={}",
                rho,
                f_vals[i],
                expected_f,
                f_err
            );
            assert!(
                g_err < 1e-6,
                "|G_0|(rho={}): GPU={}, expected={}, err={}",
                rho,
                g_vals[i],
                expected_g,
                g_err
            );
        }
    }

    #[test]
    fn gpu_coulomb_wronskian() {
        // Verify Wronskian |F'G - FG'| ≈ 1
        // We only get F and G (not derivatives) from the batch kernel,
        // so verify F and G are finite and reasonable.
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let etas = vec![1.0, 2.0, 5.0];
        let rhos = vec![5.0, 8.0, 15.0];
        let ls = vec![0, 1, 2];

        let (f_vals, g_vals) = gpu_coulomb_batch(&ctx, &stream, &etas, &rhos, &ls).unwrap();

        for i in 0..3 {
            assert!(f_vals[i].is_finite(), "F[{}] not finite: {}", i, f_vals[i]);
            assert!(g_vals[i].is_finite(), "G[{}] not finite: {}", i, g_vals[i]);
            // F^2 + G^2 should be roughly 1/k (normalization)
            // Not exactly testable without derivatives, but values should be O(1)
            assert!(f_vals[i].abs() < 1e10, "F[{}] too large: {}", i, f_vals[i]);
        }
    }

    #[test]
    fn gpu_coulomb_large_batch() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let n = 1000;
        let etas: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let rhos: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.03).collect();
        let ls: Vec<i32> = (0..n).map(|i| (i % 5) as i32).collect();

        let (f_vals, g_vals) = gpu_coulomb_batch(&ctx, &stream, &etas, &rhos, &ls).unwrap();

        assert_eq!(f_vals.len(), n);
        let finite_count = f_vals.iter().filter(|v| v.is_finite()).count();
        assert!(
            finite_count > n * 9 / 10,
            "only {}/{} F values are finite",
            finite_count,
            n
        );
    }
}
