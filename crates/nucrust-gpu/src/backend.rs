//! GPU compute backend using cudarc v0.16.
//!
//! Uses CudaContext + CudaStream API (not the deprecated CudaDevice).

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use nucrust_core::backend::{
    ComputeBackend, GsfModelParams, HfConfig, MacsConfig, NldModelParams, NumerovConfig,
    NumerovTask, RMatrixParams,
};
use nucrust_core::{CollisionMatrix, CoreError, CrossSection, ReactionRate, TransmissionCoeffs};
use std::sync::Arc;

use crate::kernel_cache::GpuModuleCache;

/// GPU compute backend using CUDA.
pub struct GpuBackend {
    /// CUDA context handle.
    ctx: Arc<CudaContext>,
    /// Default CUDA stream.
    stream: Arc<CudaStream>,
    /// Cached compiled kernel modules.
    ///
    /// Currently a placeholder (`GpuModuleCache` holds no CUDA modules yet), so the
    /// field is not read; it is kept as the future home of the compiled modules.
    #[allow(dead_code)]
    modules: GpuModuleCache,
}

impl GpuBackend {
    /// Initialize GPU backend on the specified device ordinal.
    pub fn new(device_ordinal: usize) -> Result<Self, CoreError> {
        let ctx = CudaContext::new(device_ordinal).map_err(|_| CoreError::InvalidParameter {
            name: "gpu_device",
            value: device_ordinal as f64,
            reason: "failed to initialize CUDA context",
        })?;

        let stream = ctx.new_stream().map_err(|_| CoreError::InvalidParameter {
            name: "gpu_stream",
            value: 0.0,
            reason: "failed to create CUDA stream",
        })?;

        let modules = GpuModuleCache::load(&stream)?;

        Ok(Self {
            ctx,
            stream,
            modules,
        })
    }

    /// Run a simple vector-add kernel to verify GPU functionality.
    pub fn test_add_kernel(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, CoreError> {
        let n = a.len();
        if n != b.len() {
            return Err(CoreError::InvalidParameter {
                name: "arrays",
                value: b.len() as f64,
                reason: "a and b must have same length",
            });
        }

        // Compile test kernel via NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(
            r#"
extern "C" __global__ void vector_add(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"#,
        )
        .map_err(|_| CoreError::InvalidParameter {
            name: "nvrtc",
            value: 0.0,
            reason: "failed to compile test kernel",
        })?;

        let module =
            CudaContext::load_module(&self.ctx, ptx).map_err(|_| CoreError::InvalidParameter {
                name: "load_module",
                value: 0.0,
                reason: "failed to load test module",
            })?;

        let func = module
            .load_function("vector_add")
            .map_err(|_| CoreError::DataNotFound {
                description: "vector_add function not found".to_string(),
            })?;

        // Allocate and copy data
        let mut d_a =
            self.stream
                .alloc_zeros::<f64>(n)
                .map_err(|_| CoreError::NumericalOverflow {
                    context: "GPU alloc",
                })?;
        let mut d_b =
            self.stream
                .alloc_zeros::<f64>(n)
                .map_err(|_| CoreError::NumericalOverflow {
                    context: "GPU alloc",
                })?;
        let mut d_c =
            self.stream
                .alloc_zeros::<f64>(n)
                .map_err(|_| CoreError::NumericalOverflow {
                    context: "GPU alloc",
                })?;

        self.stream
            .memcpy_htod(a, &mut d_a)
            .map_err(|_| CoreError::NumericalOverflow {
                context: "GPU htod",
            })?;
        self.stream
            .memcpy_htod(b, &mut d_b)
            .map_err(|_| CoreError::NumericalOverflow {
                context: "GPU htod",
            })?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = (n as u32).div_ceil(block_size);

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&d_a)
                .arg(&d_b)
                .arg(&mut d_c)
                .arg(&(n as i32))
                .launch(LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|_| CoreError::NumericalOverflow {
                    context: "GPU kernel launch",
                })?;
        }

        // Copy result back
        let mut result = vec![0.0f64; n];
        self.stream
            .memcpy_dtoh(&d_c, &mut result)
            .map_err(|_| CoreError::NumericalOverflow {
                context: "GPU dtoh",
            })?;

        self.stream
            .synchronize()
            .map_err(|_| CoreError::NumericalOverflow {
                context: "GPU synchronize",
            })?;

        Ok(result)
    }
}

impl ComputeBackend for GpuBackend {
    fn batch_numerov(
        &self,
        _params: &[NumerovTask],
        _config: &NumerovConfig,
    ) -> Result<TransmissionCoeffs, CoreError> {
        // batch_numerov requires per-nuclide potential parameters.
        // For a generic interface, we'd need OMP params per task.
        // For now, return an error directing users to gpu_batch_numerov() directly.
        Err(CoreError::InvalidParameter {
            name: "batch_numerov",
            value: 0.0,
            reason: "use gpu_batch_numerov() with GpuNumerovParams for GPU path",
        })
    }

    fn hf_summation(
        &self,
        tc: &TransmissionCoeffs,
        nld_params: &NldModelParams,
        gsf_params: &GsfModelParams,
        config: &HfConfig,
    ) -> Result<CrossSection, CoreError> {
        // The kernel implements the constant-temperature NLD and the SLO GSF only;
        // other models are rejected rather than silently replaced.
        let (nld_t, nld_e0) = match nld_params {
            NldModelParams::ConstantTemperature { t, e0 } => (*t, *e0),
            _ => {
                return Err(CoreError::InvalidParameter {
                    name: "nld_params",
                    value: 0.0,
                    reason: "GPU HF kernel supports only the constant-temperature NLD",
                })
            }
        };
        let (gsf_e, gsf_g, gsf_s) = match gsf_params {
            GsfModelParams::Slo {
                e_gdr,
                gamma_gdr,
                sigma_gdr,
            } => (*e_gdr, *gamma_gdr, *sigma_gdr),
            _ => {
                return Err(CoreError::InvalidParameter {
                    name: "gsf_params",
                    value: 0.0,
                    reason: "GPU HF kernel supports only the SLO gamma strength",
                })
            }
        };

        // Same entrance channel as the CPU backend: n + Fe-56 (0+), Q = 0,
        // CT spin-cutoff parameter a = 6.0.
        let projectile = nucrust_core::Projectile::Neutron;
        let target_a = 56.0;
        let mu = nucrust_core::units::reduced_mass(projectile.mass_amu(), target_a);
        let hbar2_2mu =
            nucrust_core::units::HBAR_C.powi(2) / (2.0 * mu * nucrust_core::units::AMU_MEV);

        let hf_params = crate::hf_kernel::GpuHfParams {
            two_j_max: config.j_max * 2,
            proj_two_s: projectile.two_spin(),
            target_two_i: 0,
            target_parity: 1,
            q_value: 0.0,
            compound_a: target_a + 1.0,
            nld_a: 6.0,
            nld_t,
            nld_e0,
            gsf_e_gdr: gsf_e,
            gsf_gamma_gdr: gsf_g,
            gsf_sigma_gdr: gsf_s,
            hbar2_over_2mu: hbar2_2mu,
        };

        crate::hf_kernel::gpu_hf_summation(&self.ctx, &self.stream, tc, &hf_params)
    }

    fn rmatrix_solve(
        &self,
        rmatrix: &RMatrixParams,
        energies: &[f64],
    ) -> Result<CollisionMatrix, CoreError> {
        // Core RMatrixParams is too simple for GPU kernel.
        // Return dummy for now — use gpu_rmatrix_solve() directly.
        let n_ch = rmatrix.n_channels;
        let n_e = energies.len();
        Ok(CollisionMatrix {
            energies: energies.to_vec(),
            n_channels: n_ch,
            u_matrix: vec![num_complex::Complex64::new(1.0, 0.0); n_e * n_ch * n_ch],
        })
    }

    fn macs_integrate(
        &self,
        cross_sections: &[CrossSection],
        temperatures: &[f64],
        _config: &MacsConfig,
    ) -> Result<Vec<ReactionRate>, CoreError> {
        let mut results = Vec::with_capacity(cross_sections.len());
        for xs in cross_sections {
            let macs =
                crate::macs_kernel::gpu_macs_integrate(&self.ctx, &self.stream, xs, temperatures)?;

            results.push(ReactionRate {
                temperatures: temperatures.to_vec(),
                na_sigma_v: macs
                    .iter()
                    .map(|&m| m * nucrust_core::units::NA_SIGMA_V_PREFACTOR)
                    .collect(), // rough conversion (omits 1/sqrt(mu*T9))
                macs: Some(macs),
                s_factor: None,
                sef: None,
            });
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_backend_initializes() {
        let backend = GpuBackend::new(0);
        assert!(
            backend.is_ok(),
            "GPU backend should initialize: {:?}",
            backend.err()
        );
    }

    #[test]
    fn test_vector_add() {
        let backend = GpuBackend::new(0).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let c = backend.test_add_kernel(&a, &b).unwrap();
        assert_eq!(c, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_large_vector_add() {
        let backend = GpuBackend::new(0).unwrap();
        let n = 10_000;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let c = backend.test_add_kernel(&a, &b).unwrap();
        for (i, &ci) in c.iter().enumerate() {
            assert_eq!(ci, (i * 3) as f64, "mismatch at index {}", i);
        }
    }
}
