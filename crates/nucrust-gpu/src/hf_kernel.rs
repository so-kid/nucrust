//! GPU-accelerated Hauser-Feshbach J-pi summation.

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use nucrust_core::{CoreError, CrossSection, TransmissionCoeffs};
use std::sync::Arc;

const HF_KERNEL_SRC: &str = include_str!("../../../kernels/hf_summation.cu");

/// Parameters for GPU HF summation.
pub struct GpuHfParams {
    /// Maximum 2J value for spin summation.
    pub two_j_max: i32,
    /// Reaction Q-value (MeV).
    pub q_value: f64,
    /// Level density parameter a (MeV^{-1}).
    pub nld_a: f64,
    /// Nuclear temperature T (MeV).
    pub nld_t: f64,
    /// GDR centroid energy (MeV).
    pub gsf_e_gdr: f64,
    /// GDR width (MeV).
    pub gsf_gamma_gdr: f64,
    /// GDR peak cross section (mb).
    pub gsf_sigma_gdr: f64,
    /// hbar^2 / (2 * mu) in MeV*fm^2.
    pub hbar2_over_2mu: f64,
}

/// Run HF summation on GPU.
///
/// Each energy point is one CUDA thread. The J-pi loop runs within each thread.
pub fn gpu_hf_summation(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    tc: &TransmissionCoeffs,
    params: &GpuHfParams,
) -> Result<CrossSection, CoreError> {
    let n_e = tc.energy.len();
    let n_l = (tc.l_max + 1) as i32;

    let ptx =
        cudarc::nvrtc::compile_ptx(HF_KERNEL_SRC).map_err(|_| CoreError::InvalidParameter {
            name: "nvrtc",
            value: 0.0,
            reason: "failed to compile HF kernel",
        })?;

    let module = CudaContext::load_module(ctx, ptx).map_err(|_| CoreError::InvalidParameter {
        name: "load_module",
        value: 0.0,
        reason: "failed to load HF module",
    })?;

    let func = module
        .load_function("hf_summation")
        .map_err(|_| CoreError::DataNotFound {
            description: "hf_summation function not found".to_string(),
        })?;

    // Upload data
    let mut d_energies = stream.alloc_zeros::<f64>(n_e).map_err(gpu_err)?;
    let mut d_tc = stream.alloc_zeros::<f64>(tc.data.len()).map_err(gpu_err)?;
    let mut d_sigma = stream.alloc_zeros::<f64>(n_e).map_err(gpu_err)?;
    let mut d_sigma_gamma = stream.alloc_zeros::<f64>(n_e).map_err(gpu_err)?;

    stream
        .memcpy_htod(tc.energy.as_slice(), &mut d_energies)
        .map_err(gpu_err)?;
    stream.memcpy_htod(&tc.data, &mut d_tc).map_err(gpu_err)?;

    let block_size = 256u32;
    let grid_size = (n_e as u32).div_ceil(block_size);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&d_energies)
            .arg(&d_tc)
            .arg(&(n_e as i32))
            .arg(&n_l)
            .arg(&params.two_j_max)
            .arg(&params.q_value)
            .arg(&params.nld_a)
            .arg(&params.nld_t)
            .arg(&params.gsf_e_gdr)
            .arg(&params.gsf_gamma_gdr)
            .arg(&params.gsf_sigma_gdr)
            .arg(&params.hbar2_over_2mu)
            .arg(&mut d_sigma)
            .arg(&mut d_sigma_gamma)
            .launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|_| CoreError::NumericalOverflow {
                context: "HF kernel launch",
            })?;
    }

    let mut sigma_total = vec![0.0f64; n_e];
    let mut sigma_reaction = vec![0.0f64; n_e];
    stream
        .memcpy_dtoh(&d_sigma, &mut sigma_total)
        .map_err(gpu_err)?;
    stream
        .memcpy_dtoh(&d_sigma_gamma, &mut sigma_reaction)
        .map_err(gpu_err)?;
    stream.synchronize().map_err(gpu_err)?;

    Ok(CrossSection {
        energy: tc.energy.clone(),
        sigma_total,
        sigma_elastic: vec![0.0; n_e],
        sigma_reaction,
        partial: vec![],
    })
}

fn gpu_err<E>(_: E) -> CoreError {
    CoreError::NumericalOverflow {
        context: "GPU operation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::EnergyGrid;

    #[test]
    fn gpu_hf_runs() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let energies = EnergyGrid::logarithmic(0.5, 5.0, 10).unwrap();
        let n_e = energies.len();
        let l_max = 5u32;

        // Synthetic transmission coefficients
        let n_l = (l_max + 1) as usize;
        let mut tc_data = vec![0.0; n_l * 2 * n_e];
        // Set some non-zero values for l=0, j=0.5 (j_idx=1)
        // Explicit [l][j_idx][e_idx] index arithmetic documents the layout.
        #[allow(clippy::identity_op, clippy::erasing_op)]
        for e_idx in 0..n_e {
            tc_data[0 * 2 * n_e + 1 * n_e + e_idx] = 0.8; // l=0, j=0.5
            tc_data[1 * 2 * n_e + 1 * n_e + e_idx] = 0.5; // l=1, j=1.5
        }

        let tc = TransmissionCoeffs {
            energy: energies,
            l_max,
            data: tc_data,
        };

        let a_target = 56.0_f64;
        let mu = nucrust_core::units::reduced_mass(1.008664, a_target);
        let hbar2_2mu =
            nucrust_core::units::HBAR_C.powi(2) / (2.0 * mu * nucrust_core::units::AMU_MEV);

        let params = GpuHfParams {
            two_j_max: 10,
            q_value: 7.646,
            nld_a: 6.21,
            nld_t: 0.88,
            gsf_e_gdr: 16.36,
            gsf_gamma_gdr: 4.58,
            gsf_sigma_gdr: 136.0,
            hbar2_over_2mu: hbar2_2mu,
        };

        let xs = gpu_hf_summation(&ctx, &stream, &tc, &params).unwrap();

        assert_eq!(xs.sigma_total.len(), n_e);
        for (i, &sigma) in xs.sigma_total.iter().enumerate() {
            assert!(
                sigma >= 0.0 && sigma.is_finite(),
                "sigma[{}] = {} invalid",
                i,
                sigma
            );
        }
        // At least some cross sections should be non-zero
        let max_sigma = xs.sigma_total.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_sigma > 0.0, "all sigma_cn are zero");
    }
}
