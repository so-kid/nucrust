//! GPU batch pipeline: Numerov → HF → MACS in continuous GPU execution.
//!
//! Minimizes host-device transfers by keeping intermediate results on GPU.

use cudarc::driver::{CudaContext, CudaStream};
use nucrust_core::{CoreError, CrossSection, EnergyGrid, TransmissionCoeffs};
use std::sync::Arc;

use crate::hf_kernel::{gpu_hf_summation, GpuHfParams};
use crate::macs_kernel::gpu_macs_integrate;
use crate::numerov::{gpu_batch_numerov, GpuNumerovParams};

/// Full Numerov->HF->MACS pipeline configuration.
pub struct PipelineConfig {
    /// Numerov integration parameters (potential depths, geometry).
    pub numerov: GpuNumerovParams,
    /// Hauser-Feshbach parameters (J_max, Q-value, NLD/GSF models).
    pub hf: GpuHfParams,
    /// Maximum orbital angular momentum l.
    pub max_l: u32,
    /// Temperature grid for MACS integration (GK).
    pub temperatures: Vec<f64>,
}

/// Result of the full pipeline.
pub struct PipelineResult {
    /// Transmission coefficients
    pub tc: TransmissionCoeffs,
    /// Cross sections
    pub xs: CrossSection,
    /// MACS values (mb) for each temperature
    pub macs: Vec<f64>,
}

/// Execute the full Numerov → HF → MACS pipeline on GPU.
///
/// Intermediate results (transmission coefficients, cross sections) stay
/// on GPU memory where possible. Only the final MACS result is transferred
/// back to the host.
pub fn batch_pipeline(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    energies: &EnergyGrid,
    config: &PipelineConfig,
) -> Result<PipelineResult, CoreError> {
    // Step 1: Batch Numerov → TransmissionCoeffs
    let tc = gpu_batch_numerov(ctx, stream, energies, &config.numerov, config.max_l)?;

    // Step 2: HF summation → CrossSection
    let xs = gpu_hf_summation(ctx, stream, &tc, &config.hf)?;

    // Step 3: MACS integration → rates
    let macs = gpu_macs_integrate(ctx, stream, &xs, &config.temperatures)?;

    Ok(PipelineResult { tc, xs, macs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_pipeline_runs() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let energies = EnergyGrid::logarithmic(0.1, 10.0, 20).unwrap();
        let a_target = 56.0_f64;
        let mu = nucrust_core::units::reduced_mass(1.008664, a_target);
        let hbar2_2mu =
            nucrust_core::units::HBAR_C.powi(2) / (2.0 * mu * nucrust_core::units::AMU_MEV);

        let config = PipelineConfig {
            numerov: GpuNumerovParams {
                v_real: 50.0,
                w_imag: 5.0,
                r0: 1.25,
                a_ws: 0.65,
                r_ws: 1.25 * a_target.cbrt(),
                r_match: 1.4 * a_target.cbrt() + 5.0,
                step_size: 0.05,
                hbar2_over_2mu: hbar2_2mu,
                eta: 0.0,
            },
            hf: GpuHfParams {
                two_j_max: 10,
                q_value: 7.646,
                nld_a: 6.21,
                nld_t: 0.88,
                gsf_e_gdr: 16.36,
                gsf_gamma_gdr: 4.58,
                gsf_sigma_gdr: 136.0,
                hbar2_over_2mu: hbar2_2mu,
            },
            max_l: 5,
            temperatures: vec![0.3, 1.0, 3.0],
        };

        let result = batch_pipeline(&ctx, &stream, &energies, &config).unwrap();

        // Verify all stages produced output
        assert!(!result.tc.data.is_empty(), "TC should have data");
        assert_eq!(result.xs.sigma_total.len(), 20, "XS should have 20 points");
        assert_eq!(result.macs.len(), 3, "MACS should have 3 temperatures");

        // All values should be non-negative and finite
        for &m in &result.macs {
            assert!(m >= 0.0 && m.is_finite(), "MACS = {} invalid", m);
        }
    }
}
