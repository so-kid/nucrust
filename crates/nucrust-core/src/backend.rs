use crate::{CollisionMatrix, CoreError, CrossSection, Nuclide, ReactionRate, TransmissionCoeffs};

/// Compute backend abstraction (CPU / GPU).
///
/// All methods accept batch inputs; single computations are batch-size-1.
pub trait ComputeBackend: Send + Sync {
    /// Batch Numerov integration for transmission coefficients.
    fn batch_numerov(
        &self,
        params: &[NumerovTask],
        config: &NumerovConfig,
    ) -> Result<TransmissionCoeffs, CoreError>;

    /// Hauser-Feshbach J-pi summation.
    ///
    /// NLD/GSF are passed as parameter buffers (not trait objects) because
    /// GPU kernels cannot call through vtables on the host side.
    fn hf_summation(
        &self,
        tc: &TransmissionCoeffs,
        nld_params: &NldModelParams,
        gsf_params: &GsfModelParams,
        config: &HfConfig,
    ) -> Result<CrossSection, CoreError>;

    /// R-matrix collision matrix (energy-parallel).
    fn rmatrix_solve(
        &self,
        rmatrix: &RMatrixParams,
        energies: &[f64],
    ) -> Result<CollisionMatrix, CoreError>;

    /// MACS integration (nuclide x temperature batch).
    fn macs_integrate(
        &self,
        cross_sections: &[CrossSection],
        temperatures: &[f64],
        config: &MacsConfig,
    ) -> Result<Vec<ReactionRate>, CoreError>;
}

/// Numerov integration task descriptor.
#[derive(Debug, Clone)]
pub struct NumerovTask {
    pub nuclide: Nuclide,
    pub energy_mev: f64,
    pub l: u32,
    pub j: f64,
}

/// Numerov integrator configuration.
#[derive(Debug, Clone)]
pub struct NumerovConfig {
    pub step_size: f64,
    pub r_min: f64,
    pub convergence_tl: f64,
    pub max_l: u32,
}

impl Default for NumerovConfig {
    fn default() -> Self {
        Self {
            step_size: 0.05,
            r_min: 0.01,
            convergence_tl: 1e-10,
            max_l: 30,
        }
    }
}

/// Hauser-Feshbach configuration.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub j_max: i32,
    pub max_particle_stages: u32,
    pub max_gamma_steps: u32,
    pub wfc_model: WfcModel,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            j_max: 30,
            max_particle_stages: 3,
            max_gamma_steps: 30,
            wfc_model: WfcModel::Moldauer,
        }
    }
}

/// Width fluctuation correction model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WfcModel {
    None,
    Moldauer,
    Goe,
}

/// R-matrix parameters (placeholder, to be expanded in nucrust-rmatrix).
#[derive(Debug, Clone)]
pub struct RMatrixParams {
    pub n_channels: usize,
    pub n_levels: usize,
}

/// MACS integration configuration.
#[derive(Debug, Clone)]
pub struct MacsConfig {
    pub n_quadrature_points: usize,
}

impl Default for MacsConfig {
    fn default() -> Self {
        Self {
            n_quadrature_points: 32,
        }
    }
}

/// NLD model parameters for GPU transfer.
#[derive(Debug, Clone)]
pub enum NldModelParams {
    GilbertCameron {
        a: f64,
        delta: f64,
        t: f64,
        e0: f64,
        e_match: f64,
        sigma: f64,
    },
    Bsfg {
        a: f64,
        delta: f64,
        sigma: f64,
    },
    ConstantTemperature {
        t: f64,
        e0: f64,
    },
    Ignatyuk {
        a_tilde: f64,
        delta_w: f64,
        gamma: f64,
        delta: f64,
    },
    HfbTable {
        excitations: Vec<f64>,
        spins: Vec<f64>,
        densities: Vec<f64>,
    },
}

/// GSF model parameters for GPU transfer.
#[derive(Debug, Clone)]
pub enum GsfModelParams {
    Slo {
        e_gdr: f64,
        gamma_gdr: f64,
        sigma_gdr: f64,
    },
    Eglo {
        e_gdr: f64,
        gamma_gdr: f64,
        sigma_gdr: f64,
        temperature: f64,
    },
    QrpaTable {
        energies: Vec<f64>,
        strengths_e1: Vec<f64>,
        strengths_m1: Vec<f64>,
    },
}

/// Trait for converting physical model trait objects to GPU-transferable parameters.
pub trait ToDeviceParams {
    type Params;
    fn to_params(&self, nuclide: &Nuclide) -> Self::Params;
}

// CpuBackend is implemented in the `nucrust` aggregator crate,
// which has access to all downstream computation crates.
