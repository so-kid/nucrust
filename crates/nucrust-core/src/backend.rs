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
    /// Target nuclide.
    pub nuclide: Nuclide,
    /// Center-of-mass energy (MeV).
    pub energy_mev: f64,
    /// Orbital angular momentum quantum number.
    pub l: u32,
    /// Total angular momentum quantum number j = l +/- 1/2.
    pub j: f64,
}

/// Numerov integrator configuration.
#[derive(Debug, Clone)]
pub struct NumerovConfig {
    /// Radial step size (fm).
    pub step_size: f64,
    /// Minimum integration radius (fm).
    pub r_min: f64,
    /// Convergence threshold for transmission coefficients.
    pub convergence_tl: f64,
    /// Maximum orbital angular momentum.
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
    /// Maximum total angular momentum J for the compound nucleus summation.
    pub j_max: i32,
    /// Maximum number of particle emission stages.
    pub max_particle_stages: u32,
    /// Maximum number of discrete gamma-ray steps.
    pub max_gamma_steps: u32,
    /// Width fluctuation correction model.
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
    /// No width fluctuation correction.
    None,
    /// Moldauer width fluctuation correction.
    Moldauer,
    /// GOE (Gaussian Orthogonal Ensemble) triple integral.
    Goe,
}

/// R-matrix parameters for GPU transfer and backend dispatch.
///
/// Contains serialized channel and level data. The CPU backend converts
/// these to `nucrust_rmatrix::RMatrixParams` for computation.
#[derive(Debug, Clone)]
pub struct RMatrixParams {
    /// Number of reaction channels.
    pub n_channels: usize,
    /// Number of R-matrix levels (poles).
    pub n_levels: usize,
    /// Channel data: (projectile, target_Z, target_A, q_value, separation_energy, l, s, two_j, parity_sign, radius)
    /// Flattened: `channels[i]` = RMatrixChannelData
    pub channels: Vec<RMatrixChannelData>,
    /// Level data: (energy, `reduced_widths[n_channels]`)
    pub levels: Vec<RMatrixLevelData>,
    /// Boundary condition: None = Brune, Some(vec) = Standard with B_c values.
    pub boundary_b: Option<Vec<f64>>,
}

/// Serialized R-matrix channel for backend transfer.
#[derive(Debug, Clone)]
pub struct RMatrixChannelData {
    /// Projectile type index (0=n, 1=p, 2=alpha, 3=gamma, etc.)
    pub projectile: crate::Projectile,
    /// Target nucleus.
    pub target: crate::Nuclide,
    /// Q-value (MeV).
    pub q_value: f64,
    /// Separation energy (MeV).
    pub separation_energy: f64,
    /// Orbital angular momentum.
    pub l: u32,
    /// Channel spin.
    pub s: f64,
    /// Total angular momentum J (as SpinParity).
    pub j: crate::SpinParity,
    /// Channel radius (fm).
    pub radius: f64,
}

/// Serialized R-matrix level for backend transfer.
#[derive(Debug, Clone)]
pub struct RMatrixLevelData {
    /// Pole energy (MeV).
    pub energy: f64,
    /// Reduced width amplitudes, one per channel.
    pub reduced_widths: Vec<f64>,
}

/// MACS integration configuration.
#[derive(Debug, Clone)]
pub struct MacsConfig {
    /// Number of Gauss-Laguerre quadrature points.
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
    /// Gilbert-Cameron composite model (constant temperature + Fermi gas).
    GilbertCameron {
        /// Level density parameter a (1/MeV).
        a: f64,
        /// Pairing energy shift (MeV).
        delta: f64,
        /// Nuclear temperature (MeV).
        t: f64,
        /// Energy backshift E0 (MeV).
        e0: f64,
        /// Matching energy between CT and BSFG regions (MeV).
        e_match: f64,
        /// Spin cutoff parameter.
        sigma: f64,
    },
    /// Back-shifted Fermi gas model.
    Bsfg {
        /// Level density parameter a (1/MeV).
        a: f64,
        /// Pairing energy shift (MeV).
        delta: f64,
        /// Spin cutoff parameter.
        sigma: f64,
    },
    /// Constant temperature model.
    ConstantTemperature {
        /// Nuclear temperature (MeV).
        t: f64,
        /// Energy backshift E0 (MeV).
        e0: f64,
    },
    /// Ignatyuk energy-dependent level density parameter model.
    Ignatyuk {
        /// Asymptotic level density parameter (1/MeV).
        a_tilde: f64,
        /// Shell correction energy (MeV).
        delta_w: f64,
        /// Damping parameter for shell effects.
        gamma: f64,
        /// Pairing energy shift (MeV).
        delta: f64,
    },
    /// Tabulated HFB (Hartree-Fock-Bogoliubov) level densities.
    HfbTable {
        /// Excitation energies (MeV).
        excitations: Vec<f64>,
        /// Spin values.
        spins: Vec<f64>,
        /// Level density values (1/MeV).
        densities: Vec<f64>,
    },
}

/// GSF model parameters for GPU transfer.
#[derive(Debug, Clone)]
pub enum GsfModelParams {
    /// Standard Lorentzian (Brink-Axel) model.
    Slo {
        /// Giant dipole resonance energy (MeV).
        e_gdr: f64,
        /// Giant dipole resonance width (MeV).
        gamma_gdr: f64,
        /// Giant dipole resonance peak cross section (mb).
        sigma_gdr: f64,
    },
    /// Enhanced Generalized Lorentzian model.
    Eglo {
        /// Giant dipole resonance energy (MeV).
        e_gdr: f64,
        /// Giant dipole resonance width (MeV).
        gamma_gdr: f64,
        /// Giant dipole resonance peak cross section (mb).
        sigma_gdr: f64,
        /// Nuclear temperature for EGLO damping (MeV).
        temperature: f64,
    },
    /// Tabulated QRPA (Quasi-particle RPA) strengths.
    QrpaTable {
        /// Gamma-ray energies (MeV).
        energies: Vec<f64>,
        /// E1 strength function values.
        strengths_e1: Vec<f64>,
        /// M1 strength function values.
        strengths_m1: Vec<f64>,
    },
}

/// Trait for converting physical model trait objects to GPU-transferable parameters.
pub trait ToDeviceParams {
    /// The GPU-transferable parameter type.
    type Params;
    /// Convert this model to GPU-transferable parameters for the given nuclide.
    fn to_params(&self, nuclide: &Nuclide) -> Self::Params;
}

// CpuBackend is implemented in the `nucrust` aggregator crate,
// which has access to all downstream computation crates.
