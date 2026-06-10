#![allow(clippy::useless_conversion)] // PyO3 macros generate these
#![warn(missing_docs)]
//! Python bindings for nucrust via PyO3.
//!
//! Provides:
//! - PyNuclide, PyCrossSection, PyReactionRate wrapper types
//! - calc_transmission_coeffs, calc_hf_cross_section, calc_macs, fit_reaclib functions
//! - NumPy zero-copy array access for cross section data

use numpy::PyArray1;
use pyo3::prelude::*;

use nucrust_core::{Channel, EnergyGrid, Nuclide, Projectile};

// ======================== Wrapper types ========================

/// Python wrapper for Nuclide.
#[pyclass(name = "Nuclide")]
#[derive(Clone)]
pub struct PyNuclide {
    inner: Nuclide,
}

#[pymethods]
impl PyNuclide {
    #[new]
    fn new(z: u16, a: u16) -> PyResult<Self> {
        let inner = Nuclide::new(z, a)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn z(&self) -> u16 {
        self.inner.z()
    }

    #[getter]
    fn a(&self) -> u16 {
        self.inner.a()
    }

    #[getter]
    fn n(&self) -> u16 {
        self.inner.n()
    }

    fn __repr__(&self) -> String {
        format!("Nuclide(Z={}, A={})", self.inner.z(), self.inner.a())
    }
}

/// Python wrapper for cross section results.
#[pyclass(name = "CrossSection")]
#[derive(Clone)]
pub struct PyCrossSection {
    energies: Vec<f64>,
    sigma_total: Vec<f64>,
    sigma_elastic: Vec<f64>,
    sigma_reaction: Vec<f64>,
}

#[pymethods]
impl PyCrossSection {
    /// Energy grid as NumPy array (MeV).
    fn energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.energies)
    }

    /// Total cross section as NumPy array (mb).
    fn sigma_total<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.sigma_total)
    }

    /// Elastic cross section as NumPy array (mb).
    fn sigma_elastic<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.sigma_elastic)
    }

    /// Reaction (non-elastic) cross section as NumPy array (mb).
    fn sigma_reaction<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.sigma_reaction)
    }

    #[getter]
    fn n_energies(&self) -> usize {
        self.energies.len()
    }

    fn __repr__(&self) -> String {
        format!("CrossSection({} energy points)", self.energies.len())
    }
}

/// Python wrapper for reaction rate results.
#[pyclass(name = "ReactionRate")]
#[derive(Clone)]
pub struct PyReactionRate {
    temperatures: Vec<f64>,
    na_sigma_v: Vec<f64>,
    macs: Option<Vec<f64>>,
}

#[pymethods]
impl PyReactionRate {
    /// Temperatures as NumPy array (GK).
    fn temperatures<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.temperatures)
    }

    /// NA<σv> as NumPy array (cm³/mol/s).
    fn na_sigma_v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.na_sigma_v)
    }

    /// MACS as NumPy array (mb), if available.
    fn macs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.macs
            .as_ref()
            .map(|m| PyArray1::from_slice_bound(py, m))
    }

    fn __repr__(&self) -> String {
        format!(
            "ReactionRate({} temperature points)",
            self.temperatures.len()
        )
    }
}

// ======================== Functions ========================

/// Return type for calc_transmission_coeffs to satisfy clippy::type_complexity.
type TransmissionResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Compute transmission coefficients for neutron scattering.
///
/// Args:
///     z: Target atomic number
///     a: Target mass number
///     e_min: Minimum energy (MeV)
///     e_max: Maximum energy (MeV)
///     n_energies: Number of energy points
///     l_max: Maximum orbital angular momentum (default: 20)
///
/// Returns:
///     Tuple of (energies, T_l=0 values) as NumPy arrays
#[pyfunction]
#[pyo3(signature = (z, a, e_min, e_max, n_energies, l_max=20))]
fn calc_transmission_coeffs<'py>(
    py: Python<'py>,
    z: u16,
    a: u16,
    e_min: f64,
    e_max: f64,
    n_energies: usize,
    l_max: u32,
) -> PyResult<TransmissionResult<'py>> {
    let nuclide =
        Nuclide::new(z, a).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let channel = Channel {
        projectile: Projectile::Neutron,
        target: nuclide,
        q_value: 0.0,
    };
    let energies = EnergyGrid::logarithmic(e_min, e_max, n_energies)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let config = nucrust_core::backend::NumerovConfig {
        max_l: l_max,
        ..Default::default()
    };

    let omp = nucrust_optical::omp::KoningDelaroche;
    let tc = nucrust_optical::transmission::compute_transmission_coeffs(
        &omp, &channel, &energies, &config,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Return energies and s-wave (l=0) transmission
    let e_arr = PyArray1::from_slice_bound(py, energies.as_slice());
    let n_e = energies.len();
    // l=0 data starts at offset 0, j_idx=1 (j=0.5) at offset n_e
    let t_l0: Vec<f64> = (0..n_e)
        .map(|i| {
            if tc.data.len() > n_e + i {
                tc.data[n_e + i]
            } else {
                0.0
            }
        })
        .collect();
    let t_arr = PyArray1::from_slice_bound(py, &t_l0);

    Ok((e_arr, t_arr))
}

/// Compute Hauser-Feshbach cross section.
///
/// Args:
///     z: Target atomic number
///     a: Target mass number
///     e_min, e_max, n_energies: Energy grid parameters
///     q_value: Q-value for (n,γ) reaction (MeV)
///
/// Returns:
///     PyCrossSection object
#[pyfunction]
#[pyo3(signature = (z, a, e_min, e_max, n_energies, q_value=0.0))]
fn calc_hf_cross_section(
    z: u16,
    a: u16,
    e_min: f64,
    e_max: f64,
    n_energies: usize,
    q_value: f64,
) -> PyResult<PyCrossSection> {
    let nuclide =
        Nuclide::new(z, a).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let channel = Channel {
        projectile: Projectile::Neutron,
        target: nuclide,
        q_value,
    };
    let energies = EnergyGrid::logarithmic(e_min, e_max, n_energies)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let config = nucrust_core::backend::NumerovConfig {
        max_l: 15,
        ..Default::default()
    };

    let omp = nucrust_optical::omp::KoningDelaroche;
    let tc = nucrust_optical::transmission::compute_transmission_coeffs(
        &omp, &channel, &energies, &config,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // HF with default NLD/GSF
    let nld = nucrust_hf::nld::ConstantTemperature {
        temperature: 0.88,
        e0: -1.16,
        a: 6.21,
    };
    let gsf = nucrust_hf::gsf::StandardLorentzian {
        e_gdr: 16.36,
        gamma_gdr: 4.58,
        sigma_gdr: 136.0,
        m1_params: None,
    };
    let hf_config = nucrust_hf::hf::HfConfig {
        two_j_max: 20,
        exit_channels: vec![Projectile::Gamma],
        ..nucrust_hf::hf::HfConfig::default()
    };

    let calc = nucrust_hf::hf::HfCalculation {
        entrance: &channel,
        tc_entrance: &tc,
        exit_particle_channels: vec![],
        nld: &nld,
        gsf: &gsf,
        config: &hf_config,
        discrete_levels: None,
    };

    let results = nucrust_hf::hf::hauser_feshbach(&calc)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyCrossSection {
        energies: energies.as_slice().to_vec(),
        sigma_total: results.iter().map(|r| r.sigma_cn).collect(),
        sigma_elastic: vec![0.0; n_energies],
        sigma_reaction: results
            .iter()
            .map(|r| r.sigma_channels.first().copied().unwrap_or(0.0))
            .collect(),
    })
}

/// Compute MACS and reaction rates.
///
/// Args:
///     z, a: Target nuclide
///     e_min, e_max, n_energies: Energy grid for cross section
///     q_value: Q-value (MeV)
///
/// Returns:
///     PyReactionRate object
#[pyfunction]
#[pyo3(signature = (z, a, e_min=0.001, e_max=1.0, n_energies=100, q_value=0.0))]
fn calc_macs(
    z: u16,
    a: u16,
    e_min: f64,
    e_max: f64,
    n_energies: usize,
    q_value: f64,
) -> PyResult<PyReactionRate> {
    let xs = calc_hf_cross_section(z, a, e_min, e_max, n_energies, q_value)?;

    let energy_grid = EnergyGrid::from_values(xs.energies.clone())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let cross_section = nucrust_core::CrossSection {
        energy: energy_grid,
        sigma_total: xs.sigma_total.clone(),
        sigma_elastic: xs.sigma_elastic.clone(),
        sigma_reaction: xs.sigma_reaction.clone(),
        partial: vec![],
    };

    let macs_config = nucrust_astro::MacsConfig::default();
    let macs = nucrust_astro::compute_macs(&cross_section, &macs_config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mu = nucrust_core::units::reduced_mass(Projectile::Neutron.mass_amu(), a as f64);
    let rate = nucrust_astro::compute_reaction_rate(&macs, &macs_config.temperature_grid, mu)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyReactionRate {
        temperatures: rate.temperatures,
        na_sigma_v: rate.na_sigma_v,
        macs: rate.macs,
    })
}

/// Fit REACLIB 7-parameter coefficients to rate data.
///
/// Args:
///     temperatures: Temperature grid (GK)
///     rates: NA<σv> values
///
/// Returns:
///     List of 7 fitted coefficients [a0, a1, ..., a6]
#[pyfunction]
fn fit_reaclib(temperatures: Vec<f64>, rates: Vec<f64>) -> PyResult<Vec<f64>> {
    let coeffs = nucrust_data::reaclib::fit_reaclib_params(&temperatures, &rates)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(coeffs.to_vec())
}

// ======================== Module ========================

/// nucrust: GPU-accelerated nuclear reaction rate calculations
#[pymodule]
fn nucrust_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNuclide>()?;
    m.add_class::<PyCrossSection>()?;
    m.add_class::<PyReactionRate>()?;
    m.add_function(wrap_pyfunction!(calc_transmission_coeffs, m)?)?;
    m.add_function(wrap_pyfunction!(calc_hf_cross_section, m)?)?;
    m.add_function(wrap_pyfunction!(calc_macs, m)?)?;
    m.add_function(wrap_pyfunction!(fit_reaclib, m)?)?;
    Ok(())
}
