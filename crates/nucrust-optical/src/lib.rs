pub mod cc_transmission;
pub mod deformation;
pub mod johnson_logderiv;
pub mod numerov;
pub mod omp;
pub mod potential;
pub mod transmission;

pub use cc_transmission::compute_transmission_auto;
pub use deformation::{CcChannel, CoupledChannelSystem, DeformedKoningDelaroche};
pub use omp::{Avrigeanu2014, CustomOmp, KoningDelaroche, McFaddenSatchler};
pub use potential::{coulomb_potential, spin_orbit_factor, woods_saxon, woods_saxon_deriv};
pub use transmission::compute_transmission_coeffs;
