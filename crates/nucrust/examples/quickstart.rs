//! Quick-start example from the book (book/src/quickstart.md).
//!
//! Computes neutron transmission coefficients for n + ⁵⁶Fe on the CPU.
//! Keep this file in sync with the book chapter — it exists so the
//! documented example is compile-checked by `cargo test` / CI.

use nucrust::cpu_backend::cpu_transmission_coeffs;
use nucrust::nucrust_core::backend::NumerovConfig;
use nucrust::nucrust_core::{Channel, EnergyGrid, Nuclide, Projectile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the reaction: n + ⁵⁶Fe
    let target = Nuclide::new(26, 56)?;
    let channel = Channel {
        projectile: Projectile::Neutron,
        target,
        q_value: 0.0,
    };

    // Energy grid: 0.001 to 20 MeV, 100 points (log spacing)
    let energies = EnergyGrid::logarithmic(0.001, 20.0, 100)?;

    // Compute transmission coefficients
    let config = NumerovConfig::default();
    let tc = cpu_transmission_coeffs(&channel, &energies, &config)?;

    println!("l_max = {}", tc.l_max);
    println!("T at first grid point = {:.6e}", tc.data[0]);

    Ok(())
}
