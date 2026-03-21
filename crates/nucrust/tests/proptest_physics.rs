//! Property-based tests for physical invariants.
//!
//! T-2A.14: T_{lj} ∈ [0, 1] for any (E, l, j, OMP parameters)
//! T-2B.15: σ ≥ 0 for any energy

use nucrust::cpu_backend::cpu_transmission_coeffs;
use nucrust_core::backend::NumerovConfig;
use nucrust_core::{Channel, EnergyGrid, Nuclide, Projectile};
use proptest::prelude::*;

/// Generate a valid neutron channel with random target nuclide.
fn arb_neutron_channel() -> impl Strategy<Value = Channel> {
    // Z: 20-82 (medium to heavy nuclei where KD is valid)
    // A: Z+Z/2 to Z+Z*2 (reasonable N/Z ratio)
    (20_u16..=82)
        .prop_flat_map(|z| {
            let a_min = z + z / 2;
            let a_max = (z * 3).min(250);
            (Just(z), a_min..=a_max)
        })
        .prop_map(|(z, a)| Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(z, a).unwrap(),
            q_value: 0.0,
        })
}

proptest! {
    /// T-2A.14: Transmission coefficients must be in [0, 1] for all (E, l).
    #[test]
    fn transmission_coeffs_in_unit_interval(
        energy in 0.5_f64..20.0,
        z in 24_u16..=82,
    ) {
        let a = z + z / 2 + 5; // reasonable mass number
        let nuclide = match Nuclide::new(z, a) {
            Ok(n) => n,
            Err(_) => return Ok(()),
        };
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: nuclide,
            q_value: 0.0,
        };
        let energies = EnergyGrid::from_values(vec![energy]).unwrap();
        let config = NumerovConfig {
            max_l: 5,
            ..Default::default()
        };

        let tc = cpu_transmission_coeffs(&ch, &energies, &config)?;

        for (i, &t) in tc.data.iter().enumerate() {
            prop_assert!(
                t >= 0.0 && t <= 1.0,
                "T[{}] = {} out of [0,1] for Z={}, A={}, E={}",
                i, t, z, a, energy
            );
        }
    }

    /// T-2A.14: Transmission coefficients must be finite.
    #[test]
    fn transmission_coeffs_finite(
        energy in 0.1_f64..20.0,
    ) {
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        };
        let energies = EnergyGrid::from_values(vec![energy]).unwrap();
        let config = NumerovConfig {
            max_l: 5,
            ..Default::default()
        };

        let tc = cpu_transmission_coeffs(&ch, &energies, &config)?;

        for &t in &tc.data {
            prop_assert!(t.is_finite(), "T is not finite at E={}", energy);
        }
    }

    /// T-2B.15: Cross sections must be non-negative.
    #[test]
    fn cross_section_non_negative(
        energy in 0.5_f64..15.0,
    ) {
        use nucrust::cpu_backend::CpuBackend;
        use nucrust_core::backend::{ComputeBackend, GsfModelParams, HfConfig, NldModelParams};

        let ch = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646,
        };
        let energies = EnergyGrid::from_values(vec![energy]).unwrap();
        let config = NumerovConfig {
            max_l: 5,
            ..Default::default()
        };

        let tc = cpu_transmission_coeffs(&ch, &energies, &config)?;

        let backend = CpuBackend::new();
        let nld = NldModelParams::ConstantTemperature { t: 0.88, e0: -1.16 };
        let gsf = GsfModelParams::Slo {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
        };

        let xs = backend.hf_summation(&tc, &nld, &gsf, &HfConfig::default())?;

        for &sigma in &xs.sigma_total {
            prop_assert!(sigma >= 0.0, "sigma = {} < 0 at E={}", sigma, energy);
        }
    }
}

/// Coulomb wave function Wronskian invariant under random parameters.
#[test]
fn coulomb_wronskian_property() {
    use nucrust_special::coulomb_wave;
    use proptest::test_runner::{Config, TestRunner};

    let mut runner = TestRunner::new(Config {
        cases: 50,
        ..Config::default()
    });

    runner
        .run(
            &(0.0_f64..10.0, 0.7_f64..20.0, 0_u32..5),
            |(eta, rho, l)| {
                let result = coulomb_wave(eta, rho, l, 1);
                if let Ok(r) = result {
                    let w = r.fp[0] * r.g[0] - r.f[0] * r.gp[0];
                    prop_assert!(
                        (w - 1.0).abs() < 1e-4,
                        "Wronskian = {} (eta={}, rho={}, l={})",
                        w,
                        eta,
                        rho,
                        l
                    );
                }
                Ok(())
            },
        )
        .unwrap();
}
