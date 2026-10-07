//! End-to-end integration test: transmission coefficients → HF → cross section.
//!
//! Validates that the full pipeline produces physically reasonable results
//! for 56Fe(n,γ) using the Koning-Delaroche OMP + CT level density + SLO GSF.

use nucrust::cpu_backend::cpu_transmission_coeffs;
use nucrust_core::{Channel, EnergyGrid, Nuclide, Projectile};

/// Create 56Fe neutron capture channel.
fn fe56_ng_channel() -> Channel {
    Channel {
        projectile: Projectile::Neutron,
        target: Nuclide::new(26, 56).unwrap(),
        q_value: 7.646, // Sn of 57Fe (MeV)
    }
}

#[test]
fn e2e_fe56_neutron_transmission_coefficients() {
    let ch = fe56_ng_channel();
    let energies = EnergyGrid::logarithmic(0.1, 10.0, 10).unwrap();
    let config = nucrust_core::backend::NumerovConfig {
        max_l: 10,
        ..Default::default()
    };

    let tc = cpu_transmission_coeffs(&ch, &energies, &config).unwrap();

    // Basic sanity checks
    assert!(tc.l_max >= 1, "l_max = {} (too low)", tc.l_max);
    assert!(!tc.data.is_empty(), "no transmission data computed");

    // All T values should be finite and in [0, 1]
    for (i, &t) in tc.data.iter().enumerate() {
        assert!(
            t.is_finite() && (0.0..=1.0).contains(&t),
            "T[{}] = {} out of [0,1]",
            i,
            t
        );
    }
}

#[test]
fn e2e_fe56_hf_cross_section() {
    use nucrust_hf::gsf::StandardLorentzian;
    use nucrust_hf::hf::{self, HfCalculation};
    use nucrust_hf::nld::ConstantTemperature;

    // Step 1: Compute transmission coefficients
    let ch = fe56_ng_channel();
    let energies = EnergyGrid::logarithmic(0.5, 5.0, 5).unwrap();
    let numerov_config = nucrust_core::backend::NumerovConfig {
        max_l: 8,
        ..Default::default()
    };

    let tc = cpu_transmission_coeffs(&ch, &energies, &numerov_config).unwrap();

    // Step 2: HF cross section using HfCalculation directly (with correct Q-value)
    let nld = ConstantTemperature {
        temperature: 0.88,
        e0: -1.16,
        a: 6.21,
    };
    let gsf = StandardLorentzian {
        e_gdr: 16.36,
        gamma_gdr: 4.58,
        sigma_gdr: 136.0,
        m1_params: None,
    };
    let hf_config = hf::HfConfig {
        two_j_max: 20,
        exit_channels: vec![Projectile::Gamma],
        ..hf::HfConfig::default()
    };

    let calc = HfCalculation {
        entrance: &ch,
        tc_entrance: &tc,
        exit_particle_channels: vec![],
        nld: &nld,
        gsf: &gsf,
        config: &hf_config,
        discrete_levels: None,
    };

    let results = hf::hauser_feshbach(&calc).unwrap();

    assert_eq!(results.len(), energies.len());

    for (i, result) in results.iter().enumerate() {
        let e = energies.as_slice()[i];
        // Compound formation cross section should be positive
        assert!(
            result.sigma_cn >= 0.0,
            "sigma_cn(E={} MeV) = {} < 0",
            e,
            result.sigma_cn
        );
        assert!(
            result.sigma_cn.is_finite(),
            "sigma_cn(E={} MeV) not finite",
            e
        );
        // Gamma cross section should be positive (with Q=7.6 MeV excitation)
        assert!(
            result.sigma_channels[0] >= 0.0,
            "sigma_gamma(E={} MeV) = {} < 0",
            e,
            result.sigma_channels[0]
        );
    }

    // At least one energy should have non-zero cross section
    let max_sigma_cn = results.iter().map(|r| r.sigma_cn).fold(0.0_f64, f64::max);
    assert!(
        max_sigma_cn > 0.0,
        "all sigma_cn are zero (max = {})",
        max_sigma_cn
    );
}

#[test]
fn e2e_pipeline_all_sigma_non_negative() {
    use nucrust_hf::gsf::StandardLorentzian;
    use nucrust_hf::hf::{self, HfCalculation};
    use nucrust_hf::nld::ConstantTemperature;

    let ch = fe56_ng_channel();
    let energies = EnergyGrid::logarithmic(1.0, 10.0, 5).unwrap();
    let config = nucrust_core::backend::NumerovConfig {
        max_l: 6,
        ..Default::default()
    };

    let tc = cpu_transmission_coeffs(&ch, &energies, &config).unwrap();

    let nld = ConstantTemperature {
        temperature: 0.88,
        e0: -1.16,
        a: 6.21,
    };
    let gsf = StandardLorentzian {
        e_gdr: 16.36,
        gamma_gdr: 4.58,
        sigma_gdr: 136.0,
        m1_params: None,
    };
    let hf_config = hf::HfConfig {
        two_j_max: 16,
        exit_channels: vec![Projectile::Gamma],
        ..hf::HfConfig::default()
    };

    let calc = HfCalculation {
        entrance: &ch,
        tc_entrance: &tc,
        exit_particle_channels: vec![],
        nld: &nld,
        gsf: &gsf,
        config: &hf_config,
        discrete_levels: None,
    };

    let results = hf::hauser_feshbach(&calc).unwrap();

    for result in &results {
        assert!(
            result.sigma_cn >= 0.0,
            "negative sigma_cn: {}",
            result.sigma_cn
        );
        for &s in &result.sigma_channels {
            assert!(s >= 0.0, "negative partial sigma: {}", s);
        }
    }
}

#[test]
fn e2e_fe56_cross_section_magnitudes() {
    use nucrust_core::traits::LevelDensity;
    use nucrust_core::Parity;
    use nucrust_hf::gsf::StandardLorentzian;
    use nucrust_hf::hf::{self, HfCalculation};
    use nucrust_hf::nld::ConstantTemperature;

    let ch = fe56_ng_channel();
    let energies = EnergyGrid::from_values(vec![0.03, 1.0]).unwrap();
    let config = nucrust_core::backend::NumerovConfig {
        max_l: 10,
        ..Default::default()
    };
    let tc = cpu_transmission_coeffs(&ch, &energies, &config).unwrap();

    // s-wave transmission (j = 1/2 slot) of Koning-Delaroche at 1 MeV: ~0.93.
    let t0 = tc.get(0, 1, 1);
    assert!(t0 > 0.85 && t0 < 1.0, "T_0(1 MeV) = {t0}");
    assert_eq!(tc.get_l_averaged(0, 1), t0);

    // CT level density tuned to the s-wave resonance spacing of 57Fe (D0 ~ 25 keV).
    let nld = ConstantTemperature {
        temperature: 0.88,
        e0: 1.94,
        a: 6.21,
    };
    let fe57 = Nuclide::new(26, 57).unwrap();
    let d0_kev = 1e3 / nld.rho(&fe57, 7.646, 0.5, Parity::Positive);
    assert!(d0_kev > 15.0 && d0_kev < 40.0, "D0 = {d0_kev} keV");

    let gsf = StandardLorentzian {
        e_gdr: 16.36,
        gamma_gdr: 4.58,
        sigma_gdr: 136.0,
        m1_params: None,
    };
    let hf_config = hf::HfConfig {
        two_j_max: 20,
        exit_channels: vec![Projectile::Gamma],
        ..hf::HfConfig::default()
    };
    let calc = HfCalculation {
        entrance: &ch,
        tc_entrance: &tc,
        exit_particle_channels: vec![],
        nld: &nld,
        gsf: &gsf,
        config: &hf_config,
        discrete_levels: None,
    };
    let r = hf::hauser_feshbach(&calc).unwrap();

    // Compound formation (= optical reaction) cross section at 1 MeV: ~2.3 b, not ~11 b.
    assert!(
        r[1].sigma_cn > 1800.0 && r[1].sigma_cn < 2800.0,
        "sigma_CN(1 MeV) = {} mb",
        r[1].sigma_cn
    );
    // Radiative capture: ~10 mb at 30 keV, a few mb at 1 MeV (inelastic competition
    // is not included here, which matters only above the 847 keV 2+ level).
    assert!(
        r[0].sigma_channels[0] > 3.0 && r[0].sigma_channels[0] < 30.0,
        "sigma_gamma(30 keV) = {} mb",
        r[0].sigma_channels[0]
    );
    assert!(
        r[1].sigma_channels[0] > 1.0 && r[1].sigma_channels[0] < 10.0,
        "sigma_gamma(1 MeV) = {} mb",
        r[1].sigma_channels[0]
    );
}
