//! Host-side replica of the `hf_summation.cu` kernel arithmetic.
//!
//! The CUDA kernel is NVRTC-compiled at runtime and cannot run without a GPU.
//! [`hf_summation_host`] performs the same per-energy computation on the CPU,
//! statement for statement, so that the kernel's physics (transmission layout,
//! spin/parity coupling, GSF units, gamma transmission) can be tested against
//! `nucrust-hf` without CUDA hardware. Any change to the kernel must be mirrored
//! here.

use nucrust_core::TransmissionCoeffs;

/// 0.1 fm²/mb / (3 π² (ħc)²) in mb⁻¹ MeV⁻² (same value as `nucrust_hf::gsf::GSF_E1_CONST`).
const GSF_E1_CONST: f64 = 8.673_733_205_921_499e-8;
/// SLO single-particle M1 strength without scissors mode (MeV⁻³).
const GSF_M1_DEFAULT: f64 = 1.0e-9;
/// SLO E2 coefficient: f_E2 = 5.2e-8 / E_GDR² (MeV⁻⁵).
const GSF_E2_COEFF: f64 = 5.2e-8;
const NUMERICAL_FLOOR: f64 = 1e-30;
const MIN_EXCITATION: f64 = 0.1;
const N_GAMMA_POINTS: usize = 50;

/// Parameters for GPU HF summation (n + target capture with compound elastic competition).
///
/// Models: constant-temperature level density and Standard Lorentzian GSF
/// (no scissors mode), matching `nucrust_hf::nld::ConstantTemperature` and
/// `nucrust_hf::gsf::StandardLorentzian { m1_params: None, .. }`.
#[derive(Debug, Clone)]
pub struct GpuHfParams {
    /// Maximum 2J value for spin summation.
    pub two_j_max: i32,
    /// Two times the projectile spin (1 for nucleons, 0 for alphas).
    pub proj_two_s: i32,
    /// Two times the target ground-state spin.
    pub target_two_i: i32,
    /// Target ground-state parity (+1 or -1).
    pub target_parity: i32,
    /// Reaction Q-value (MeV); compound excitation U = E + Q.
    pub q_value: f64,
    /// Mass number of the compound nucleus (spin cutoff of the CT level density).
    pub compound_a: f64,
    /// Level density parameter a (MeV^{-1}), used for the spin cutoff.
    pub nld_a: f64,
    /// Nuclear temperature T (MeV).
    pub nld_t: f64,
    /// Constant-temperature energy shift E0 (MeV).
    pub nld_e0: f64,
    /// GDR centroid energy (MeV).
    pub gsf_e_gdr: f64,
    /// GDR width (MeV).
    pub gsf_gamma_gdr: f64,
    /// GDR peak cross section (mb).
    pub gsf_sigma_gdr: f64,
    /// hbar^2 / (2 * mu) in MeV*fm^2.
    pub hbar2_over_2mu: f64,
}

fn entrance_transmission(
    tc_data: &[f64],
    n_e: usize,
    n_l: usize,
    e_idx: usize,
    two_j: i32,
    parity: i32,
    p: &GpuHfParams,
) -> f64 {
    let mut t_sum = 0.0;
    for l in 0..n_l {
        let orbital_parity = if l % 2 == 0 { 1 } else { -1 };
        if orbital_parity * p.target_parity != parity {
            continue;
        }
        for j_idx in 0..2_usize {
            let two_j_particle = if p.proj_two_s == 0 {
                if j_idx == 0 {
                    continue;
                }
                2 * l as i32
            } else {
                2 * l as i32 + (2 * j_idx as i32 - 1)
            };
            if two_j_particle < 0 {
                continue;
            }
            let two_i = p.target_two_i;
            if (two_j_particle + two_i + two_j) % 2 != 0
                || two_j < (two_j_particle - two_i).abs()
                || two_j > two_j_particle + two_i
            {
                continue;
            }
            t_sum += tc_data[l * 2 * n_e + j_idx * n_e + e_idx];
        }
    }
    t_sum
}

fn rho_ct(excitation: f64, two_jf: i32, p: &GpuHfParams) -> f64 {
    let u = excitation - p.nld_e0;
    if u < 0.0 {
        return 0.0;
    }
    let rho_tot = (1.0 / p.nld_t) * (u / p.nld_t).exp();
    let u_cut = excitation.max(0.01);
    let sigma_sq = 0.0888 * p.compound_a.powf(2.0 / 3.0) * (p.nld_a * u_cut).sqrt();
    if sigma_sq <= 0.0 {
        return 0.0;
    }
    let spin = two_jf as f64 / 2.0;
    let j_half = spin + 0.5;
    let f_spin =
        (2.0 * spin + 1.0) / (2.0 * sigma_sq) * (-j_half * j_half / (2.0 * sigma_sq)).exp();
    0.5 * rho_tot * f_spin
}

fn gsf_slo(e_gamma: f64, multipole: usize, p: &GpuHfParams) -> f64 {
    if e_gamma <= 0.0 {
        return 0.0;
    }
    match multipole {
        0 => {
            let e2 = e_gamma * e_gamma;
            let e02 = p.gsf_e_gdr * p.gsf_e_gdr;
            let g2 = p.gsf_gamma_gdr * p.gsf_gamma_gdr;
            let denom = (e2 - e02) * (e2 - e02) + e2 * g2;
            if denom == 0.0 {
                return 0.0;
            }
            GSF_E1_CONST * p.gsf_sigma_gdr * p.gsf_gamma_gdr * e_gamma * p.gsf_gamma_gdr / denom
        }
        1 => GSF_M1_DEFAULT,
        _ => GSF_E2_COEFF / (p.gsf_e_gdr * p.gsf_e_gdr),
    }
}

fn gamma_transmission(excitation: f64, two_j: i32, p: &GpuHfParams) -> f64 {
    if excitation < MIN_EXCITATION {
        return 0.0;
    }
    let mut t_gamma = 0.0;
    let e_cont_max = excitation;
    if e_cont_max > 0.1 {
        let de = e_cont_max / N_GAMMA_POINTS as f64;
        for multipole in 0..3_usize {
            let l_order = if multipole == 2 { 2 } else { 1 };
            for i in 1..N_GAMMA_POINTS {
                let e_gamma = i as f64 * de;
                let u_residual = excitation - e_gamma;
                if u_residual < 0.0 {
                    break;
                }
                let f_xl = gsf_slo(e_gamma, multipole, p);
                let mut e_factor = e_gamma * e_gamma * e_gamma;
                if l_order == 2 {
                    e_factor *= e_gamma * e_gamma;
                }
                let two_l = 2 * l_order;
                let j_min = (two_j - two_l).abs();
                let j_max = two_j + two_l;
                for two_jf in (j_min..=j_max).step_by(2) {
                    let rho = rho_ct(u_residual, two_jf, p);
                    t_gamma += f_xl * e_factor * rho * de;
                }
            }
        }
    }
    2.0 * std::f64::consts::PI * t_gamma
}

/// Host replica of the `hf_summation` kernel.
///
/// Returns `(sigma_cn, sigma_gamma)` in mb for every energy of `tc`.
pub fn hf_summation_host(tc: &TransmissionCoeffs, p: &GpuHfParams) -> (Vec<f64>, Vec<f64>) {
    let n_e = tc.energy.len();
    let n_l = (tc.l_max + 1) as usize;
    let mut sigma_out = vec![0.0; n_e];
    let mut sigma_gamma_out = vec![0.0; n_e];

    for (e_idx, &energy) in tc.energy.as_slice().iter().enumerate() {
        if energy <= 0.0 {
            continue;
        }
        let k_sq = energy / p.hbar2_over_2mu;
        let pi_over_k2 = std::f64::consts::PI / k_sq * 10.0;
        let excitation = energy + p.q_value;
        let spin_weight = (p.proj_two_s + 1) as f64 * (p.target_two_i + 1) as f64;
        let mut sig_cn = 0.0;
        let mut sig_gam = 0.0;

        for two_j in 0..=p.two_j_max {
            for parity in [1, -1] {
                let t_a = entrance_transmission(&tc.data, n_e, n_l, e_idx, two_j, parity, p);
                if t_a < NUMERICAL_FLOOR {
                    continue;
                }
                let t_gamma = gamma_transmission(excitation, two_j, p);
                let t_total = t_a + t_gamma;
                if t_total < NUMERICAL_FLOOR {
                    continue;
                }
                let g_j = (two_j + 1) as f64 / spin_weight;
                let prefactor = pi_over_k2 * g_j;
                sig_cn += prefactor * t_a;
                sig_gam += prefactor * t_a * t_gamma / t_total;
            }
        }
        sigma_out[e_idx] = sig_cn;
        sigma_gamma_out[e_idx] = sig_gam;
    }
    (sigma_out, sigma_gamma_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::units::{reduced_mass, AMU_MEV, HBAR_C};
    use nucrust_core::{Channel, EnergyGrid, Nuclide, Projectile};
    use nucrust_hf::gsf::StandardLorentzian;
    use nucrust_hf::hf::{hauser_feshbach, HfCalculation, HfConfig};
    use nucrust_hf::nld::ConstantTemperature;

    fn rel_diff(a: f64, b: f64) -> f64 {
        if a == b {
            0.0
        } else {
            (a - b).abs() / a.abs().max(b.abs())
        }
    }

    /// n + Fe-56 at a few energies with Fe-56-like T_lj (2 slots for every l).
    fn fe56_tc() -> TransmissionCoeffs {
        let energy = EnergyGrid::from_values(vec![0.1, 1.0, 3.0]).unwrap();
        let n_e = energy.len();
        let l_max = 3_u32;
        let mut data = vec![0.0; (l_max as usize + 1) * 2 * n_e];
        // [l][j_idx] values at each energy; l = 0 slot 0 (j = -1/2) stays empty.
        let base = [[0.0, 0.93], [0.21, 0.20], [0.35, 0.38], [0.004, 0.003]];
        for (l, slots) in base.iter().enumerate() {
            for (j_idx, &t) in slots.iter().enumerate() {
                for e_idx in 0..n_e {
                    let scale = 0.5 + 0.25 * e_idx as f64;
                    data[l * 2 * n_e + j_idx * n_e + e_idx] = (t * scale).min(1.0);
                }
            }
        }
        TransmissionCoeffs {
            energy,
            l_max,
            data,
        }
    }

    fn fe56_params(two_j_max: i32) -> GpuHfParams {
        let mu = reduced_mass(Projectile::Neutron.mass_amu(), 56.0);
        GpuHfParams {
            two_j_max,
            proj_two_s: 1,
            target_two_i: 0,
            target_parity: 1,
            q_value: 7.646,
            compound_a: 57.0,
            nld_a: 6.21,
            nld_t: 0.88,
            nld_e0: -1.16,
            gsf_e_gdr: 16.36,
            gsf_gamma_gdr: 4.58,
            gsf_sigma_gdr: 136.0,
            hbar2_over_2mu: HBAR_C * HBAR_C / (2.0 * mu * AMU_MEV),
        }
    }

    /// The kernel arithmetic reproduces nucrust-hf's CPU Hauser-Feshbach
    /// (compound formation and capture) for n + Fe-56.
    #[test]
    fn kernel_replica_matches_cpu_hauser_feshbach() {
        let tc = fe56_tc();
        let two_j_max = 20;
        let params = fe56_params(two_j_max);
        let (sigma_cn, sigma_gamma) = hf_summation_host(&tc, &params);

        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: params.q_value,
        };
        let nld = ConstantTemperature {
            temperature: params.nld_t,
            e0: params.nld_e0,
            a: params.nld_a,
        };
        let gsf = StandardLorentzian {
            e_gdr: params.gsf_e_gdr,
            gamma_gdr: params.gsf_gamma_gdr,
            sigma_gdr: params.gsf_sigma_gdr,
            m1_params: None,
        };
        let config = HfConfig {
            two_j_max,
            exit_channels: vec![Projectile::Gamma],
            ..HfConfig::default()
        };
        let calc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc,
            exit_particle_channels: vec![],
            nld: &nld,
            gsf: &gsf,
            config: &config,
            discrete_levels: None,
        };
        let cpu = hauser_feshbach(&calc).unwrap();

        for (e_idx, r) in cpu.iter().enumerate() {
            assert!(r.sigma_cn > 0.0 && r.sigma_channels[0] > 0.0);
            assert!(
                rel_diff(sigma_cn[e_idx], r.sigma_cn) < 1e-12,
                "E[{e_idx}]: sigma_cn kernel {} vs CPU {}",
                sigma_cn[e_idx],
                r.sigma_cn
            );
            assert!(
                rel_diff(sigma_gamma[e_idx], r.sigma_channels[0]) < 1e-12,
                "E[{e_idx}]: sigma_gamma kernel {} vs CPU {}",
                sigma_gamma[e_idx],
                r.sigma_channels[0]
            );
        }
    }

    /// sigma_CN = pi/k^2 * sum_{lj} (2j+1)/2 * T_lj for n + 0+ (requires the
    /// two-slot layout for l = 0 and j-I coupling with parity selection).
    #[test]
    fn kernel_replica_sigma_cn_equals_reaction_cross_section() {
        let tc = fe56_tc();
        let params = fe56_params(40);
        let (sigma_cn, _) = hf_summation_host(&tc, &params);
        let n_e = tc.energy.len();
        for (e_idx, &energy) in tc.energy.as_slice().iter().enumerate() {
            let mut weighted = 0.0;
            for l in 0..=tc.l_max as usize {
                for j_idx in 0..2_usize {
                    let two_j = 2 * l as i32 + 2 * j_idx as i32 - 1;
                    if two_j < 0 {
                        continue;
                    }
                    weighted +=
                        (two_j as f64 + 1.0) / 2.0 * tc.data[l * 2 * n_e + j_idx * n_e + e_idx];
                }
            }
            let k_sq = energy / params.hbar2_over_2mu;
            let sigma_r = std::f64::consts::PI / k_sq * weighted * 10.0;
            assert!(
                rel_diff(sigma_cn[e_idx], sigma_r) < 1e-12,
                "E = {energy}: sigma_cn {} vs sigma_R {}",
                sigma_cn[e_idx],
                sigma_r
            );
        }
    }

    /// Spin-0 projectile (alpha-like) on a 0+ target: only J = l with pi = (-1)^l,
    /// T_l read from slot 1.
    #[test]
    fn spin_zero_projectile_reads_slot_one() {
        let tc = TransmissionCoeffs {
            energy: EnergyGrid::from_values(vec![1.0]).unwrap(),
            l_max: 1,
            data: vec![0.7, 0.4, 0.9, 0.2], // slot 0 values must be ignored
        };
        let params = GpuHfParams {
            proj_two_s: 0,
            ..fe56_params(10)
        };
        let t = |two_j, parity| entrance_transmission(&tc.data, 1, 2, 0, two_j, parity, &params);
        assert_eq!(t(0, 1), 0.4);
        assert_eq!(t(2, -1), 0.2);
        assert_eq!(t(0, -1), 0.0);
        assert_eq!(t(2, 1), 0.0);
        assert_eq!(t(1, 1), 0.0);
    }
}
