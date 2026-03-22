//! Optical model potential implementations.
//!
//! Built-in models:
//! - Koning-Delaroche (2003): global nucleon OMP, 24 ≤ A ≤ 209
//! - McFadden-Satchler (1966): alpha-particle OMP
//! - Avrigeanu et al. (2014): improved alpha-particle OMP
//! - CustomOmp: user-defined parameters from TOML configuration

use nucrust_core::traits::OpticalPotential;
use nucrust_core::{Channel, CoreError, Projectile};
use num_complex::Complex64;

use crate::potential::*;

// ============================================================================
// Koning-Delaroche Global Nucleon OMP (2003)
// ============================================================================

/// Koning-Delaroche global nucleon optical model potential.
///
/// Nucl. Phys. A 713 (2003) 231-310.
/// Valid for n and p on targets 24 ≤ A ≤ 209, E < 200 MeV.
#[derive(Debug, Clone)]
pub struct KoningDelaroche;

/// Energy-dependent potential depths and geometry for KD.
struct KdPotential {
    v_real: f64,
    w_vol: f64,
    w_surf: f64,
    v_so: f64,
    w_so: f64,
    /// Real volume and imaginary volume share the same geometry in KD.
    rv: f64,
    av: f64,
    /// Surface (derivative) term geometry.
    rd: f64,
    ad: f64,
    /// Spin-orbit geometry.
    rso: f64,
    aso: f64,
    /// Coulomb radius parameter.
    rc: f64,
}

impl KoningDelaroche {
    /// Compute all KD potential depths and geometry for a given channel and energy.
    ///
    /// Parameters from Koning & Delaroche, Nucl. Phys. A 713 (2003) 231-310,
    /// Tables 14-15 (global parameterization). Cross-checked against the MARLEY
    /// C++ implementation and the IAEA RIPL-2 Fortran code (om-kd02.for).
    fn compute_potential(&self, e_cm: f64, channel: &Channel) -> KdPotential {
        let a = channel.target.a() as f64;
        let z = channel.target.z() as f64;
        let n = (channel.target.a() - channel.target.z()) as f64;
        let a13 = a.cbrt();
        let is_proton = channel.projectile == Projectile::Proton;

        // Asymmetry parameter
        let asym = (n - z) / a;

        // Fermi energy
        let e_f = if is_proton {
            -8.4075 + 0.01378 * a
        } else {
            -11.2814 + 0.02646 * a
        };

        // Energy relative to Fermi energy
        let de = e_cm - e_f;
        let de2 = de * de;
        let de3 = de2 * de;

        // =================================================================
        // Real volume depth: V = v1 * (1 - v2*de + v3*de² - v4*de³)
        //   + Vcbar * (v2 - 2*v3*de + 3*v4*de²)  [proton only]
        // =================================================================
        let (v1, v2, v3, v4);
        if is_proton {
            v1 = 59.30 + 21.0 * asym - 0.024 * a; // +asym for protons
            v2 = 0.007067 + 4.23e-6 * a;
            v3 = 1.729e-5 + 1.136e-8 * a;
            v4 = 7.0e-9; // same for n/p
        } else {
            v1 = 59.30 - 21.0 * asym - 0.024 * a; // -asym for neutrons
            v2 = 0.007228 - 1.48e-6 * a;
            v3 = 1.994e-5 - 2.0e-8 * a;
            v4 = 7.0e-9;
        }

        let mut v_real = v1 * (1.0 - v2 * de + v3 * de2 - v4 * de3);

        // Coulomb correction for protons (Lane isovector term)
        // Vcbar = 1.73 * Z / Rc (MeV), where the 1.73 factor absorbs e²
        if is_proton {
            let rc_param = 1.198 + 0.697 * a.powf(-2.0 / 3.0) + 12.994 * a.powf(-5.0 / 3.0);
            let r_c = rc_param * a13;
            let vcbar = 1.73 * z / r_c;
            v_real += vcbar * (v2 - 2.0 * v3 * de + 3.0 * v4 * de2);
        }

        // =================================================================
        // Imaginary volume depth: W_v = w1 * de² / (de² + w2²)
        // =================================================================
        let (w1, w2);
        if is_proton {
            w1 = 14.667 + 0.009629 * a;
            w2 = 73.55 + 0.0795 * a; // same as neutron w2
        } else {
            w1 = 12.195 + 0.0167 * a;
            w2 = 73.55 + 0.0795 * a;
        }
        let w_vol = (w1 * de2 / (de2 + w2 * w2)).max(0.0);

        // =================================================================
        // Imaginary surface depth: W_d = d1 * de² * exp(-d2*de) / (de² + d3²)
        // =================================================================
        let (d1, d2, d3);
        if is_proton {
            d1 = 16.0 + 16.0 * asym; // +asym for protons
        } else {
            d1 = 16.0 - 16.0 * asym; // -asym for neutrons
        }
        // d2 is the same for n/p
        d2 = 0.0180 + 0.003802 / (1.0 + ((a - 156.0) / 8.0).exp());
        d3 = 11.5; // same for n/p

        let w_surf = (d1 * de2 * (-d2 * de).exp() / (de2 + d3 * d3)).max(0.0);

        // =================================================================
        // Real spin-orbit: V_so = vso1 * exp(-vso2 * de)
        // =================================================================
        let vso1 = 5.922 + 0.0030 * a; // same for n/p
        let vso2 = 0.0040; // same for n/p
        let v_so = vso1 * (-vso2 * de).exp();

        // =================================================================
        // Imaginary spin-orbit: W_so = wso1 * de² / (de² + wso2²)
        // =================================================================
        let wso1 = -3.1; // same for n/p
        let wso2 = 160.0; // same for n/p
        let w_so = wso1 * de2 / (de2 + wso2 * wso2);

        // =================================================================
        // Geometry parameters
        // Real volume and imaginary volume share the same geometry.
        // =================================================================
        let rv = 1.3039 - 0.4054 / a13;
        let av = 0.6778 - 1.487e-4 * a;

        // Surface geometry: same radius as volume, different diffuseness for p
        let rd = 1.3424 - 0.01585 * a13;
        let ad = if is_proton {
            0.5187 + 5.205e-4 * a
        } else {
            0.5446 - 1.656e-4 * a
        };

        let rso = 1.1854 - 0.647 / a13;
        let aso = 0.59;

        let rc = if is_proton {
            1.198 + 0.697 * a.powf(-2.0 / 3.0) + 12.994 * a.powf(-5.0 / 3.0)
        } else {
            0.0
        };

        KdPotential {
            v_real,
            w_vol,
            w_surf,
            v_so,
            w_so,
            rv,
            av,
            rd,
            ad,
            rso,
            aso,
            rc,
        }
    }
}

impl OpticalPotential for KoningDelaroche {
    fn potential(&self, r: f64, e_cm: f64, l: u32, j: f64, channel: &Channel) -> Complex64 {
        let p = self.compute_potential(e_cm, channel);
        let ls = spin_orbit_factor(l, j);
        let r_safe = r.max(0.01);

        // Real and imaginary volume use the same WS shape factor (rv, av)
        let f_v = woods_saxon(r, p.rv, p.av, channel);
        // Surface uses its own geometry (rd, ad)
        let df_s = woods_saxon_deriv(r, p.rd, p.ad, channel);
        // Spin-orbit uses its own geometry (rso, aso)
        let f_so = woods_saxon_deriv(r, p.rso, p.aso, channel);
        let v_c = coulomb_potential(r, p.rc, channel);

        let real = -p.v_real * f_v + p.v_so * ls * f_so / r_safe + v_c;
        // Absorption: positive Im(V) causes outgoing flux to decrease in the
        // Fox-Goodwin Numerov integration used by this crate.
        let imag = p.w_vol * f_v + p.w_surf * df_s - p.w_so * ls * f_so / r_safe;

        Complex64::new(real, imag)
    }

    fn coulomb_radius(&self, channel: &Channel) -> f64 {
        let p = self.compute_potential(10.0, channel);
        p.rc * (channel.target.a() as f64).cbrt()
    }

    fn matching_radius(&self, channel: &Channel) -> f64 {
        let a13 = (channel.target.a() as f64).cbrt();
        // R_match = 1.25 * A^{1/3} + 4.5 * a (diffuseness)
        1.25 * a13 + 4.5 * 0.65
    }

    fn name(&self) -> &str {
        "Koning-Delaroche (2003)"
    }
}

// ============================================================================
// McFadden-Satchler Alpha OMP (1966)
// ============================================================================

/// McFadden-Satchler alpha-particle optical model potential.
///
/// Simple 4-parameter global OMP: V, W, r0, a.
/// Nucl. Phys. 84 (1966) 177-200.
#[derive(Debug, Clone)]
pub struct McFaddenSatchler;

impl OpticalPotential for McFaddenSatchler {
    fn potential(&self, r: f64, _e_cm: f64, _l: u32, _j: f64, channel: &Channel) -> Complex64 {
        // Standard parameters
        let v_real = 185.0; // MeV
        let w_imag = 25.0; // MeV
        let r0 = 1.40;
        let a = 0.52;
        let rc = 1.40;

        let f = woods_saxon(r, r0, a, channel);
        let v_c = coulomb_potential(r, rc, channel);

        Complex64::new(-v_real * f + v_c, w_imag * f) // positive Im = absorption
    }

    fn coulomb_radius(&self, channel: &Channel) -> f64 {
        1.40 * (channel.target.a() as f64).cbrt()
    }

    fn matching_radius(&self, channel: &Channel) -> f64 {
        let a13 = (channel.target.a() as f64).cbrt();
        1.40 * a13 + 4.5 * 0.52
    }

    fn name(&self) -> &str {
        "McFadden-Satchler (1966)"
    }
}

// ============================================================================
// Avrigeanu 2014 Alpha OMP
// ============================================================================

/// Avrigeanu et al. alpha-particle optical model potential (2014).
///
/// Improved energy-dependent alpha OMP.
/// Phys. Rev. C 90 (2014) 044612.
#[derive(Debug, Clone)]
pub struct Avrigeanu2014;

impl OpticalPotential for Avrigeanu2014 {
    fn potential(&self, r: f64, e_cm: f64, _l: u32, _j: f64, channel: &Channel) -> Complex64 {
        let a = channel.target.a() as f64;

        // Energy-dependent real depth
        let v_real = 101.1 + 6.051 * a.powf(1.0 / 3.0) - 0.248 * e_cm;
        let r_v = 1.245;
        let a_v = 0.817 - 0.0085 * a.cbrt();

        // Volume imaginary
        let w_vol = (12.64 + 0.365 * e_cm - 0.001 * e_cm * e_cm).max(0.0);
        let r_w = 1.570;
        let a_w = 0.692 - 0.020 * a.cbrt();

        // Surface imaginary
        let w_surf = (77.0 - 0.53 * e_cm).max(0.0);
        let r_d = 1.334;
        let a_d = 0.531;

        let rc = 1.245;

        let f_v = woods_saxon(r, r_v, a_v, channel);
        let f_w = woods_saxon(r, r_w, a_w, channel);
        let df_s = woods_saxon_deriv(r, r_d, a_d, channel);
        let v_c = coulomb_potential(r, rc, channel);

        let real = -v_real * f_v + v_c;
        let imag = w_vol * f_w + w_surf * df_s; // positive Im = absorption

        Complex64::new(real, imag)
    }

    fn coulomb_radius(&self, channel: &Channel) -> f64 {
        1.245 * (channel.target.a() as f64).cbrt()
    }

    fn matching_radius(&self, channel: &Channel) -> f64 {
        let a13 = (channel.target.a() as f64).cbrt();
        1.57 * a13 + 4.5 * 0.69
    }

    fn name(&self) -> &str {
        "Avrigeanu (2014)"
    }
}

// ============================================================================
// Custom OMP (user-defined parameters)
// ============================================================================

/// Custom optical model potential with user-defined parameters.
///
/// Supports volume real, volume imaginary, surface imaginary,
/// and spin-orbit terms, each with Woods-Saxon geometry.
#[derive(Debug, Clone)]
pub struct CustomOmp {
    /// Real volume depth (MeV).
    pub v_real: f64,
    /// Imaginary volume depth (MeV).
    pub w_vol: f64,
    /// Imaginary surface depth (MeV).
    pub w_surf: f64,
    /// Real spin-orbit depth (MeV).
    pub v_so: f64,
    /// Imaginary spin-orbit depth (MeV).
    pub w_so: f64,
    /// Real volume radius parameter r0 (fm).
    pub rv: f64,
    /// Real volume diffuseness (fm).
    pub av: f64,
    /// Imaginary volume radius parameter (fm).
    pub rw: f64,
    /// Imaginary volume diffuseness (fm).
    pub aw: f64,
    /// Surface radius parameter (fm).
    pub rd: f64,
    /// Surface diffuseness (fm).
    pub ad: f64,
    /// Spin-orbit radius parameter (fm).
    pub rso: f64,
    /// Spin-orbit diffuseness (fm).
    pub aso: f64,
    /// Coulomb radius parameter (fm).
    pub rc: f64,
}

impl CustomOmp {
    /// Creates a new `CustomOmp` with default parameter values.
    pub fn new() -> Self {
        Self {
            v_real: 50.0,
            w_vol: 0.0,
            w_surf: 10.0,
            v_so: 6.0,
            w_so: 0.0,
            rv: 1.25,
            av: 0.65,
            rw: 1.25,
            aw: 0.65,
            rd: 1.25,
            ad: 0.65,
            rso: 1.10,
            aso: 0.65,
            rc: 1.25,
        }
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.rv <= 0.0 || self.av <= 0.0 {
            return Err(CoreError::InvalidParameter {
                name: "rv/av",
                value: self.rv,
                reason: "radius and diffuseness must be positive",
            });
        }
        Ok(())
    }
}

impl Default for CustomOmp {
    fn default() -> Self {
        Self::new()
    }
}

impl OpticalPotential for CustomOmp {
    fn potential(&self, r: f64, _e_cm: f64, l: u32, j: f64, channel: &Channel) -> Complex64 {
        let ls = spin_orbit_factor(l, j);
        let r_safe = r.max(0.01);

        let f_v = woods_saxon(r, self.rv, self.av, channel);
        let f_w = woods_saxon(r, self.rw, self.aw, channel);
        let df_s = woods_saxon_deriv(r, self.rd, self.ad, channel);
        let f_so = woods_saxon_deriv(r, self.rso, self.aso, channel);
        let v_c = coulomb_potential(r, self.rc, channel);

        let real = -self.v_real * f_v + self.v_so * ls * f_so / r_safe + v_c;
        let imag = -self.w_vol * f_w - self.w_surf * df_s + self.w_so * ls * f_so / r_safe;

        Complex64::new(real, imag)
    }

    fn coulomb_radius(&self, channel: &Channel) -> f64 {
        self.rc * (channel.target.a() as f64).cbrt()
    }

    fn matching_radius(&self, channel: &Channel) -> f64 {
        let a13 = (channel.target.a() as f64).cbrt();
        self.rv.max(self.rw).max(self.rd) * a13 + 4.5 * self.av.max(self.aw).max(self.ad)
    }

    fn name(&self) -> &str {
        "Custom OMP"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::Nuclide;

    fn fe56_n() -> Channel {
        Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        }
    }

    fn fe56_p() -> Channel {
        Channel {
            projectile: Projectile::Proton,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        }
    }

    #[test]
    fn kd_neutron_potential_finite() {
        let kd = KoningDelaroche;
        let ch = fe56_n();
        // Test at r=2 fm inside the nucleus
        let v = kd.potential(2.0, 1.0, 0, 0.5, &ch);
        // Potential should be finite and have significant magnitude
        assert!(v.re.is_finite(), "V_real should be finite: {}", v.re);
        assert!(v.im.is_finite(), "W should be finite: {}", v.im);
        assert!(
            v.re.abs() > 1.0,
            "V_real should have significant magnitude: {}",
            v.re
        );
    }

    #[test]
    fn kd_proton_has_coulomb() {
        let kd = KoningDelaroche;
        let ch = fe56_p();
        // At large r, potential should be positive (Coulomb only)
        let v = kd.potential(20.0, 10.0, 0, 0.5, &ch);
        assert!(v.re > 0.0, "V at r=20 should be Coulomb-dominated");
    }

    #[test]
    fn kd_matching_radius_reasonable() {
        let kd = KoningDelaroche;
        let ch = fe56_n();
        let r_match = kd.matching_radius(&ch);
        // Should be around 7-10 fm for Fe-56
        assert!(r_match > 5.0 && r_match < 15.0, "r_match = {}", r_match);
    }

    #[test]
    fn mcfadden_satchler_alpha() {
        let ms = McFaddenSatchler;
        let ch = Channel {
            projectile: Projectile::Alpha,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        };
        let v = ms.potential(0.0, 20.0, 0, 0.0, &ch);
        assert!(v.re < -100.0); // Deep real well for alpha
        assert!(v.im > 0.0); // Absorptive (positive Im in our Numerov convention)
    }

    #[test]
    fn custom_omp_default_works() {
        let omp = CustomOmp::default();
        assert!(omp.validate().is_ok());
        let ch = fe56_n();
        let v = omp.potential(0.0, 10.0, 0, 0.5, &ch);
        assert!(v.re < 0.0);
    }
}
