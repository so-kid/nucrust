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
///
/// Uses the global parameterization ([`KdParameters::global`]); see
/// [`KoningDelarocheLocal`] for a nucleus-specific (local) parameter set.
#[derive(Debug, Clone)]
pub struct KoningDelaroche;

/// Parameters of the Koning-Delaroche (KD03) functional form.
///
/// With `dE = E - e_f` (E: the energy passed to [`OpticalPotential::potential`]):
///
/// - `V_v  = v1 (1 - v2 dE + v3 dE^2 - v4 dE^3)`
///   `+ Vc_bar v1 (v2 - 2 v3 dE + 3 v4 dE^2)` (protons, `Vc_bar = 1.73 Z / (rc A^1/3)`)
/// - `W_v  = w1 dE^2 / (dE^2 + w2^2)`
/// - `W_d  = d1 dE^2 exp(-d2 dE) / (dE^2 + d3^2)`
/// - `V_so = vso1 exp(-vso2 dE)`, `W_so = wso1 dE^2 / (dE^2 + wso2^2)`
///
/// with Woods-Saxon geometry `(rv, av)` for both volume terms, `(rd, ad)` for the surface
/// and `(rso, aso)` for the Thomas spin-orbit term (KD Eq. 2). The local sets of the TALYS
/// structure database (`optical/neutron/n-*.omp`) use the same form: line 1 gives `e_f`
/// (and `rc`), line 2 `rv av v1 v2 v3 w1 w2`, line 3 `rd ad d1 d2 d3`, line 4
/// `rso aso vso1 vso2 wso1 wso2`; `v4` is 7e-9 there.
#[derive(Debug, Clone, PartialEq)]
pub struct KdParameters {
    /// Fermi energy E_f (MeV).
    pub e_f: f64,
    /// Real volume depth parameters.
    pub v1: f64,
    /// Real volume depth parameters.
    pub v2: f64,
    /// Real volume depth parameters.
    pub v3: f64,
    /// Real volume depth parameters.
    pub v4: f64,
    /// Imaginary volume depth parameters.
    pub w1: f64,
    /// Imaginary volume depth parameters.
    pub w2: f64,
    /// Imaginary surface depth parameters.
    pub d1: f64,
    /// Imaginary surface depth parameters.
    pub d2: f64,
    /// Imaginary surface depth parameters.
    pub d3: f64,
    /// Real spin-orbit depth parameters.
    pub vso1: f64,
    /// Real spin-orbit depth parameters.
    pub vso2: f64,
    /// Imaginary spin-orbit depth parameters.
    pub wso1: f64,
    /// Imaginary spin-orbit depth parameters.
    pub wso2: f64,
    /// Volume radius parameter (fm).
    pub rv: f64,
    /// Volume diffuseness (fm).
    pub av: f64,
    /// Surface radius parameter (fm).
    pub rd: f64,
    /// Surface diffuseness (fm).
    pub ad: f64,
    /// Spin-orbit radius parameter (fm).
    pub rso: f64,
    /// Spin-orbit diffuseness (fm).
    pub aso: f64,
    /// Coulomb radius parameter (fm); only used for charged projectiles.
    pub rc: f64,
}

impl KdParameters {
    /// Global KD parameters for a nucleon channel.
    ///
    /// Koning & Delaroche, Nucl. Phys. A 713 (2003) 231-310, Tables 14-15. Cross-checked
    /// against the MARLEY C++ implementation and the IAEA RIPL-2 Fortran code (om-kd02.for).
    pub fn global(channel: &Channel) -> Self {
        let a = channel.target.a() as f64;
        let z = channel.target.z() as f64;
        let n = a - z;
        let a13 = a.cbrt();
        let is_proton = channel.projectile == Projectile::Proton;
        // Asymmetry parameter
        let asym = (n - z) / a;

        if is_proton {
            Self {
                e_f: -8.4075 + 0.01378 * a,
                v1: 59.30 + 21.0 * asym - 0.024 * a,
                v2: 0.007067 + 4.23e-6 * a,
                v3: 1.729e-5 + 1.136e-8 * a,
                v4: 7.0e-9,
                w1: 14.667 + 0.009629 * a,
                w2: 73.55 + 0.0795 * a,
                d1: 16.0 + 16.0 * asym,
                d2: 0.0180 + 0.003802 / (1.0 + ((a - 156.0) / 8.0).exp()),
                d3: 11.5,
                vso1: 5.922 + 0.0030 * a,
                vso2: 0.0040,
                wso1: -3.1,
                wso2: 160.0,
                rv: 1.3039 - 0.4054 / a13,
                av: 0.6778 - 1.487e-4 * a,
                rd: 1.3424 - 0.01585 * a13,
                ad: 0.5187 + 5.205e-4 * a,
                rso: 1.1854 - 0.647 / a13,
                aso: 0.59,
                rc: 1.198 + 0.697 * a.powf(-2.0 / 3.0) + 12.994 * a.powf(-5.0 / 3.0),
            }
        } else {
            Self {
                e_f: -11.2814 + 0.02646 * a,
                v1: 59.30 - 21.0 * asym - 0.024 * a,
                v2: 0.007228 - 1.48e-6 * a,
                v3: 1.994e-5 - 2.0e-8 * a,
                v4: 7.0e-9,
                w1: 12.195 + 0.0167 * a,
                w2: 73.55 + 0.0795 * a,
                d1: 16.0 - 16.0 * asym,
                d2: 0.0180 + 0.003802 / (1.0 + ((a - 156.0) / 8.0).exp()),
                d3: 11.5,
                vso1: 5.922 + 0.0030 * a,
                vso2: 0.0040,
                wso1: -3.1,
                wso2: 160.0,
                rv: 1.3039 - 0.4054 / a13,
                av: 0.6778 - 1.487e-4 * a,
                rd: 1.3424 - 0.01585 * a13,
                ad: 0.5446 - 1.656e-4 * a,
                rso: 1.1854 - 0.647 / a13,
                aso: 0.59,
                rc: 0.0,
            }
        }
    }

    /// Energy-dependent depths at energy `e` (MeV).
    fn depths(&self, e: f64, channel: &Channel) -> KdDepths {
        let de = e - self.e_f;
        let de2 = de * de;
        let de3 = de2 * de;

        let mut v_real = self.v1 * (1.0 - self.v2 * de + self.v3 * de2 - self.v4 * de3);
        // Coulomb correction for charged projectiles: V(E - Vc_bar) to first order,
        // Vc_bar * v1 (v2 - 2 v3 dE + 3 v4 dE^2), with Vc_bar = 1.73 Z / R_c (MeV).
        if channel.projectile.z() > 0 && self.rc > 0.0 {
            let r_c = self.rc * (channel.target.a() as f64).cbrt();
            let vcbar = 1.73 * channel.target.z() as f64 / r_c;
            v_real += vcbar * self.v1 * (self.v2 - 2.0 * self.v3 * de + 3.0 * self.v4 * de2);
        }

        KdDepths {
            v_real,
            w_vol: (self.w1 * de2 / (de2 + self.w2 * self.w2)).max(0.0),
            w_surf: (self.d1 * de2 * (-self.d2 * de).exp() / (de2 + self.d3 * self.d3)).max(0.0),
            v_so: self.vso1 * (-self.vso2 * de).exp(),
            w_so: self.wso1 * de2 / (de2 + self.wso2 * self.wso2),
        }
    }

    /// Complex potential in this crate's convention (absorption = positive `Im`).
    fn potential(&self, r: f64, e: f64, l: u32, j: f64, channel: &Channel) -> Complex64 {
        let d = self.depths(e, channel);
        let ls = spin_orbit_factor(l, j);

        // Real and imaginary volume use the same WS shape factor (rv, av)
        let f_v = woods_saxon(r, self.rv, self.av, channel);
        // Surface uses its own geometry (rd, ad)
        let df_s = woods_saxon_deriv(r, self.rd, self.ad, channel);
        // Spin-orbit: Thomas form (KD Eq. 2), U_so = (V_so + i W_so) * 2 lambda_pi^2
        // (1/r) df/dr <l.s>, attractive for j = l + 1/2.
        let f_so = thomas_spin_orbit(r, self.rso, self.aso, channel);
        let v_c = coulomb_potential(r, self.rc, channel);

        let real = -d.v_real * f_v + d.v_so * ls * f_so + v_c;
        // Absorption: positive Im(V) causes outgoing flux to decrease in the
        // Fox-Goodwin Numerov integration used by this crate, so every imaginary
        // term enters with the opposite sign of the physical U = ... - i W f.
        let imag = d.w_vol * f_v + d.w_surf * df_s - d.w_so * ls * f_so;

        Complex64::new(real, imag)
    }

    /// Radius beyond which every term is negligible.
    fn matching_radius(&self, channel: &Channel) -> f64 {
        woods_saxon_tail_radius(self.rv, self.av, channel)
            .max(woods_saxon_tail_radius(self.rd, self.ad, channel))
            .max(woods_saxon_tail_radius(self.rso, self.aso, channel))
    }
}

/// Energy-dependent KD depths (MeV).
struct KdDepths {
    v_real: f64,
    w_vol: f64,
    w_surf: f64,
    v_so: f64,
    w_so: f64,
}

impl OpticalPotential for KoningDelaroche {
    fn potential(&self, r: f64, e_cm: f64, l: u32, j: f64, channel: &Channel) -> Complex64 {
        KdParameters::global(channel).potential(r, e_cm, l, j, channel)
    }

    fn coulomb_radius(&self, channel: &Channel) -> f64 {
        KdParameters::global(channel).rc * (channel.target.a() as f64).cbrt()
    }

    fn matching_radius(&self, channel: &Channel) -> f64 {
        KdParameters::global(channel).matching_radius(channel)
    }

    fn name(&self) -> &str {
        "Koning-Delaroche (2003)"
    }
}

/// Koning-Delaroche potential with an explicit (e.g. nucleus-specific) parameter set.
///
/// TALYS uses such local KD03-form sets by default (`localomp y`) where its structure
/// database has one, e.g. for n + Fe-56 `e_f = -9.42`, `v1 = 56.8`, `rv = 1.186`, ... The
/// parameters are used as given, for whatever channel the potential is evaluated on.
#[derive(Debug, Clone, PartialEq)]
pub struct KoningDelarocheLocal {
    /// The parameter set.
    pub params: KdParameters,
}

impl KoningDelarocheLocal {
    /// Wraps a parameter set.
    pub fn new(params: KdParameters) -> Self {
        Self { params }
    }
}

impl OpticalPotential for KoningDelarocheLocal {
    fn potential(&self, r: f64, e: f64, l: u32, j: f64, channel: &Channel) -> Complex64 {
        self.params.potential(r, e, l, j, channel)
    }

    fn coulomb_radius(&self, channel: &Channel) -> f64 {
        self.params.rc * (channel.target.a() as f64).cbrt()
    }

    fn matching_radius(&self, channel: &Channel) -> f64 {
        self.params.matching_radius(channel)
    }

    fn name(&self) -> &str {
        "Koning-Delaroche (local parameters)"
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
        woods_saxon_tail_radius(1.40, 0.52, channel)
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
        woods_saxon_tail_radius(1.245, 0.817 - 0.0085 * a13, channel)
            .max(woods_saxon_tail_radius(1.570, 0.692 - 0.020 * a13, channel))
            .max(woods_saxon_tail_radius(1.334, 0.531, channel))
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

        let f_v = woods_saxon(r, self.rv, self.av, channel);
        let f_w = woods_saxon(r, self.rw, self.aw, channel);
        let df_s = woods_saxon_deriv(r, self.rd, self.ad, channel);
        let f_so = thomas_spin_orbit(r, self.rso, self.aso, channel);
        let v_c = coulomb_potential(r, self.rc, channel);

        let real = -self.v_real * f_v + self.v_so * ls * f_so + v_c;
        // Same convention as `KoningDelaroche`: positive Im(V) is absorptive here.
        let imag = self.w_vol * f_w + self.w_surf * df_s - self.w_so * ls * f_so;

        Complex64::new(real, imag)
    }

    fn coulomb_radius(&self, channel: &Channel) -> f64 {
        self.rc * (channel.target.a() as f64).cbrt()
    }

    fn matching_radius(&self, channel: &Channel) -> f64 {
        woods_saxon_tail_radius(self.rv, self.av, channel)
            .max(woods_saxon_tail_radius(self.rw, self.aw, channel))
            .max(woods_saxon_tail_radius(self.rd, self.ad, channel))
            .max(woods_saxon_tail_radius(self.rso, self.aso, channel))
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
    fn kd_matching_radius_beyond_potential_tail() {
        let kd = KoningDelaroche;
        let ch = fe56_n();
        let r_match = kd.matching_radius(&ch);
        // ~27 fm for Fe-56 (R_v ~ 4.6 fm + 33 a_v): far past the old 7.7 fm, where the
        // Woods-Saxon tail is still ~1e-2 of its depth.
        assert!(r_match > 25.0 && r_match < 30.0, "r_match = {}", r_match);
        for l in [0, 5] {
            let v = kd.potential(r_match, 1.0, l, l as f64 + 0.5, &ch);
            assert!(v.norm() < 1e-12, "|V(r_match)| = {} MeV", v.norm());
        }
    }

    #[test]
    fn kd_spin_orbit_is_attractive_thomas_form() {
        // At r = R_so the normalized derivative g = 1, so the j = l +- 1/2 splitting is
        // U(l+1/2) - U(l-1/2) = -(V_so + i W_so) (2l+1)/2 / (a_so R_so)   (KD Eq. 2).
        let kd = KoningDelaroche;
        let ch = fe56_n();
        let e = 5.0;
        let params = KdParameters::global(&ch);
        let p = params.depths(e, &ch);
        let aso = params.aso;
        let r_so = params.rso * 56f64.cbrt();
        for l in [1u32, 4] {
            let lf = l as f64;
            let up = kd.potential(r_so, e, l, lf + 0.5, &ch);
            let down = kd.potential(r_so, e, l, lf - 0.5, &ch);
            let split = up - down;
            let scale = (2.0 * lf + 1.0) / 2.0 / (aso * r_so);
            assert!(split.re < 0.0, "j = l+1/2 must be more attractive");
            assert!(
                (split.re - (-p.v_so * scale)).abs() < 1e-12,
                "l={l}: Re split {} vs {}",
                split.re,
                -p.v_so * scale
            );
            // Imaginary part: this crate stores -Im(U) (absorption > 0).
            assert!(
                (split.im - (p.w_so * scale)).abs() < 1e-12,
                "l={l}: Im split {} vs {}",
                split.im,
                p.w_so * scale
            );
        }
    }

    #[test]
    fn kd_proton_coulomb_correction_is_shifted_energy() {
        // KD Eq. (7): V_p(E) = V(E) + Vc_bar v1 (v2 - 2 v3 dE + 3 v4 dE^2) = V(E - Vc_bar)
        // to first order, i.e. V(E) - Vc_bar dV/dE with the full depth derivative.
        let ch = fe56_p();
        let params = KdParameters::global(&ch);
        let vcbar = 1.73 * 26.0 / (params.rc * 56f64.cbrt());
        let no_coulomb = KdParameters {
            rc: 0.0,
            ..params.clone()
        };
        for e in [1.0, 10.0, 50.0] {
            let v = params.depths(e, &ch).v_real;
            let eps = 1e-4;
            let dvde = (no_coulomb.depths(e + eps, &ch).v_real
                - no_coulomb.depths(e - eps, &ch).v_real)
                / (2.0 * eps);
            let expected = no_coulomb.depths(e, &ch).v_real - vcbar * dvde;
            assert!((v - expected).abs() < 1e-6, "E={e}: {v} vs {expected}");
            // ~ +3 MeV for Fe-56, not the ~0.06 MeV of a correction without v1.
            assert!(v - no_coulomb.depths(e, &ch).v_real > 2.0);
        }
    }

    #[test]
    fn kd_local_with_global_parameters_matches_global() {
        let ch = fe56_n();
        let local = KoningDelarocheLocal::new(KdParameters::global(&ch));
        let kd = KoningDelaroche;
        assert_eq!(local.matching_radius(&ch), kd.matching_radius(&ch));
        for r in [0.5, 4.0, 6.0] {
            for (l, j) in [(0, 0.5), (3, 2.5), (3, 3.5)] {
                assert_eq!(
                    local.potential(r, 2.0, l, j, &ch),
                    kd.potential(r, 2.0, l, j, &ch)
                );
            }
        }
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
    fn custom_omp_is_absorptive() {
        // Same imaginary-part convention as KD: positive surface/volume depths absorb flux.
        let omp = CustomOmp::default();
        let ch = fe56_n();
        let kd = KoningDelaroche;
        let big_r = omp.rd * 56f64.cbrt();
        assert!(omp.potential(big_r, 5.0, 0, 0.5, &ch).im > 0.0);
        assert!(kd.potential(big_r, 5.0, 0, 0.5, &ch).im > 0.0);
        let s = crate::numerov::numerov_integrate(
            &omp,
            &ch,
            5.0,
            0,
            0.5,
            &nucrust_core::backend::NumerovConfig::default(),
        )
        .unwrap();
        assert!(s.norm() < 1.0, "|S| = {} must be < 1", s.norm());
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
