//! Accuracy validation against TALYS-2.25 golden reference data (SRS ACC-02 / ACC-03).
//!
//! Reference data: `tests/reference_data/talys/{fe56,u238}/` (TALYS-2.25, git 2c5d15f68f,
//! default physics plus `outinverse y`, `transpower 20`, `transeps 1e-14`).
//!
//! Each quantity has two tests:
//! - a *regression* test asserting the current maximum relative error does not grow
//!   (ratchet bound = measured maximum rounded up to 2 significant figures, x1.1), and
//! - an *acceptance* test at the SRS target (relative error < 1e-6), `#[ignore]`d with the
//!   measured state until the model differences listed below are resolved.
//!
//! Run `cargo test -p nucrust --test golden_talys -- --include-ignored --nocapture` to see
//! the comparison tables.
//!
//! # Frames and energies
//!
//! - TALYS transmission coefficients are computed on its CM emission grid
//!   (0.001, 0.002, 0.005, 0.01, ... MeV) and printed at the LAB energy
//!   `E_lab = E_cm (M + m_n) / M`. nucrust takes CM energies, so printed energies are
//!   converted back with the TALYS masses (target atomic mass and neutron mass from the
//!   "BASIC REACTION PARAMETERS" block of `raw/output.gz`). The factor is checked against
//!   `metadata.json` (`transmission_E_lab_over_E_grid`) and the converted energies are
//!   snapped to 4 significant digits (all TALYS grid points are round numbers; the snap is
//!   asserted to move each energy by < 2e-6 relative, i.e. within print precision).
//! - Cross-section tables are given at LAB incident energies; nucrust is evaluated at
//!   `E_cm = E_lab M / (M + m_n)`. Both codes use the CM wave number in `pi / k^2`.
//!
//! # Known sources of the measured errors (see task.md T-2A.13 / T-2B.14)
//!
//! ACC-02 (ranked by impact, from an off-line study with a custom potential/solver):
//! 1. Spin-orbit term of `KoningDelaroche`: sign reversed (j = l+1/2 is made repulsive) and
//!    strength off by 1/a_so (Thomas form `2 lambda_pi^2 V_so (1/r) df/dr l.s` = `-(V_so/a_so)
//!    l.s g(r)/r` with g the normalized derivative). Visible as swapped T(l-1/2)/T(l+1/2).
//! 2. Matching radius `1.25 A^1/3 + 2.9 fm` (7.7 fm for Fe-56) truncates the Woods-Saxon
//!    tail: high-l T are up to ~100% low. A ~20 fm radius converges.
//! 3. TALYS runs ECIS with relativistic kinematics (`relativistic y`); this changes k^2 by
//!    ~E/(2 m c^2) and high-l T by ~l times that (~10% at 20 MeV, l ~ 15).
//! 4. TALYS uses `localomp y`: Fe-56 and U-238 take the *local* KD03-form parameter sets of
//!    `structure/optical/neutron/n-{Fe,U}.omp` (Fe-56: Ef = -9.42, v1 = 56.8, rv = 1.186,
//!    ...), nucrust the *global* KD set (U-238 is outside its A <= 209 range). TALYS does not
//!    use coupled channels for neutrons here (`rotational` unset), so U-238 is spherical too.
//! 5. The log-derivative `(R_N - 1)/h` used to match the Numerov solution is O(h): ~1.5% on
//!    T_0 at h = 0.05 fm. Two-point matching removes it.
//! 6. ECIS evaluates the KD depths at the LAB energy (nucrust: E_cm); nucrust's reduced mass
//!    uses A instead of the atomic mass (~2e-5 on k^2).
//!
//! With 1-5 fixed (local parameters, relativistic kinematics, 20 fm, two-point matching)
//! the Fe-56 study reached max 6.7e-3 / median 3.4e-3, with a remaining ~-0.5% offset on
//! T_0 already at 1 keV that is not explained yet (dump the TALYS potential with
//! `outomp y` to compare depths). TALYS computes in single precision and prints 7
//! significant digits, so ~1e-6 is also the resolution floor of this golden data.
//!
//! ACC-03a: sigma_R inherits the T_lj errors (at keV energies nucrust/TALYS sigma_R ratio
//! equals the T_0 ratio to ~2e-4, confirming the same CM `pi/k^2` convention).
//!
//! ACC-03b (capture), model differences: TALYS uses ldmodel 1 (CT + Fermi gas fitted to
//! discrete levels and D0), SMLO E1 (strength 9) + M1 with upbend, Moldauer width
//! fluctuations below 11.2 MeV, all particle channels (p, alpha, (n,2n), ...),
//! pre-equilibrium and the multi-step cascade (`sigma_res` = A+1 production). nucrust here:
//! CT level density tuned to D0(57Fe) (as in `e2e_pipeline.rs`), SLO E1 with constant
//! M1/E2, no WFC (`wfc_model` is not applied by `hauser_feshbach`), gamma + neutron
//! (compound elastic + inelastic to the first 11 Fe-56 levels + CT continuum) only.
//! Below E_cm = 10 keV the neutron exit channel returns 0 because
//! `exit_particle_continuum_transmission` drops emission energies < `MIN_EMISSION_ENERGY`
//! (10 keV), which removes compound elastic and makes sigma_gamma = sigma_CN (a factor
//! ~500 too large); the regression test therefore also ratchets E_cm >= 10 keV separately.
//! The 57Fe CT (E0 = 1.94 MeV) supplies gamma final states over the whole 0..Sn range,
//! including 0 <= U < E0 (E0 is a shift, not a threshold).

use std::path::{Path, PathBuf};

use nucrust::cpu_backend::cpu_transmission_coeffs;
use nucrust_core::backend::NumerovConfig;
use nucrust_core::{Channel, EnergyGrid, Nuclide, Parity, Projectile, TransmissionCoeffs};

// ---------------------------------------------------------------------------------------
// Reference data access
// ---------------------------------------------------------------------------------------

/// Neutron mass (amu) as printed by TALYS-2.25.
const TALYS_M_NEUTRON: f64 = 1.008665;
/// Fe-56 atomic mass (amu) as printed by TALYS-2.25 (`raw/output.gz`).
const TALYS_M_FE56: f64 = 55.934936;
/// U-238 atomic mass (amu) as printed by TALYS-2.25 (`raw/output.gz`).
const TALYS_M_U238: f64 = 238.050787;

/// Values of |T_TALYS| at or below this are skipped in ACC-02: TALYS works in single
/// precision with an absolute ECIS noise floor, so tiny T are not meaningful relative to 1e-6.
const T_SKIP_BELOW: f64 = 1e-8;

fn talys_dir(nuclide: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/talys")
        .join(nuclide);
    assert!(
        dir.is_dir(),
        "TALYS reference data directory not found: {}",
        dir.display()
    );
    dir
}

/// Read a whitespace-separated numeric table, skipping `#` comments. Panics on any error.
fn read_table(path: &Path, n_cols: usize) -> Vec<Vec<f64>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let rows: Vec<Vec<f64>> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let row: Vec<f64> = l
                .split_whitespace()
                .map(|t| {
                    t.parse()
                        .unwrap_or_else(|e| panic!("{}: bad number {t:?}: {e}", path.display()))
                })
                .collect();
            assert_eq!(row.len(), n_cols, "{}: bad row {l:?}", path.display());
            row
        })
        .collect();
    assert!(!rows.is_empty(), "{}: no data rows", path.display());
    rows
}

/// LAB/CM factor (M + m_n)/M from the TALYS masses, cross-checked against metadata.json.
fn lab_over_cm(nuclide: &str, m_target: f64) -> f64 {
    let ratio = (m_target + TALYS_M_NEUTRON) / m_target;
    let meta_path = talys_dir(nuclide).join("metadata.json");
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&meta_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", meta_path.display())),
    )
    .unwrap_or_else(|e| panic!("{}: invalid JSON: {e}", meta_path.display()));
    let meta_ratio = meta["transmission"]["transmission_E_lab_over_E_grid"]
        .as_f64()
        .expect("metadata.json: transmission.transmission_E_lab_over_E_grid missing");
    assert!(
        (ratio - meta_ratio).abs() < 1e-6,
        "{nuclide}: (M+m)/M = {ratio} disagrees with metadata {meta_ratio}"
    );
    ratio
}

/// Round to `digits` significant digits.
fn round_sig(x: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits - 1 - x.abs().log10().floor() as i32);
    (x * scale).round() / scale
}

fn neutron_channel(z: u16, a: u16, q_value: f64) -> Channel {
    Channel {
        projectile: Projectile::Neutron,
        target: Nuclide::new(z, a).unwrap(),
        q_value,
    }
}

fn numerov_config() -> NumerovConfig {
    NumerovConfig {
        max_l: 25,
        convergence_tl: 1e-12,
        ..NumerovConfig::default()
    }
}

// ---------------------------------------------------------------------------------------
// Comparison bookkeeping
// ---------------------------------------------------------------------------------------

struct Point {
    e: f64,
    what: String,
    nucrust: f64,
    talys: f64,
}

impl Point {
    fn rel(&self) -> f64 {
        (self.nucrust - self.talys).abs() / self.talys.abs()
    }
}

struct Summary {
    max_rel: f64,
    worst: String,
    median_rel: f64,
    n: usize,
}

fn summarize(title: &str, points: &[Point], print_all: bool) -> Summary {
    assert!(!points.is_empty(), "{title}: no comparison points");
    for p in points {
        assert!(
            p.nucrust.is_finite(),
            "{title}: non-finite nucrust value at E={} {}",
            p.e,
            p.what
        );
    }
    let worst = points
        .iter()
        .max_by(|a, b| a.rel().total_cmp(&b.rel()))
        .unwrap();
    let mut rels: Vec<f64> = points.iter().map(Point::rel).collect();
    rels.sort_by(f64::total_cmp);
    let median_rel = rels[rels.len() / 2];

    println!("\n=== {title} ===");
    println!(
        "{:>12} {:>14} {:>14} {:>14} {:>10}",
        "E_cm [MeV]", "quantity", "nucrust", "TALYS", "rel.err"
    );
    let row = |p: &Point| {
        println!(
            "{:>12.6} {:>14} {:>14.6e} {:>14.6e} {:>10.3e}",
            p.e,
            p.what,
            p.nucrust,
            p.talys,
            p.rel()
        )
    };
    if print_all {
        points.iter().for_each(row);
    } else {
        // One line per energy: the worst point at that energy.
        let mut i = 0;
        while i < points.len() {
            let e = points[i].e;
            let mut j = i;
            let mut w = &points[i];
            while j < points.len() && points[j].e == e {
                if points[j].rel() > w.rel() {
                    w = &points[j];
                }
                j += 1;
            }
            row(w);
            i = j;
        }
    }
    let worst_desc = format!("E_cm={:.6} MeV {}", worst.e, worst.what);
    println!(
        "--> {} points; max rel.err {:.3e} at {}; median {:.3e}",
        points.len(),
        worst.rel(),
        worst_desc,
        median_rel
    );
    Summary {
        max_rel: worst.rel(),
        worst: worst_desc,
        median_rel,
        n: points.len(),
    }
}

fn assert_within(title: &str, s: &Summary, bound: f64) {
    assert!(
        s.max_rel <= bound,
        "{title}: max rel.err {:.3e} (at {}; median {:.3e}, {} points) exceeds {:.3e}",
        s.max_rel,
        s.worst,
        s.median_rel,
        s.n,
        bound
    );
}

// ---------------------------------------------------------------------------------------
// ACC-02: neutron transmission coefficients T_lj
// ---------------------------------------------------------------------------------------

/// Compare nucrust T_lj (KD global, Numerov) with TALYS T(L-1/2,L), T(L+1/2,L).
fn acc02_points(nuclide: &str, z: u16, a: u16, m_target: f64) -> Vec<Point> {
    let path = talys_dir(nuclide).join(format!("{nuclide}_transmission_jsplit.dat"));
    // Columns: E_lab, L, T(L-1/2,L), T(L+1/2,L), T_avg(L)
    let rows = read_table(&path, 5);
    let factor = lab_over_cm(nuclide, m_target);

    // CM grid energies, in file order (ascending).
    let mut e_cm: Vec<f64> = Vec::new();
    let mut e_lab_seen: Vec<f64> = Vec::new();
    for r in &rows {
        if e_lab_seen.last() != Some(&r[0]) {
            let raw = r[0] / factor;
            let snapped = round_sig(raw, 4);
            assert!(
                (snapped - raw).abs() / raw < 2e-6,
                "{nuclide}: E_lab {} -> E_cm {raw} is not a round TALYS grid energy",
                r[0]
            );
            e_lab_seen.push(r[0]);
            e_cm.push(snapped);
        }
    }
    assert!(e_cm.len() >= 60, "{nuclide}: only {} energies", e_cm.len());
    assert!(
        (e_cm[0] - 0.001).abs() < 1e-12,
        "first CM energy {}",
        e_cm[0]
    );

    let ch = neutron_channel(z, a, 0.0);
    let grid = EnergyGrid::from_values(e_cm.clone()).unwrap();
    let tc = cpu_transmission_coeffs(&ch, &grid, &numerov_config()).unwrap();

    let mut points = Vec::new();
    for r in &rows {
        let e_idx = e_lab_seen.iter().position(|&e| e == r[0]).unwrap();
        let l = r[1] as u32;
        assert_eq!(r[1], l as f64, "non-integer L in {}", path.display());
        for (j_idx, &t_talys) in [r[2], r[3]].iter().enumerate() {
            if l == 0 && j_idx == 0 {
                assert_eq!(t_talys, 0.0, "TALYS T(-1/2, 0) should be 0");
                continue;
            }
            if t_talys <= T_SKIP_BELOW {
                continue;
            }
            let t_nucrust = if l <= tc.l_max {
                tc.get(l, j_idx, e_idx)
            } else {
                0.0
            };
            let two_j = 2 * l as i32 + 2 * j_idx as i32 - 1;
            points.push(Point {
                e: e_cm[e_idx],
                what: format!("l={l} j={two_j}/2"),
                nucrust: t_nucrust,
                talys: t_talys,
            });
        }
    }
    points
}

/// Measured (1339 points): max 1.342 at E_cm = 4.5 MeV, l=4 j=9/2 (spin-orbit reversal);
/// median 3.5e-1; T_0 max 3.6e-2. Bound = 1.4 x 1.1.
const ACC02_FE56_RATCHET: f64 = 1.54;
/// Measured (1616 points): max 5.660 at E_cm = 0.9 MeV, l=7 j=15/2; median 4.2e-1.
/// Bound = 5.7 x 1.1.
const ACC02_U238_RATCHET: f64 = 6.27;

#[test]
fn acc02_fe56_tlj_regression() {
    let pts = acc02_points("fe56", 26, 56, TALYS_M_FE56);
    let s = summarize(
        "ACC-02 Fe-56 n T_lj vs TALYS (worst per energy)",
        &pts,
        false,
    );
    assert_within("ACC-02 Fe-56 regression", &s, ACC02_FE56_RATCHET);
}

#[test]
#[ignore = "ACC-02 not yet met: measured max rel err 1.3e0 at E_cm=4.5 MeV l=4 j=9/2 (median 3.5e-1); \
            see module docs: KD spin-orbit sign/strength, matching radius, relativistic kinematics, local OMP"]
fn acc02_fe56_tlj_srs_target() {
    let pts = acc02_points("fe56", 26, 56, TALYS_M_FE56);
    let s = summarize("ACC-02 Fe-56 n T_lj vs TALYS (all points)", &pts, true);
    assert_within("ACC-02 Fe-56 SRS target", &s, 1e-6);
}

#[test]
fn acc02_u238_tlj_regression() {
    let pts = acc02_points("u238", 92, 238, TALYS_M_U238);
    let s = summarize(
        "ACC-02 U-238 n T_lj vs TALYS (worst per energy)",
        &pts,
        false,
    );
    assert_within("ACC-02 U-238 regression", &s, ACC02_U238_RATCHET);
}

#[test]
#[ignore = "ACC-02 not yet met: measured max rel err 5.7e0 at E_cm=0.9 MeV l=7 j=15/2 (median 4.2e-1); \
            see module docs: as Fe-56, plus TALYS local U-238 OMP vs global KD outside A<=209"]
fn acc02_u238_tlj_srs_target() {
    let pts = acc02_points("u238", 92, 238, TALYS_M_U238);
    let s = summarize("ACC-02 U-238 n T_lj vs TALYS (all points)", &pts, true);
    assert_within("ACC-02 U-238 SRS target", &s, 1e-6);
}

// ---------------------------------------------------------------------------------------
// ACC-03: Fe-56 reaction and capture cross sections
// ---------------------------------------------------------------------------------------

/// Fe-56 cross-section table: (E_lab, sigma_res, sigma_reac) in (MeV, mb).
fn fe56_cross_sections() -> Vec<(f64, f64, f64)> {
    let path = talys_dir("fe56").join("fe56_cross_sections.dat");
    // Columns: E_lab sigma_ng_binary sigma_res sigma_nonel sigma_el sigma_tot sigma_compel
    //          sigma_shapeel sigma_reac
    let rows = read_table(&path, 9);
    assert_eq!(rows.len(), 117, "{}: expected 117 energies", path.display());
    rows.iter().map(|r| (r[0], r[2], r[8])).collect()
}

struct Fe56Hf {
    e_cm: Vec<f64>,
    sigma_cn: Vec<f64>,
    sigma_gamma: Vec<f64>,
}

/// Fe-56 + n with the closest setup nucrust supports (see the module docs).
fn fe56_hf() -> Fe56Hf {
    use nucrust_hf::gsf::StandardLorentzian;
    use nucrust_hf::hf::{self, DiscreteLevelInfo, DiscreteLevels, ExitChannelData, HfCalculation};
    use nucrust_hf::nld::ConstantTemperature;

    let xs = fe56_cross_sections();
    let m_over_m_plus_n = TALYS_M_FE56 / (TALYS_M_FE56 + TALYS_M_NEUTRON);
    let e_cm: Vec<f64> = xs
        .iter()
        .map(|&(e_lab, _, _)| e_lab * m_over_m_plus_n)
        .collect();

    // Sn(57Fe) = Q(n,g) as printed by TALYS.
    let ch = neutron_channel(26, 56, 7.64617);
    let grid = EnergyGrid::from_values(e_cm.clone()).unwrap();
    let cfg = numerov_config();
    let tc = cpu_transmission_coeffs(&ch, &grid, &cfg).unwrap();

    // Neutron emission T on a separate CM grid covering 1 keV .. 21 MeV.
    let exit_ch = neutron_channel(26, 56, 0.0);
    let exit_grid = EnergyGrid::logarithmic(1e-3, 21.0, 120).unwrap();
    let tc_exit: TransmissionCoeffs = cpu_transmission_coeffs(&exit_ch, &exit_grid, &cfg).unwrap();

    // Compound nucleus 57Fe: CT tuned to D0(57Fe) ~ 25 keV (as in e2e_pipeline.rs).
    let nld_cn = ConstantTemperature {
        temperature: 0.88,
        e0: 1.94,
        a: 6.21,
    };
    // Residual 56Fe: CT with T = 1.0 MeV, E0 fixed so that N(E) = exp((E - E0)/T) gives
    // the 11 discrete levels below E_complete = 3.45 MeV (TALYS discrete level file).
    let nld_fe56 = ConstantTemperature {
        temperature: 1.0,
        e0: 3.45 - 11f64.ln(),
        a: 6.0,
    };
    let lev = |energy: f64, spin: f64, parity: Parity| DiscreteLevelInfo {
        energy,
        spin,
        parity,
    };
    let p = Parity::Positive;
    let fe56_levels = DiscreteLevels {
        levels: vec![
            lev(0.0, 0.0, p),
            lev(0.846778, 2.0, p),
            lev(2.085105, 4.0, p),
            lev(2.657589, 2.0, p),
            lev(2.941500, 0.0, p),
            lev(2.959972, 2.0, p),
            lev(3.120110, 1.0, p),
            lev(3.122970, 4.0, p),
            lev(3.369950, 2.0, p),
            lev(3.388550, 6.0, p),
            lev(3.445348, 3.0, p),
        ],
        e_complete: 3.45,
    };
    let gsf = StandardLorentzian {
        e_gdr: 16.36,
        gamma_gdr: 4.58,
        sigma_gdr: 136.0,
        m1_params: None,
    };
    let hf_config = hf::HfConfig {
        two_j_max: 51,
        exit_channels: vec![Projectile::Gamma, Projectile::Neutron],
        ..hf::HfConfig::default()
    };
    let calc = HfCalculation {
        entrance: &ch,
        tc_entrance: &tc,
        exit_particle_channels: vec![ExitChannelData {
            channel: &exit_ch,
            tc: &tc_exit,
            separation_energy: Some(ch.q_value),
            daughter_nld: Some(&nld_fe56),
            daughter_nuclide: Some(exit_ch.target),
            daughter_discrete: Some(&fe56_levels),
        }],
        nld: &nld_cn,
        gsf: &gsf,
        config: &hf_config,
        discrete_levels: None,
    };
    let r = hf::hauser_feshbach(&calc).unwrap();
    assert_eq!(r.len(), e_cm.len());
    Fe56Hf {
        e_cm,
        sigma_cn: r.iter().map(|x| x.sigma_cn).collect(),
        sigma_gamma: r.iter().map(|x| x.sigma_channels[0]).collect(),
    }
}

fn acc03a_points() -> Vec<Point> {
    let xs = fe56_cross_sections();
    let hf = fe56_hf();
    xs.iter()
        .zip(hf.e_cm.iter().zip(&hf.sigma_cn))
        .map(|(&(_, _, sigma_reac), (&e, &s))| Point {
            e,
            what: "sigma_R [mb]".into(),
            nucrust: s,
            talys: sigma_reac,
        })
        .collect()
}

fn acc03b_points() -> Vec<Point> {
    let xs = fe56_cross_sections();
    let hf = fe56_hf();
    xs.iter()
        .zip(hf.e_cm.iter().zip(&hf.sigma_gamma))
        .map(|(&(_, sigma_res, _), (&e, &s))| Point {
            e,
            what: "sigma_ng [mb]".into(),
            nucrust: s,
            talys: sigma_res,
        })
        .collect()
}

/// Measured (117 points): max 4.214e-2 at E_cm = 7.98 MeV; median 3.0e-2 (3.6% at 1 keV,
/// following T_0). Bound = 4.3e-2 x 1.1.
const ACC03A_RATCHET: f64 = 4.73e-2;
/// Measured (117 points): max 4.949e2 at E_cm = 9.82 keV (compound elastic lost below
/// 10 keV, see module docs); median 6.3e-1. Bound = 5.0e2 x 1.1.
const ACC03B_RATCHET: f64 = 550.0;
/// Measured for E_cm >= 10 keV (90 points): max 1.085e1 at E_cm = 19.6 MeV (missing
/// (n,p)/(n,2n)/pre-equilibrium competition); median 5.9e-1, 0.46-0.65 (nucrust low) from
/// 10 keV to 2 MeV. Bound = 11 x 1.1.
const ACC03B_ABOVE_10KEV_RATCHET: f64 = 12.1;

#[test]
fn acc03a_fe56_reaction_xs_regression() {
    let s = summarize(
        "ACC-03a Fe-56 sigma_R vs TALYS sigma_reac",
        &acc03a_points(),
        true,
    );
    assert_within("ACC-03a regression", &s, ACC03A_RATCHET);
}

#[test]
#[ignore = "ACC-03a not yet met: measured max rel err 4.2e-2 at E_cm=7.98 MeV (median 3.0e-2); \
            inherits the ACC-02 T_lj errors, see module docs"]
fn acc03a_fe56_reaction_xs_srs_target() {
    let s = summarize(
        "ACC-03a Fe-56 sigma_R vs TALYS sigma_reac",
        &acc03a_points(),
        true,
    );
    assert_within("ACC-03a SRS target", &s, 1e-6);
}

#[test]
fn acc03b_fe56_capture_xs_regression() {
    let pts = acc03b_points();
    let s = summarize("ACC-03b Fe-56 sigma(n,g) vs TALYS sigma_res", &pts, true);
    assert_within("ACC-03b regression", &s, ACC03B_RATCHET);
    let above: Vec<Point> = pts.into_iter().filter(|p| p.e >= 0.01).collect();
    let s = summarize("ACC-03b Fe-56 sigma(n,g), E_cm >= 10 keV", &above, false);
    assert_within(
        "ACC-03b regression (E_cm >= 10 keV)",
        &s,
        ACC03B_ABOVE_10KEV_RATCHET,
    );
}

#[test]
#[ignore = "ACC-03b not yet met: measured max rel err 4.9e2 at E_cm=9.8 keV (median 6.3e-1; 1.1e1 at 19.6 MeV); \
            missing compound elastic below 10 keV, NLD/GSF/WFC/competing-channel model differences, see module docs"]
fn acc03b_fe56_capture_xs_srs_target() {
    let s = summarize(
        "ACC-03b Fe-56 sigma(n,g) vs TALYS sigma_res",
        &acc03b_points(),
        true,
    );
    assert_within("ACC-03b SRS target", &s, 1e-6);
}
