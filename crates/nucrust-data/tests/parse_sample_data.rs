//! Integration tests: verify that the sample RIPL-3 and REACLIB data files
//! in `data/ripl3_sample/` and `data/reaclib_sample/` can be parsed correctly.

use nucrust_core::spin::Parity;
use nucrust_data::reaclib::parse_reaclib;
use nucrust_data::ripl3::*;

/// Path to project root (2 levels up from crate root)
fn project_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read_sample(rel_path: &str) -> String {
    let path = project_root().join(rel_path);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e))
}

#[test]
fn parse_sample_levels() {
    let data = read_sample("data/ripl3_sample/levels/z026.dat");
    let isotopes = parse_discrete_levels(&data).unwrap();

    // Should contain Fe-56 and Fe-57
    assert_eq!(isotopes.len(), 2, "Expected 2 isotopes (Fe-56, Fe-57)");

    let fe56 = &isotopes[0];
    assert_eq!(fe56.nuclide.z(), 26);
    assert_eq!(fe56.nuclide.a(), 56);
    assert_eq!(fe56.n_levels, 20);
    assert!((fe56.sn.unwrap() - 11.1974).abs() < 1e-3);
    assert!((fe56.sp.unwrap() - 8.8841).abs() < 1e-3);
    // Ground state: 0+
    assert!((fe56.levels[0].energy - 0.0).abs() < 1e-6);
    assert!((fe56.levels[0].spin.unwrap() - 0.0).abs() < 1e-6);
    assert_eq!(fe56.levels[0].parity, Some(Parity::Positive));
    // First excited: 2+ at 0.8468 MeV
    assert!((fe56.levels[1].energy - 0.8468).abs() < 1e-4);
    assert!((fe56.levels[1].spin.unwrap() - 2.0).abs() < 1e-6);

    let fe57 = &isotopes[1];
    assert_eq!(fe57.nuclide.z(), 26);
    assert_eq!(fe57.nuclide.a(), 57);
    assert_eq!(fe57.n_levels, 30);
    assert!((fe57.sn.unwrap() - 7.6464).abs() < 1e-3);
    // Ground state: 1/2-
    assert!((fe57.levels[0].spin.unwrap() - 0.5).abs() < 1e-6);
    assert_eq!(fe57.levels[0].parity, Some(Parity::Negative));
}

#[test]
fn parse_sample_masses() {
    let data = read_sample("data/ripl3_sample/masses/mass-frdm.dat");
    let masses = parse_mass_table(&data).unwrap();

    assert!(masses.len() >= 15, "Expected at least 15 mass entries");

    // Find Fe-56
    let fe56 = masses
        .iter()
        .find(|m| m.nuclide.z() == 26 && m.nuclide.a() == 56)
        .unwrap();
    assert!((fe56.mass_excess - (-60.601)).abs() < 0.01);

    // Find Fe-57
    let fe57 = masses
        .iter()
        .find(|m| m.nuclide.z() == 26 && m.nuclide.a() == 57)
        .unwrap();
    assert!((fe57.mass_excess - (-60.176)).abs() < 0.01);

    // Find light particles
    let h1 = masses
        .iter()
        .find(|m| m.nuclide.z() == 1 && m.nuclide.a() == 1)
        .unwrap();
    assert!((h1.mass_excess - 7.289).abs() < 0.01);

    let he4 = masses
        .iter()
        .find(|m| m.nuclide.z() == 2 && m.nuclide.a() == 4)
        .unwrap();
    assert!((he4.mass_excess - 2.425).abs() < 0.01);
}

#[test]
fn parse_sample_omp() {
    let data = read_sample("data/ripl3_sample/optical/om-parameter-u.dat");
    let sets = parse_omp_database(&data).unwrap();

    assert_eq!(sets.len(), 2, "Expected 2 OMP sets (neutron + proton)");

    // Neutron set (set_number < 4000)
    let neutron_set = &sets[0];
    assert_eq!(neutron_set.set_number, 2405);
    assert!(neutron_set.is_neutron());
    assert!(neutron_set.is_applicable(26, 56, 10.0));

    // Proton set (set_number >= 4000)
    let proton_set = &sets[1];
    assert_eq!(proton_set.set_number, 4416);
    assert!(proton_set.is_proton());
    assert!(proton_set.is_applicable(26, 56, 10.0));
}

#[test]
fn parse_sample_level_density() {
    let data = read_sample("data/ripl3_sample/densities/total/level-densities-hfb.dat");
    let params = parse_level_density_params(&data).unwrap();

    assert!(params.len() >= 10, "Expected at least 10 NLD entries");

    let fe56 = params
        .iter()
        .find(|p| p.nuclide.z() == 26 && p.nuclide.a() == 56)
        .unwrap();
    assert!((fe56.a - 6.210).abs() < 0.001);
    assert!((fe56.delta - (-0.520)).abs() < 0.001);

    let fe57 = params
        .iter()
        .find(|p| p.nuclide.z() == 26 && p.nuclide.a() == 57)
        .unwrap();
    assert!((fe57.a - 6.350).abs() < 0.001);
}

#[test]
fn parse_sample_gdr() {
    let data = read_sample("data/ripl3_sample/gamma/gdr-parameters.dat");
    let params = parse_gdr_params(&data).unwrap();

    assert!(params.len() >= 10);

    let fe56 = params
        .iter()
        .find(|p| p.nuclide.z() == 26 && p.nuclide.a() == 56)
        .unwrap();
    assert!((fe56.e_gdr1 - 16.36).abs() < 0.01);
    assert!((fe56.sigma_gdr1 - 136.0).abs() < 0.1);
    assert!(fe56.e_gdr2.is_none()); // single component

    let fe57 = params
        .iter()
        .find(|p| p.nuclide.z() == 26 && p.nuclide.a() == 57)
        .unwrap();
    assert!(fe57.e_gdr2.is_some()); // double component
}

#[test]
fn parse_sample_gsf_table() {
    let data = read_sample("data/ripl3_sample/gamma/gamma-strength/z026.dat");
    let tables = parse_gsf_table(&data, 26).unwrap();

    assert_eq!(tables.len(), 2, "Expected GSF tables for Fe-56 and Fe-57");
    assert_eq!(tables[0].nuclide.a(), 56);
    assert_eq!(tables[0].entries.len(), 10);
    assert!(tables[0].entries[0].f_e1 > 0.0);

    assert_eq!(tables[1].nuclide.a(), 57);
    assert_eq!(tables[1].entries.len(), 10);
}

#[test]
fn parse_sample_hfb_density() {
    let data = read_sample("data/ripl3_sample/densities/microscopic/z026.dat");
    let tables = parse_hfb_density_table(&data, 26).unwrap();

    assert_eq!(tables.len(), 2, "Expected HFB tables for Fe-56 and Fe-57");
    assert_eq!(tables[0].nuclide.a(), 56);
    assert_eq!(tables[0].entries.len(), 24);
    assert_eq!(tables[1].nuclide.a(), 57);
    assert_eq!(tables[1].entries.len(), 24);
}

#[test]
fn parse_sample_resonances() {
    let data = read_sample("data/ripl3_sample/resonances/resonances0.dat");
    let entries = parse_resonances(&data).unwrap();

    assert!(entries.len() >= 5);

    let fe57 = entries
        .iter()
        .find(|e| e.nuclide.z() == 26 && e.nuclide.a() == 57)
        .unwrap();
    assert!(fe57.d0.is_some());
    assert!(fe57.s0.is_some());
}

#[test]
fn parse_sample_shell_corrections() {
    let data = read_sample("data/ripl3_sample/shellcorrections/shell-corrections.dat");
    let entries = parse_shell_corrections(&data).unwrap();

    assert!(entries.len() >= 10);

    let fe56 = entries
        .iter()
        .find(|e| e.nuclide.z() == 26 && e.nuclide.a() == 56)
        .unwrap();
    assert!((fe56.shell_correction - (-3.47)).abs() < 0.01);

    // U-238 should have non-zero deformation
    let u238 = entries
        .iter()
        .find(|e| e.nuclide.z() == 92 && e.nuclide.a() == 238)
        .unwrap();
    assert!(u238.beta2.is_some());
    assert!(u238.beta2.unwrap() > 0.2);
}

#[test]
fn parse_sample_reaclib() {
    let data = read_sample("data/reaclib_sample/reaclib_sample.dat");
    let entries = parse_reaclib(&data).unwrap();

    assert!(
        entries.len() >= 5,
        "Expected at least 5 REACLIB entries, got {}",
        entries.len()
    );

    // Find n + fe56 -> fe57 + g
    let ng = entries
        .iter()
        .find(|e| {
            e.reactants.contains(&"n".to_string())
                && e.reactants.contains(&"fe56".to_string())
                && e.products.contains(&"fe57".to_string())
        })
        .expect("Should find n+fe56 -> fe57+g entry");
    assert!((ng.q_value - 7.646).abs() < 0.01);

    // Evaluate rate at T9 = 1.0 (should be a reasonable positive number)
    let rate = ng.evaluate(1.0);
    assert!(rate > 0.0, "Rate should be positive at T9=1");
    assert!(rate.is_finite(), "Rate should be finite");
}
