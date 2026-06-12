# RIPL-3

The [Reference Input Parameter Library](https://www-nds.iaea.org/RIPL-3/) (RIPL-3) is the standard nuclear data library maintained by the IAEA, providing input parameters for nuclear reaction calculations.

## Supported Data Types

nucrust parses the following RIPL-3 data files:

| File | Parser | Description |
|------|--------|-------------|
| `levels/z*.dat` | `parse_discrete_levels()` | Discrete nuclear levels and gamma transitions |
| `masses/mass-frdm95.dat` | `parse_mass_table()` | Nuclear mass table (FRDM 1995) |
| `optical/om-parameter-u.dat` | `parse_omp_database()` | Optical model parameters |
| `densities/level-density-param.dat` | `parse_level_density_params()` | Level density parameters (a, Δ, σ) |
| `densities/total/level-densities-hfb/z*.tab` | `parse_hfb_density_table()` | HFB level density tables |
| `gamma/gdr-parameters.dat` | `parse_gdr_params()` | Giant dipole resonance parameters |
| `gamma/strength/z*.dat` | `parse_gsf_table()` | Gamma strength function tables |
| `resonances/resonances.dat` | `parse_resonances()` | Resonance parameters |
| `fission/fission-barrier-exp.dat` | `parse_fission_barriers()` | Fission barrier heights |
| `masses/shell-corrections.dat` | `parse_shell_corrections()` | Shell correction energies |

## Usage

```rust,ignore
use nucrust::nucrust_data::ripl3;

// Parse discrete levels for Z=26
let data = std::fs::read_to_string("data/ripl3/levels/z026.dat")?;
let isotopes = ripl3::parse_discrete_levels(&data)?;

for iso in &isotopes {
    println!("{}: {} levels", iso.nuclide, iso.levels.len());
}
```

## Data Directory Structure

nucrust expects RIPL-3 data in the standard directory layout:

```text
ripl3/
├── levels/
│   ├── z001.dat
│   ├── z002.dat
│   └── ...
├── masses/
│   └── mass-frdm95.dat
├── optical/
│   └── om-parameter-u.dat
├── densities/
│   └── ...
└── gamma/
    └── ...
```

Set the path via `models.ripl3_path` in TOML configuration.
