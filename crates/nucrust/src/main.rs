use clap::Parser;
use std::fs;

mod cli;
use cli::{Cli, Commands};

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Calc {
            config,
            backend,
            format,
            output,
        } => run_calc(&config, &backend, &format, &output),

        Commands::Batch { config, backend } => run_batch(&config, &backend),

        Commands::Fit { config, method } => run_fit(&config, &method),

        Commands::Info { nuclide, ripl3 } => run_info(&nuclide, ripl3.as_deref()),

        Commands::Export {
            input,
            format,
            output,
        } => run_export(&input, &format, output.as_deref()),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run_calc(
    config_path: &std::path::Path,
    backend: &str,
    formats: &[String],
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_str = fs::read_to_string(config_path)?;
    let config = nucrust_data::config::parse_config(&config_str)?;

    println!(
        "Calculating {}{} on {}{} (E: {:.3}–{:.3} MeV, {} points)",
        config.projectile,
        config.target.a,
        element_symbol(config.target.z),
        config.target.a,
        config.energy.min,
        config.energy.max,
        config.energy.points
    );
    println!("Backend: {backend}");
    println!(
        "Models: OMP={}, NLD={}, GSF={}",
        config.models.omp, config.models.nld, config.models.gsf
    );

    // Build channel and energy grid
    let nuclide = nucrust_core::Nuclide::new(config.target.z, config.target.a)?;
    let projectile = match config.projectile.as_str() {
        "n" => nucrust_core::Projectile::Neutron,
        "p" => nucrust_core::Projectile::Proton,
        "alpha" | "a" => nucrust_core::Projectile::Alpha,
        _ => {
            return Err(format!("Unknown projectile: {}", config.projectile).into());
        }
    };
    let channel = nucrust_core::Channel {
        projectile,
        target: nuclide,
        q_value: 0.0, // TODO: look up from mass table
    };

    let energies = if config.energy.spacing == "log" {
        nucrust_core::EnergyGrid::logarithmic(
            config.energy.min,
            config.energy.max,
            config.energy.points,
        )?
    } else {
        nucrust_core::EnergyGrid::linear(
            config.energy.min,
            config.energy.max,
            config.energy.points,
        )?
    };

    // Compute transmission coefficients
    let numerov_config = nucrust_core::backend::NumerovConfig::default();
    let tc = nucrust::cpu_backend::cpu_transmission_coeffs(&channel, &energies, &numerov_config)?;

    println!("Transmission coefficients computed (l_max = {})", tc.l_max);

    // Output
    fs::create_dir_all(output_dir)?;
    for fmt in formats {
        match fmt.as_str() {
            "json" => {
                let path = output_dir.join("result.json");
                let output = serde_json::json!({
                    "target": { "z": config.target.z, "a": config.target.a },
                    "projectile": config.projectile,
                    "l_max": tc.l_max,
                    "n_energies": energies.len(),
                    "energy_min": config.energy.min,
                    "energy_max": config.energy.max,
                });
                fs::write(&path, serde_json::to_string_pretty(&output)?)?;
                println!("Written: {}", path.display());
            }
            "table" => {
                let path = output_dir.join("transmission.tsv");
                let mut lines = vec!["# E(MeV)\tT_l=0".to_string()];
                for (i, &e) in energies.as_slice().iter().enumerate() {
                    let t0 = if tc.data.len() > i { tc.data[i] } else { 0.0 };
                    lines.push(format!("{:.6}\t{:.6e}", e, t0));
                }
                fs::write(&path, lines.join("\n"))?;
                println!("Written: {}", path.display());
            }
            _ => {
                println!("Format '{}' not yet supported", fmt);
            }
        }
    }

    Ok(())
}

fn run_batch(
    config_path: &std::path::Path,
    backend: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Batch calculation from {:?} (backend: {backend})",
        config_path
    );
    println!("Not yet implemented.");
    Ok(())
}

fn run_fit(config_path: &std::path::Path, method: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("R-matrix fitting from {:?} (method: {method})", config_path);
    println!("Not yet implemented.");
    Ok(())
}

fn run_info(
    nuclide_str: &str,
    _ripl3_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Nuclide info: {nuclide_str}");
    // Parse nuclide name (e.g., "Fe56" → Z=26, A=56)
    println!("(Nuclide lookup not yet implemented)");
    Ok(())
}

fn run_export(
    input: &std::path::Path,
    format: &str,
    output: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Export {:?} → format '{format}'", input);
    if let Some(out) = output {
        println!("Output: {:?}", out);
    }
    println!("Not yet implemented.");
    Ok(())
}

/// Simple Z → element symbol lookup.
fn element_symbol(z: u16) -> &'static str {
    const SYMBOLS: &[&str] = &[
        "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
        "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
        "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
        "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu",
    ];
    SYMBOLS.get(z as usize).unwrap_or(&"??")
}
